#!/usr/bin/env python3
"""Parallel article fetcher that fills the same cache wikipedia_facts.py reads.

Serial fetching there is fragile (60s timeouts x 5 retries per dead node).
This fetcher uses a thread pool, short timeouts and shared node rotation; it
writes identical cache entries so the facts pipeline's fetch phase becomes a
pure cache replay.
"""
import argparse, hashlib, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clash_rotate import NodeRotator  # noqa: E402

UA = "MTimeFactResearch/0.1 (academic dataset construction; contact: research)"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=Path, default=Path("data/wiki_entities/entities_obscure.jsonl"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache/wiki"))
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--api-host", default="en.wikipedia.org")
    ap.add_argument("--chars", type=int, default=7000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=15)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--clash-controller", default=None,
                    help="ClashX controller URL for node rotation (optional)")
    args = ap.parse_args(argv)

    rotator = None
    if not args.no_rotate and args.clash_controller:
        rotator = NodeRotator(controller=args.clash_controller, rotate_every=10 ** 9)
    elif not args.no_rotate:
        print("node rotation disabled: no --clash-controller given", flush=True)
    lock = threading.Lock()
    stats = {"fetched": 0, "cached": 0, "failed": 0, "rotations": 0}

    entities = [json.loads(l) for l in args.entities.read_text(encoding="utf-8").splitlines() if l.strip()]
    extract_dir = args.cache_dir / "extracts"
    extract_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(title):
        key = hashlib.sha256(("extract:%s:%s:%d" % (args.api_host, title, args.chars)).encode()).hexdigest()[:24]
        return extract_dir / (key + ".json")

    def rotate_once():
        with lock:
            if time.time() - stats.get("last_rotate", 0) < 5:
                return
            stats["last_rotate"] = time.time()
            stats["rotations"] += 1
            if rotator:
                try:
                    rotator.rotate()
                except Exception:
                    pass

    def fetch_one(entity):
        """One title per request: exchars (long extracts) requires exlimit=1."""
        title = entity["title"]
        path = cache_path(title)
        params = urlencode({"action": "query", "format": "json", "formatversion": "2",
                            "prop": "extracts|revisions", "explaintext": "1",
                            "exchars": args.chars, "exlimit": "1",
                            "rvprop": "ids|timestamp", "rvslots": "main",
                            "redirects": "1", "titles": title})
        url = "https://%s/w/api.php?%s" % (args.api_host, params)
        opener = build_opener(ProxyHandler({"http": args.proxy, "https": args.proxy}) if args.proxy else {})
        req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
        for attempt in range(args.retries):
            try:
                with opener.open(req, timeout=args.timeout) as resp:
                    data = json.load(resp)
                path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                stats["fetched"] += 1
                return
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                if attempt == args.retries - 1:
                    stats["failed"] += 1
                    print("FAILED %s: %s" % (title, str(exc)[:80]), file=sys.stderr, flush=True)
                    return
                rotate_once()
                time.sleep(0.6 + attempt * 0.7)

    # drop cache files written without an extract (batched requests return only one page)
    dropped = 0
    for e in entities:
        p = cache_path(e["title"])
        if not p.exists():
            continue
        try:
            page = (json.loads(p.read_text(encoding="utf-8")).get("query") or {}).get("pages") or [{}]
        except ValueError:
            p.unlink(); dropped += 1; continue
        page = page[0]
        if not page.get("missing") and not (page.get("extract") or "").strip():
            p.unlink(); dropped += 1
    pending = [e for e in entities if not cache_path(e["title"]).exists()]
    stats["cached"] = len(entities) - len(pending)
    print(json.dumps({"entities": len(entities), "invalid_cache_dropped": dropped,
                      "pending": len(pending)}), flush=True)
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch_one, pending))
    print(json.dumps({"entities": len(entities), **stats,
                      "minutes": round((time.time() - started) / 60, 1)}), flush=True)
    if rotator:
        rotator.restore()
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
