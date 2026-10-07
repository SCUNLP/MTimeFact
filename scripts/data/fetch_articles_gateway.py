#!/usr/bin/env python3
"""Fetch obscure-entity articles through the api.wikimedia.org gateway.

The Action API caps exchars at 1200 chars (and was rate-limiting us), so this
fetches full wikitext from the gateway, converts it to plain text and writes the
same per-article cache files wikipedia_facts.py reads.
"""
import argparse, hashlib, json, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clash_rotate import NodeRotator  # noqa: E402

UA = "MTimeFactResearch/0.1 (academic dataset construction; contact: research)"


def wikitext_to_text(wt):
    t = wt
    t = re.sub(r'<!--.*?-->', '', t, flags=re.S)
    t = re.sub(r'<ref[^>]*/>', '', t)
    t = re.sub(r'<ref[^>]*>.*?</ref>', '', t, flags=re.S)
    t = re.sub(r'<gallery[^>]*>.*?</gallery>', '', t, flags=re.S)
    t = re.sub(r'\{\|.*?\|\}', '', t, flags=re.S)
    for _ in range(8):
        t = re.sub(r'\{\{[^{}]*\}\}', '', t)
    t = re.sub(r'\[\[(?:File|Image|Category|Wikipedia|Template|Help|Portal|Module|Special):[^\]]*\]\]',
               '', t, flags=re.I)
    t = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]', r'\1', t)
    t = re.sub(r'\[(?:https?://\S+)\s+([^\]]*)\]', r'\1', t)
    t = re.sub(r'\[(?:https?://\S+)\]', '', t)
    t = re.sub(r'<[^>]+>', '', t)
    t = re.sub(r"'{2,5}", '', t)
    t = re.sub(r'^=+\s*(.*?)\s*=+\s*$', r'\1', t, flags=re.M)
    t = re.sub(r'^[*#:;]+\s*', '', t, flags=re.M)
    t = re.sub(r'\n{3,}', '\n\n', t)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    t = re.sub(r'\{\{[^{}]*$', '', t, flags=re.M)
    return t.strip()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=Path, default=Path("data/wiki_entities/entities_obscure.jsonl"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache/wiki"))
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--chars", type=int, default=7000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=25)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-chars", type=int, default=0,
                    help="treat cached extracts shorter than this as pending (re-fetch in full)")
    ap.add_argument("--clash-controller", default=None,
                    help="ClashX controller URL for node rotation (optional)")
    args = ap.parse_args(argv)

    rotator = NodeRotator(controller=args.clash_controller, rotate_every=10 ** 9) if args.clash_controller else None
    lock = threading.Lock()

    def rotate_once():
        with lock:
            if rotator is None or time.time() - rotate_once.last < 6:
                return
            rotate_once.last = time.time()
            try:
                rotator.rotate()
            except Exception:
                pass
    rotate_once.last = 0.0

    entities = [json.loads(l) for l in args.entities.read_text(encoding="utf-8").splitlines() if l.strip()]
    extract_dir = args.cache_dir / "extracts"
    extract_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(title):
        key = hashlib.sha256(("extract:en.wikipedia.org:%s:%d" % (title, args.chars)).encode()).hexdigest()[:24]
        return extract_dir / (key + ".json")

    def valid(title):
        p = cache_path(title)
        if not p.exists():
            return False
        try:
            page = (json.loads(p.read_text(encoding="utf-8")).get("query") or {}).get("pages") or [{}]
        except ValueError:
            return False
        page = page[0]
        if page.get("missing"):
            return True
        text = (page.get("extract") or "").strip()
        return bool(text) and len(text) >= args.min_chars

    pending = [e for e in entities if not valid(e["title"])]
    if args.limit:
        pending = pending[:args.limit]
    print(json.dumps({"entities": len(entities), "pending": len(pending)}), flush=True)

    stats = {"fetched": 0, "missing": 0, "failed": 0}

    def fetch_one(entity):
        title = entity["title"]
        url = "https://api.wikimedia.org/core/v1/wikipedia/%s/page/%s" % (
            args.lang, quote(title.replace(" ", "_"), safe=""))
        opener = build_opener(ProxyHandler({"http": args.proxy, "https": args.proxy}) if args.proxy else {})
        req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
        for attempt in range(args.retries):
            try:
                with opener.open(req, timeout=args.timeout) as resp:
                    data = json.load(resp)
                text = wikitext_to_text(data.get("source") or "")
                latest = data.get("latest") or {}
                payload = {"query": {"pages": [{
                    "title": data.get("title") or title,
                    "extract": text,
                    "revisions": [{"revid": latest.get("id"), "timestamp": latest.get("timestamp")}],
                }]}}
                cache_path(title).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                stats["fetched"] += 1
                return
            except HTTPError as exc:
                if exc.code == 404:
                    payload = {"query": {"pages": [{"title": title, "missing": True}]}}
                    cache_path(title).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                    stats["missing"] += 1
                    return
                if attempt == args.retries - 1:
                    stats["failed"] += 1
                    print("FAILED %s: %s" % (title, str(exc)[:70]), file=sys.stderr, flush=True)
                    return
                rotate_once(); time.sleep(1.0 + attempt)
            except (URLError, TimeoutError, OSError, ValueError) as exc:
                if attempt == args.retries - 1:
                    stats["failed"] += 1
                    print("FAILED %s: %s" % (title, str(exc)[:70]), file=sys.stderr, flush=True)
                    return
                rotate_once(); time.sleep(1.0 + attempt)

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch_one, pending))
    print(json.dumps({**stats, "minutes": round((time.time() - started) / 60, 1),
                      "still_pending": sum(1 for e in entities if not valid(e["title"]))}), flush=True)
    if rotator:
        rotator.restore()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
