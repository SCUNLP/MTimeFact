#!/usr/bin/env python3
"""Repair the hard-set translation: fill in records dropped by failed batches and
re-translate records that stayed Chinese, then rewrite the output in source order.
"""
import argparse, json, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fix_untranslated import CJK, fix_record  # noqa: E402
from translate_synth_claims import pin_host  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("data/synthesized_hard.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("data/synthesized_hard_en.jsonl"))
    ap.add_argument("--base-url", default="http://your-llm-endpoint/v1")
    ap.add_argument("--model", default="your-model")
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pin-host", default=None)
    ap.add_argument("--pin-ip", default=None)
    args = ap.parse_args(argv)
    args.strip_any = True
    if args.pin_host and args.pin_ip:
        pin_host(args.pin_host, args.pin_ip)

    src = [json.loads(l) for l in args.source.read_text(encoding="utf-8").splitlines() if l.strip()]
    have = {}
    if args.output.exists():
        for l in args.output.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                have[r["id"]] = r

    def clean(r):
        return not CJK.search(r.get("claim") or "") and not CJK.search(r.get("explanation") or "")

    todo = [i for i, s in enumerate(src)
            if src[i]["id"] not in have or not clean(have[src[i]["id"]])]
    print(json.dumps({"source": len(src), "existing_ok": len(src) - len(todo), "to_repair": len(todo)}), flush=True)

    def work(i):
        sid = src[i]["id"]
        existing = have.get(sid)
        r = existing if (existing is not None and clean(existing)) else dict(src[i])
        return i, fix_record(r, args, sid)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, i) for i in todo]
        done = 0
        for f in futures:
            i, fixed = f.result()
            have[src[i]["id"]] = fixed
            done += 1
            if done % 100 == 0 or done == len(todo):
                ok = sum(1 for j in todo if clean(have.get(src[j]["id"], {})))
                print("repaired %d/%d (clean %d)" % (done, len(todo), ok), flush=True)

    rows = [have.get(s["id"], s) for s in src]
    residual = sum(1 for r in rows if not clean(r))
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.output)
    print(json.dumps({"status": "ok", "records": len(rows), "residual_cjk": residual}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
