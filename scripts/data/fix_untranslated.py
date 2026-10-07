#!/usr/bin/env python3
"""Retry records whose claim/explanation stayed Chinese after batch translation.

Falls back to claim-only / explanation-only calls when a combined call fails.
Rewrites the output file in place with the fixed records.
"""
import argparse, json, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from translate_synth_claims import PROMPT, build_items, pin_host, request_llm

CJK = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')

CLAIM_ONLY = """Translate this Chinese fact-checking claim into English.
Never change, add or drop any number, date, name or negation. Keep entity names in English.
Output JSON only: {{"claim": "..."}}

claim: 「{claim}」"""

EXPL_ONLY = """Translate this Chinese fact-checking explanation into English.
Never change, add or drop any number, date, name or negation. Keep entity names in English.
Output JSON only: {{"explanation": "..."}}

explanation: 「{explanation}」"""

STRIP_CLAIM = """Rewrite this English sentence so that it contains no Chinese characters at all.
Render every Chinese name, title or term in its standard English form (known English name, or pinyin).
Keep meaning, numbers, dates and all other names unchanged.
Output JSON only: {{"claim": "..."}}

claim: {claim}"""

STRIP_EXPL = """Rewrite this English explanation so that it contains no Chinese characters at all.
Render every Chinese name, title or term in its standard English form (known English name, or pinyin).
Keep meaning, numbers, dates and all other names unchanged.
Output JSON only: {{"explanation": "..."}}

explanation: {explanation}"""


def cjk_ratio(text):
    text = text or ""
    return len(CJK.findall(text)) / max(len(text), 1)


def try_call(base_url, model, prompt, timeout, retries):
    try:
        r = request_llm(base_url, model, prompt, timeout, retries)
        return r if isinstance(r, dict) else None
    except Exception:
        return None


def fix_record(rec, args, tag=""):
    import time as _t
    t0 = _t.time()
    claim, explanation = rec.get("claim") or "", rec.get("explanation") or ""
    need_claim, need_expl = cjk_ratio(claim) > args.threshold, cjk_ratio(explanation) > args.threshold

    # step A: combined single-record call (direct, bypasses any bad cached result)
    if need_claim or need_expl:
        ta = _t.time()
        r = try_call(args.base_url, args.model,
                     PROMPT.format(count=1, items=build_items([rec])), args.timeout, args.retries)
        print("  %s stepA %.1fs ok=%s" % (tag, _t.time()-ta, bool(r)), flush=True)
        row = ((r or {}).get("results") or [{}])[0]
        if need_claim and row.get("claim"):
            claim = row["claim"].strip()
        if need_expl and row.get("explanation"):
            explanation = row["explanation"].strip()
        need_claim, need_expl = cjk_ratio(claim) > args.threshold, cjk_ratio(explanation) > args.threshold

    # step B: field-separated calls
    if need_claim:
        ta = _t.time()
        r = try_call(args.base_url, args.model, CLAIM_ONLY.format(claim=(rec.get("claim") or "")[:1500]),
                     args.timeout, args.retries)
        print("  %s stepB-claim %.1fs ok=%s" % (tag, _t.time()-ta, bool(r)), flush=True)
        v = (r or {}).get("claim")
        if isinstance(v, str) and v.strip():
            claim = v.strip()
    if need_expl:
        ta = _t.time()
        r = try_call(args.base_url, args.model, EXPL_ONLY.format(explanation=(rec.get("explanation") or "")[:1500]),
                     args.timeout, args.retries)
        print("  %s stepB-expl %.1fs ok=%s" % (tag, _t.time()-ta, bool(r)), flush=True)
        v = (r or {}).get("explanation")
        if isinstance(v, str) and v.strip():
            explanation = v.strip()

    # step C: strip any remaining Chinese fragment (names/titles) into English form
    if args.strip_any:
        if CJK.search(claim):
            ta = _t.time()
            r = try_call(args.base_url, args.model, STRIP_CLAIM.format(claim=claim[:2000]),
                         args.timeout, args.retries)
            print("  %s stepC-claim %.1fs ok=%s" % (tag, _t.time()-ta, bool(r)), flush=True)
            v = (r or {}).get("claim")
            if isinstance(v, str) and v.strip() and not CJK.search(v):
                claim = v.strip()
        if CJK.search(explanation):
            ta = _t.time()
            r = try_call(args.base_url, args.model, STRIP_EXPL.format(explanation=explanation[:2000]),
                         args.timeout, args.retries)
            print("  %s stepC-expl %.1fs ok=%s" % (tag, _t.time()-ta, bool(r)), flush=True)
            v = (r or {}).get("explanation")
            if isinstance(v, str) and v.strip() and not CJK.search(v):
                explanation = v.strip()
    print("  %s done %.1fs clean=%s" % (tag, _t.time()-t0,
          not CJK.search(claim) and not CJK.search(explanation)), flush=True)
    return dict(rec, claim=claim, explanation=explanation)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("data/synthesized_claims_final_en.jsonl"))
    ap.add_argument("--base-url", default="http://your-llm-endpoint/v1")
    ap.add_argument("--model", default="your-model")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--pin-host", default=None)
    ap.add_argument("--pin-ip", default=None)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--strip-any", action="store_true",
                    help="also clean up records that only contain small Chinese fragments (names/titles)")
    args = ap.parse_args(argv)
    if args.pin_host and args.pin_ip:
        pin_host(args.pin_host, args.pin_ip)

    def needs_fix(r):
        if args.strip_any:
            return bool(CJK.search(r.get("claim") or "") or CJK.search(r.get("explanation") or ""))
        return (cjk_ratio(r.get("claim")) > args.threshold
                or cjk_ratio(r.get("explanation")) > args.threshold)

    rows = [json.loads(l) for l in args.output.read_text(encoding="utf-8").splitlines() if l.strip()]
    todo = [i for i, r in enumerate(rows) if needs_fix(r)]
    print(json.dumps({"records": len(rows), "to_fix": len(todo)}), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fix_record, rows[i], args, rows[i].get("id", str(i))): i for i in todo}
        done = 0
        for f, i in futures.items():
            rows[i] = f.result()
            done += 1
            if done % 20 == 0 or done == len(todo):
                clean = sum(1 for j in todo if not needs_fix(rows[j]))
                print("fixed %d/%d (clean %d)" % (done, len(todo), clean), flush=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.output)
    residual = sum(1 for i in todo if needs_fix(rows[i]))
    print(json.dumps({"status": "ok", "fixed_clean": len(todo)-residual, "residual": residual}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
