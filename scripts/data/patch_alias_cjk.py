#!/usr/bin/env python3
"""Final cleanup: replace residual Chinese entity aliases with canonical English names.

Deterministic alias map derived from data/wiki_entities/entities.jsonl (label_zh -> title),
plus one LLM re-translation for a field that is still fully Chinese.
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from translate_synth_claims import pin_host, request_llm  # noqa

EXPL_ONLY = """Translate this Chinese fact-checking explanation into English.
Never change, add or drop any number, date, name or negation. Keep entity names in English.
Output JSON only: {{"explanation": "..."}}

explanation: 「{explanation}」"""

CJK = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')

# zh alias -> canonical English (entity title); verified against entities.jsonl
ALIAS = {
    "速可達硬漢": "ASAP Rocky",       # Q129910
    "騙子·貝茨": "Swizz Beatz",        # Q240523
    "騙子·贝茨": "Swizz Beatz",
    "都會小子": "Metro Boomin",        # Q16235273
    "娜語錄第二章": "Confessions II",  # Q139373325
    "非常盗": "Now You See Me",        # zh rendering used in a Now You See Me record
}
OUT = Path("data/synthesized_claims_final_en.jsonl")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--base-url", default="http://your-llm-endpoint/v1")
    ap.add_argument("--model", default="your-model")
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--pin-host", default=None)
    ap.add_argument("--pin-ip", default=None)
    args = ap.parse_args(argv)
    if args.pin_host and args.pin_ip:
        pin_host(args.pin_host, args.pin_ip)

    rows = [json.loads(l) for l in args.output.read_text(encoding="utf-8").splitlines() if l.strip()]
    touched = retranslated = 0
    for r in rows:
        hit = False
        for field in ("claim", "explanation"):
            v = r.get(field) or ""
            for zh, en in ALIAS.items():
                if zh in v:
                    v = v.replace(zh, en)
                    hit = True
            r[field] = v
        # any remaining Chinese prose -> re-translate from the Chinese source
        expl = r.get("explanation") or ""
        if CJK.search(expl):
            zh_expl = expl
            for a, b in ALIAS.items():
                zh_expl = zh_expl.replace(a, b)
            try:
                res = request_llm(args.base_url, args.model,
                                  EXPL_ONLY.format(explanation=zh_expl[:1500]), args.timeout, args.retries)
                v = (res or {}).get("explanation")
                if isinstance(v, str) and v.strip() and not CJK.search(v):
                    r["explanation"] = v.strip()
                    retranslated += 1
                    hit = True
            except Exception as exc:
                print("retranslate failed for %s: %s" % (r.get("id"), exc), flush=True)
        if hit:
            touched += 1
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.output)
    residual = sum(1 for r in rows
                   if CJK.search(r.get("claim") or "") or CJK.search(r.get("explanation") or ""))
    print(json.dumps({"status": "ok", "touched": touched, "retranslated": retranslated,
                      "records_with_cjk_left": residual}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
