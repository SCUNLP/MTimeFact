#!/usr/bin/env python3
"""Merge the three English multi-hop datasets into one file.

- main set records get difficulty="standard" (hard/obscure already carry theirs)
- exact duplicate claim texts (whitespace-normalized) are dropped, first wins
- id uniqueness is verified across files
Writes the merged EN file plus the matching CN file (same ids, same order).
"""
import json, re, sys
from collections import Counter
from pathlib import Path

SOURCES = [
    ("data/synthesized_claims_final_en.jsonl", "data/synthesized_claims_final.jsonl", "standard"),
    ("data/synthesized_hard_en.jsonl", "data/synthesized_hard.jsonl", "hard"),
    ("data/synthesized_obscure_en.jsonl", "data/synthesized_obscure.jsonl", "obscure"),
]
OUT_EN = Path("data/synthesized_claims_all_en.jsonl")
OUT_CN = Path("data/synthesized_claims_all.jsonl")


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main():
    merged_en, merged_cn = [], []
    stats = Counter(); seen_ids = set(); seen_claims = set(); dups = 0
    per_file = []
    for en_path, cn_path, default_difficulty in SOURCES:
        en_rows = load(en_path)
        cn_rows = load(cn_path)
        by_id = {r["id"]: r for r in cn_rows}
        kept = dropped_dup = 0
        for r in en_rows:
            key = re.sub(r"\s+", "", (r.get("claim") or ""))
            # same id (or whitespace-normalized same text) already merged from an
            # earlier file: cross-file duplicate on shared paths, keep the first
            if r["id"] in seen_ids or key in seen_claims:
                dropped_dup += 1; dups += 1
                continue
            seen_claims.add(key); seen_ids.add(r["id"])
            r.setdefault("difficulty", default_difficulty)
            if r["difficulty"] is None:
                r["difficulty"] = default_difficulty
            merged_en.append(r)
            cn = by_id.get(r["id"])
            if cn is None:
                raise SystemExit("CN record missing for id %s" % r["id"])
            cn.setdefault("difficulty", default_difficulty)
            if cn["difficulty"] is None:
                cn["difficulty"] = default_difficulty
            merged_cn.append(cn)
            kept += 1
        per_file.append({"en": en_path, "records": len(en_rows), "kept": kept,
                         "claim_dups_dropped": dropped_dup})
        stats[default_difficulty] += kept
        print(json.dumps(per_file[-1]), flush=True)

    for out, rows in ((OUT_EN, merged_en), (OUT_CN, merged_cn)):
        tmp = out.with_suffix(out.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as h:
            for r in rows:
                h.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(out)

    label = Counter(r["label"] for r in merged_en)
    hop = Counter(r["hop_count"] for r in merged_en)
    pert = Counter(r.get("perturbation") or "positive" for r in merged_en)
    tw = sum(1 for r in merged_en if (r.get("time_window") or {}).get("start"))
    report = {
        "total_records": len(merged_en), "claim_dups_dropped": dups,
        "by_difficulty": dict(stats), "by_label": {"true": label[True], "false": label[False]},
        "by_hop": dict(sorted(hop.items(), key=lambda x: str(x[0]))),
        "by_perturbation": dict(pert.most_common()),
        "time_window_coverage": tw, "per_file": per_file,
        "outputs": [str(OUT_EN), str(OUT_CN)],
    }
    Path(str(OUT_EN) + ".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
