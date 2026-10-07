#!/usr/bin/env python3
"""Select the final entity set (>=10k goal) from the pageviews census pool.

Reads one or more census files (wikipedia_entities.py output), groups entities
by category, sorts each group by popularity (monthly views), applies
per-category quotas and optional supplement files (category-harvested
entities), and writes the final entities.jsonl for wikipedia_facts.py.

Usage:
  python3 scripts/data/select_entities.py \
      --census data/wiki_entities/census.jsonl \
      --supplement data/wiki_entities/supplement.jsonl \
      --output data/wiki_entities/entities.jsonl \
      --quota-person 5000 --quota-place 1600 --quota-organization 1500 \
      --quota-work 1000 --quota-event 700 --quota-country 250
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

CATEGORIES = ("person", "place", "organization", "work", "event", "country")


def load(paths):
    rows = []
    for path in paths or []:
        path = Path(path)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--census", type=Path, action="append", required=True)
    parser.add_argument("--supplement", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quota-person", type=int, default=5000)
    parser.add_argument("--quota-place", type=int, default=1600)
    parser.add_argument("--quota-organization", type=int, default=1500)
    parser.add_argument("--quota-work", type=int, default=1000)
    parser.add_argument("--quota-event", type=int, default=700)
    parser.add_argument("--quota-country", type=int, default=250)
    args = parser.parse_args(argv)
    quotas = {"person": args.quota_person, "place": args.quota_place,
              "organization": args.quota_organization, "work": args.quota_work,
              "event": args.quota_event, "country": args.quota_country}
    pool = {}
    for row in load(args.census) + load(args.supplement):
        qid = row.get("qid")
        if not qid or row.get("category") not in quotas:
            continue
        if qid not in pool:
            pool[qid] = row
        else:  # keep the entry with the higher recorded popularity
            if row.get("monthly_views_max", 0) > pool[qid].get("monthly_views_max", 0):
                pool[qid] = row
    groups = defaultdict(list)
    for row in pool.values():
        groups[row["category"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: -r.get("monthly_views_max", 0))
    selected, shortfalls = [], {}
    for category in CATEGORIES:
        rows = groups.get(category, [])
        take = rows[:quotas[category]]
        selected.extend(take)
        shortfalls[category] = max(0, quotas[category] - len(rows))
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(out)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "pool_by_category": {c: len(groups.get(c, [])) for c in CATEGORIES},
              "quotas": quotas, "selected": len(selected),
              "selected_by_category": {c: min(quotas[c], len(groups.get(c, []))) for c in CATEGORIES},
              "shortfalls": shortfalls, "output": str(out)}
    Path(str(out) + ".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps({"status": "ok", **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
