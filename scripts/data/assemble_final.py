#!/usr/bin/env python3
"""Assemble the final claim dataset from real + LLM-generated (+ rule fallback) parts.

Deterministic selection to reach ``--target`` records:

  1. every real record from structure_claims (label_origin=original)
  2. LLM records spread round-robin over parents, truth statements first,
     then perturbation variants — so no single parent dominates
  3. rule-based perturbation variants fill any remaining gap

Global deduplication by normalized claim text.  Output carries the union
schema: claim / evidence / label (true|false) / timestamps / provenance
(``label_origin`` + ``derived_from`` + ``augmentation`` | ``generation``).

Usage:
  python3 scripts/data/assemble_final.py \
      --real data/claims_structured.jsonl \
      --generated data/claims_llm.jsonl \
      --fallback data/claims_augmented.jsonl \
      --output data/claims_final.jsonl --target 10000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def norm(value):
    return re.sub(r"\s+", "", value or "")


def load(path):
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def round_robin_by_parent(records, groups_order):
    """Yield records one per parent per round following the given method order."""
    buckets = defaultdict(lambda: defaultdict(list))
    parents = []
    seen_parents = set()
    for record in records:
        parent = record.get("derived_from") or record["id"]
        method = record.get("extraction_method", "other")
        buckets[parent][method].append(record)
        if parent not in seen_parents:
            seen_parents.add(parent)
            parents.append(parent)
    for method in groups_order:
        for parent in parents:
            queue = buckets[parent].get(method)
            if queue:
                yield queue.pop(0)
    # any method not listed (or leftovers) afterwards
    leftovers = []
    for parent in parents:
        for method, queue in buckets[parent].items():
            leftovers.extend(queue)
    yield from leftovers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--fallback", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.target < 1:
        parser.error("--target must be positive")
    real = [r for r in load(args.real) if r.get("label_origin", "original") == "original"]
    for record in real:
        record.setdefault("label_origin", "original")
        record.setdefault("label", False)
    generated = load(args.generated)
    fallback = [r for r in load(args.fallback) if r.get("label_origin") != "original"]
    selected, seen = [], set()

    def take(records):
        for record in records:
            if len(selected) >= args.target:
                return
            key = (norm(record.get("claim")), record.get("published_at") or "")
            if not key[0] or key in seen:
                continue
            seen.add(key)
            selected.append(record)

    take(real)
    if len(selected) < args.target:
        take(round_robin_by_parent(
            generated, ("llm_true_statement", "llm_perturbation")))
    if len(selected) < args.target:
        take(round_robin_by_parent(fallback, ("perturbed",)))
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(out)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": args.target, "total": len(selected),
        "shortfall": max(0, args.target - len(selected)),
        "by_label": {str(k): v for k, v in Counter(r.get("label") for r in selected).items()},
        "by_origin": {str(k): v for k, v in Counter(r.get("label_origin") for r in selected).items()},
        "by_method": {str(k): v for k, v in Counter(r.get("extraction_method") for r in selected).items()},
        "by_dataset": {str(k): v for k, v in Counter(r.get("dataset") for r in selected).items()},
        "by_column": {str(k): v for k, v in Counter(r.get("column") for r in selected).most_common()},
        "by_year": {str(k): v for k, v in sorted(Counter((r.get("published_at") or "")[:4] for r in selected).items())},
        "real_available": len(real), "generated_available": len(generated), "fallback_available": len(fallback),
        "output": str(out),
        "notes": [
            "Selection order: all real records, then LLM records round-robin (truth statements first), then rule variants as filler.",
            "label_origin distinguishes real (original), LLM-generated (llm_generated) and rule-perturbed (perturbed).",
            "LLM truth statements (label=true) are grounded in the parent evidence; perturbations remain refuted rumors (label=false).",
        ],
    }
    Path(str(out) + ".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps({"status": "ok", "total": len(selected), "by_label": report["by_label"],
                      "by_origin": report["by_origin"], "shortfall": report["shortfall"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
