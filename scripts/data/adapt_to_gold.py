#!/usr/bin/env python3
"""Adapt the merged multi-hop dataset into the evaluation gold format.

Output records carry the fields the four-stage evaluation harness consumes:
  id / claim / label / hop_count / time_window (full ISO days)
  reasoning_graph {nodes, edges} built from path_nodes + evidence_chain
  evidence [{id, url, text, timestamp}] built from the facts database
  reference_date (day-precision claim date, else null)
  perturbation / difficulty / method retained for slicing

Every adapted record is validated with the harness's own validators
(queries._validate_gold, retrieval._window, verdict.normalize_label).
"""
import calendar, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mtimefact.evaluation.queries import _validate_gold  # noqa: E402
from mtimefact.evaluation.retrieval import _window  # noqa: E402
from mtimefact.evaluation.verdict import normalize_label  # noqa: E402


def month_bounds(value):
    """'2014-06' -> ('2014-06-01', '2014-06-30'); year/day pass through widened."""
    if not value:
        return None, None
    if len(value) == 4:                                   # year
        return value + "-01-01", value + "-12-31"
    if len(value) == 7:                                   # month
        year, month = int(value[:4]), int(value[5:7])
        return value + "-01", "%s-%02d" % (value[:7], calendar.monthrange(year, month)[1])
    return value, value                                    # day


def edge_dates(value):
    start, end = month_bounds(value)
    return {"start": start, "end": end}


def main():
    src = Path("data/synthesized_claims_all_en.jsonl")
    out = Path("data/mtimefact_gold_en.jsonl")

    # fact database for evidence text / url / timestamp
    facts = {}
    for f in ("data/wiki_facts_enriched.jsonl", "data/wiki_facts_obscure.jsonl"):
        for line in Path(f).open(encoding="utf-8"):
            if line.strip():
                x = json.loads(line)
                facts[x["id"]] = x

    rows, failures = [], Counter()
    seen_ids = set()
    for line in src.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        rid = r["id"]
        if not rid or rid in seen_ids:
            failures["bad_or_duplicate_id"] += 1
            continue
        seen_ids.add(rid)
        if not (r.get("claim") or "").strip():
            failures["empty_claim"] += 1
            continue

        tw = r.get("time_window") or {}
        if not (tw.get("start") and tw.get("end")):
            failures["no_window"] += 1
            continue
        start, _ = month_bounds(tw["start"])
        _, end = month_bounds(tw["end"])

        nodes_label = r.get("path_nodes") or []
        chain = r.get("evidence_chain") or []
        if not nodes_label or not chain:
            failures["no_graph"] += 1
            continue
        graph = {
            "nodes": [{"id": "n%d" % i, "label": label, "type": "entity"}
                      for i, label in enumerate(nodes_label)],
            "edges": [{"id": "e%d" % i, "head": "n%d" % i, "tail": "n%d" % (i + 1),
                       "relation": e.get("relation"), **edge_dates(e.get("date"))}
                      for i, e in enumerate(chain)],
        }

        evidence, seen_facts = [], set()
        for e in chain:
            fid = e.get("fact_id")
            if not fid or fid in seen_facts:
                continue
            seen_facts.add(fid)
            db = facts.get(fid) or {}
            evidence.append({"id": fid,
                             "url": (db.get("source") or {}).get("url"),
                             "text": db.get("fact") or e.get("fact") or "",
                             "timestamp": db.get("date") or e.get("date")})
        if not evidence:                                  # fall back to the evidence string
            evidence = [{"id": rid + "-ev0", "url": None,
                         "text": r.get("evidence") or "", "timestamp": None}]

        ref = r.get("date")
        if not ref or len(str(ref)) != 10:
            ref = None

        gold = {"id": rid, "claim": r["claim"].strip(), "label": r["label"],
                "hop_count": r.get("hop_count"),
                "time_window": {"start": start, "end": end},
                "reasoning_graph": graph, "evidence": evidence,
                "reference_date": ref,
                "perturbation": r.get("perturbation"),
                "difficulty": r.get("difficulty"), "method": r.get("method")}

        # validate with the harness's own rules
        try:
            _validate_gold(gold)
            _window(gold["time_window"], "gold window")
            normalize_label(gold["label"])
        except ValueError as exc:
            failures["validation: " + str(exc)[:60]] += 1
            continue
        rows.append(gold)

    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as h:
        for g in rows:
            h.write(json.dumps(g, ensure_ascii=False) + "\n")
    tmp.replace(out)

    report = {"input": str(src), "output": str(out), "adapted": len(rows),
              "by_label": {"true": sum(1 for g in rows if g["label"]),
                           "false": sum(1 for g in rows if not g["label"])},
              "by_difficulty": dict(Counter(g["difficulty"] for g in rows)),
              "failures": dict(failures)}
    Path(str(out) + ".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
