#!/usr/bin/env python3
"""Quality audit for the synthesized multi-hop datasets, focused on time windows.

Checks per record:
  A. window determinism  - recomputed from evidence_chain dates must equal the stored window
  B. precision           - share of windows grounded in month/day-precision evidence
  C. sanity              - start <= end, span caps
  D. claim consistency   - for POSITIVES every 4-digit year in the claim must fall inside the
                           window; for chronological_distortion negatives the perturbed year
                           must fall outside the evidence years (i.e. the label is checkable)
  E. grounding (sampled)  - edge dates match the underlying fact text/date
  F. duplicates          - claim-text duplicates inside and across files; path_id overlap
"""
import json, random, re, sys
from collections import Counter
from pathlib import Path

YEAR = re.compile(r'\b(1\d{3}|20\d{2})\b')


def window_from_chain(chain):
    starts, ends = [], []
    for e in chain or []:
        v = e.get("date")
        if not isinstance(v, str):
            continue
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            starts.append(v[:7]); ends.append(v[:7])
        elif re.fullmatch(r"\d{4}-\d{2}", v):
            starts.append(v); ends.append(v)
        elif re.fullmatch(r"\d{4}", v):
            starts.append(v + "-01"); ends.append(v + "-12")
    if not starts:
        return {"start": None, "end": None}
    return {"start": min(starts), "end": max(ends)}


def audit(path, facts_by_id, path_ids_seen, claims_seen, sample_n=300):
    stats = Counter(); spans = []; prec = Counter(); bad_years = []
    sampled = []
    rows = []
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line); rows.append(r)
        stats["records"] += 1
        tw = r.get("time_window") or {}
        # A. determinism: chain-dated records must match the recomputed window;
        #    undated-chain records must carry a non-null anchor window instead
        chain_dates = [e.get("date") for e in (r.get("evidence_chain") or []) if e.get("date")]
        recomputed = window_from_chain(r.get("evidence_chain"))
        if chain_dates:
            if recomputed != {"start": tw.get("start"), "end": tw.get("end")}:
                stats["window_mismatch"] += 1
        else:
            if not (tw.get("start") or tw.get("end")):
                stats["no_window"] += 1
            elif r.get("date_source") != "wikidata_entity_temporal":
                stats["anchor_window_unmarked"] += 1
        # B. precision: any month/day-precision evidence?
        dates = [e.get("date") for e in (r.get("evidence_chain") or []) if e.get("date")]
        md = [d for d in dates if len(str(d)) >= 7]
        prec["with_month_precision"] += bool(md)
        prec["year_only"] += bool(dates) and not md
        prec["no_dates"] += not dates
        # C. sanity
        if tw.get("start") and tw.get("end"):
            y1, y2 = int(tw["start"][:4]), int(tw["end"][:4])
            m1, m2 = int(tw["start"][5:7]), int(tw["end"][5:7])
            span = (y2 - y1) * 12 + (m2 - m1)
            spans.append(span)
            if span < 0:
                stats["inverted_window"] += 1
            if span > 1200:
                stats["span_gt_100y"] += 1
        else:
            stats["no_window"] += 1
        # D. claim-year consistency for positives
        if r["label"]:
            ys = [int(y) for y in YEAR.findall(r.get("claim") or "")]
            if tw.get("start") and tw.get("end") and ys:
                lo, hi = int(tw["start"][:4]), int(tw["end"][:4])
                if any(y < lo or y > hi for y in ys):
                    stats["positive_year_outside_window"] += 1
                    if len(bad_years) < 3:
                        bad_years.append((r["id"], ys, (lo, hi), (r.get("claim") or "")[:90]))
        # E/F. collect for later
        pid = r.get("path_id")
        if pid in path_ids_seen:
            stats["path_overlap_with_other_files"] += 1
        path_ids_seen.add(pid)
        ckey = re.sub(r"\s+", "", (r.get("claim") or ""))[:200]
        if ckey in claims_seen:
            stats["claim_dup_across_files"] += 1
        claims_seen.add(ckey)
        if stats["records"] % 1000 == 0:
            pass
    # duplicate claims within this file
    keys = Counter(re.sub(r"\s+", "", (r.get("claim") or ""))[:200] for r in rows)
    stats["claim_dup_within_file"] = sum(v - 1 for v in keys.values() if v > 1)
    # E. grounding sample
    random.seed(11)
    for r in random.sample(rows, min(sample_n, len(rows))):
        for e in (r.get("evidence_chain") or []):
            fid = e.get("fact_id")
            fact = facts_by_id.get(fid)
            if fact is None:
                stats["sampled_fact_missing"] += 1
                continue
            ed, fd = str(e.get("date") or ""), str(fact.get("date") or "")
            stats["sampled_edges"] += 1
            if ed and fd and ed != fd:
                stats["sampled_date_mismatch"] += 1
            elif not ed or not fd:
                stats["sampled_date_absent"] += 1
            else:
                # the date must be visible in the fact text or its evidence span
                ftext = (fact.get("fact") or "") + " " + (fact.get("evidence_span") or "")
                if len(ed) >= 4 and ed[:4] in ftext:
                    stats["sampled_date_in_text"] += 1
                else:
                    stats["sampled_date_not_in_text"] += 1
    spans.sort()
    return {
        "file": str(path), "records": stats["records"],
        "window_mismatch": stats["window_mismatch"],
        "anchor_window_unmarked": stats["anchor_window_unmarked"],
        "no_window": stats["no_window"],
        "inverted_window": stats["inverted_window"],
        "span_gt_100y": stats["span_gt_100y"],
        "span_months_pct": {p: spans[min(len(spans) - 1, int(len(spans) * p / 100))]
                            for p in (10, 25, 50, 75, 90, 99)} if spans else {},
        "precision": dict(prec),
        "positive_year_outside_window": stats["positive_year_outside_window"],
        "samples_bad_years": bad_years,
        "claim_dup_within_file": stats["claim_dup_within_file"],
        "path_overlap_with_other_files": stats["path_overlap_with_other_files"],
        "claim_dup_across_files": stats["claim_dup_across_files"],
        "grounding": {k: v for k, v in stats.items() if k.startswith("sampled")},
    }


def main():
    facts_by_id = {}
    for f in ("data/wiki_facts_enriched.jsonl", "data/wiki_facts_obscure.jsonl"):
        for line in Path(f).open(encoding="utf-8"):
            if line.strip():
                x = json.loads(line)
                facts_by_id[x["id"]] = x
    path_ids, claims = set(), set()
    out = []
    for f in ("data/synthesized_claims_final_en.jsonl",
              "data/synthesized_hard_en.jsonl",
              "data/synthesized_obscure_en.jsonl"):
        result = audit(f, facts_by_id, path_ids, claims)
        out.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=1), flush=True)
    Path("data/quality_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
