#!/usr/bin/env python3
"""Select the hardest multi-hop paths and write them to a new paths directory.

Hardness signals:
  * tight evidence dates   - min gap between consecutive edge dates, and total span
  * rare relations         - mean prior probability of the path relations in the KG
  * deeper reasoning       - hop count
  * less recognizable      - max monthly pageviews among path nodes

All edges must carry a date so the time window is fully evidence-grounded.
Writes data/graphs/paths_hard/paths_<n>hop.json in the same format as the
original library, plus a selection report.
"""
import argparse, json, re
from collections import Counter
from pathlib import Path

VIEW_PRIOR = None


def parse_month(value):
    m = re.match(r'(\d{4})(?:-(\d{2}))?', str(value or ""))
    return (int(m.group(1)), int(m.group(2) or 1)) if m else None


def months_between(a, b):
    return abs((a[0] - b[0]) * 12 + a[1] - b[1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths-dir", type=Path, default=Path("data/graphs/paths"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/graphs/paths_hard"))
    ap.add_argument("--entities", type=Path, default=Path("data/wiki_entities/entities.jsonl"))
    ap.add_argument("--triples", type=Path, default=Path("data/wiki_triples_enriched.jsonl"))
    ap.add_argument("--top", type=int, default=3000)
    ap.add_argument("--max-min-gap", type=int, default=48, help="max months between consecutive evidence dates")
    ap.add_argument("--max-span", type=int, default=300, help="max months covered by the dated evidence")
    ap.add_argument("--max-hop2-3", type=int, default=450,
                    help="cap on 2/3-hop paths inside the hard set")
    ap.add_argument("--report", type=Path, default=Path("data/hard_paths_report.json"))
    args = ap.parse_args(argv)

    views = {}
    for line in args.entities.open(encoding="utf-8"):
        if line.strip():
            e = json.loads(line)
            views[e["qid"]] = e.get("monthly_views_max") or 0

    rel_freq = Counter()
    for line in args.triples.open(encoding="utf-8"):
        if line.strip():
            rel_freq[json.loads(line).get("relation") or "?"] += 1
    total = sum(rel_freq.values()) or 1

    scored = []
    skipped_undated = 0
    for hop_file in sorted(args.paths_dir.glob("paths_*hop.json")):
        payload = json.loads(hop_file.read_text(encoding="utf-8"))
        for p in payload.get("paths") or []:
            dates = [parse_month(e.get("date")) for e in p["edges"]]
            dated = [d for d in dates if d]
            if len(dated) < 2:
                skipped_undated += 1
                continue
            dates_sorted = sorted(dated)
            gaps = [months_between(dates_sorted[i], dates_sorted[i + 1])
                    for i in range(len(dates_sorted) - 1)]
            min_gap = min(gaps)
            span = months_between(dates_sorted[0], dates_sorted[-1])
            if min_gap > args.max_min_gap or span > args.max_span:
                continue
            dated_ratio = len(dated) / len(p["edges"])
            rels = [e.get("relation") for e in p["edges"]]
            prior = sum(rel_freq.get(r, 0) / total for r in rels) / max(len(rels), 1)
            node_views = [views.get(n["qid"]) for n in p["nodes"]]
            node_views = [v for v in node_views if v is not None]
            max_views = max(node_views) if node_views else 0
            # score: higher = harder
            score = (1.0 / (1.0 + min_gap)) * 40 \
                    + (1.0 / (1.0 + span / 12.0)) * 20 \
                    + (1.0 / (1.0 + prior * 1000)) * 20 \
                    + (p["hops"] / 5.0) * 10 \
                    + dated_ratio * 5 \
                    + (1.0 / (1.0 + max_views / 1e6)) * 5
            scored.append({"path": p, "score": score, "min_gap": min_gap, "span": span,
                           "prior": prior, "max_views": max_views, "dated_ratio": dated_ratio})
    scored.sort(key=lambda x: x["score"], reverse=True)

    picked, deep_cap = [], args.max_hop2_3
    for item in scored:
        if item["path"]["hops"] <= 3:
            if deep_cap <= 0:
                continue
            deep_cap -= 1
        picked.append(item["path"])
        if len(picked) >= args.top:
            break

    by_hop = Counter(p["hops"] for p in picked)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for hop in sorted(by_hop):
        payload = {"hops": hop, "count": by_hop[hop],
                   "paths": [p for p in picked if p["hops"] == hop]}
        (args.out_dir / f"paths_{hop}hop.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    sel = [s for s in scored if s["path"]["id"] in {p["id"] for p in picked}]
    pcts = ((10, .1), (25, .25), (50, .5), (75, .75), (90, .9))
    def q(key, percentile):
        vals = sorted(s[key] for s in sel)
        return vals[min(len(vals) - 1, int(len(vals) * percentile))]
    report = {
        "paths_total": len(scored) + skipped_undated,
        "paths_skipped_lt2_dated_edges": skipped_undated,
        "paths_with_2plus_dated_edges": len(scored),
        "picked": len(picked), "by_hop": dict(sorted(by_hop.items())),
        "selected_stats": {
            "min_gap_months_pct": {label: q("min_gap", p) for label, p in pcts},
            "span_months_pct": {label: q("span", p) for label, p in pcts},
            "mean_relation_prior_pct": {label: q("prior", p) for label, p in pcts},
            "max_node_views_pct": {label: q("max_views", p) for label, p in pcts},
            "dated_edge_ratio_pct": {label: round(q("dated_ratio", p), 2) for label, p in pcts},
        },
        "output_dir": str(args.out_dir),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
