#!/usr/bin/env python3
"""Measure the difficulty profile of the synthesized multi-hop claims.

Dimensions:
  1. hop count and path length
  2. entity obscurity      - monthly Wikipedia pageviews of path nodes and mentioned entities
  3. relation rarity       - frequency of each path relation in the KG
  4. temporal tightness    - min gap between consecutive evidence dates (months)
  5. perturbation subtlety - year delta for chronological_distortion, same-category
                             substitution for entity_substitution, numeric delta for data_perturbation
Outputs a JSON report and prints summary tables.
"""
import argparse, json, re, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

YEAR = re.compile(r'\b(1[0-9]{3}|20[0-9]{2})\b')
NUM = re.compile(r'\b\d+(?:\.\d+)?\b')
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'&.\-]*")


def parse_month(date_str):
    """'1929' -> (1929, 1); '2015-06' -> (2015, 6); '2015-06-11' -> (2015, 6)."""
    if not date_str:
        return None
    m = re.match(r'(\d{4})(?:-(\d{2}))?', str(date_str))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) or 1)


def months_between(a, b):
    return abs((a[0] - b[0]) * 12 + (a[1] - b[1]))


def pct(values, p):
    if not values:
        return None
    values = sorted(values)
    k = min(len(values) - 1, max(0, int(round((len(values) - 1) * p / 100.0))))
    return values[k]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/synthesized_claims_final_en.jsonl"))
    ap.add_argument("--entities", type=Path, default=Path("data/wiki_entities/entities.jsonl"))
    ap.add_argument("--triples", type=Path, default=Path("data/wiki_triples_enriched.jsonl"))
    ap.add_argument("--report", type=Path, default=Path("data/difficulty_report.json"))
    ap.add_argument("--hard-out", type=Path, default=None, help="write the hard subset here")
    args = ap.parse_args(argv)

    # entity table: title -> (views, category)
    ent = {}
    for line in args.entities.open(encoding="utf-8"):
        if not line.strip():
            continue
        e = json.loads(line)
        ent[e["title"]] = (e.get("monthly_views_max") or 0, e.get("category") or "?")
    print("entities:", len(ent), flush=True)

    # relation frequencies from the KG
    rel_freq = Counter()
    for line in args.triples.open(encoding="utf-8"):
        if not line.strip():
            continue
        t = json.loads(line)
        rel_freq[t.get("relation") or "?"] += 1
    rel_total = sum(rel_freq.values()) or 1
    print("triples:", rel_total, "relations:", len(rel_freq), flush=True)

    # multi-token capitalised candidates for entity matching in claims
    def candidates(text):
        toks = WORD.findall(text)
        out, n = [], len(toks)
        for i in range(n):
            if not toks[i][:1].isupper() and not toks[i][:1].isdigit():
                continue
            for ln in range(1, 6):
                if i + ln > n:
                    break
                g = " ".join(toks[i:i + ln])
                out.append(g)
        return out

    stats = defaultdict(list)
    hops = Counter(); pert = Counter()
    views_min = []; views_mean = []; rel_p = []
    gap_list = []; span_list = []; claim_words = []
    chrono_delta = []; sub_samecat = Counter(); sub_views_ratio = []
    data_delta = []
    missing_nodes = 0; node_total = 0
    hard_ids = []
    hard_reasons = Counter()

    n = 0
    for line in args.input.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line); n += 1
        hop = r.get("hop_count") or len(r.get("evidence_chain") or []) or 0
        hops[hop] += 1
        pert[r.get("perturbation") or "positive"] += 1
        claim = r.get("claim") or ""

        # --- entity obscurity along the path
        node_views = []
        for node in r.get("path_nodes") or []:
            node_total += 1
            v = ent.get(node)
            if v is None:
                missing_nodes += 1
            else:
                node_views.append(v[0])
        if node_views:
            vmin = min(node_views); vmean = sum(node_views) / len(node_views)
            views_min.append(vmin); views_mean.append(vmean)
        else:
            vmin = vmean = None

        # --- relation rarity along the path
        path_rels = r.get("path_relations") or []
        if path_rels:
            freq = [rel_freq.get(x, 0) / rel_total for x in path_rels]
            rel_p.append(sum(freq) / len(freq))          # mean prior probability
        else:
            freq = []

        # --- temporal tightness
        dates = [parse_month(e.get("date")) for e in (r.get("evidence_chain") or [])]
        dates = sorted(d for d in dates if d)
        if len(dates) >= 2:
            gaps = [months_between(dates[i], dates[i + 1]) for i in range(len(dates) - 1)]
            gap_list.append(min(gaps))
            span_list.append(months_between(dates[0], dates[-1]))
        claim_words.append(len(claim.split()))

        # --- perturbation subtlety
        kind = r.get("perturbation")
        this_chrono = None
        this_sub_samecat = None
        ev_years = set()
        for e in r.get("evidence_chain") or []:
            pm = parse_month(e.get("date"))
            if pm:
                ev_years.add(pm[0])
        if kind == "chronological_distortion" and ev_years:
            claim_years = [int(y) for y in YEAR.findall(claim)]
            odd = [y for y in claim_years if y not in ev_years]
            if odd:
                d = min(min(abs(y - ey) for ey in ev_years) for y in odd)
                chrono_delta.append(d)
                this_chrono = d
        elif kind == "entity_substitution":
            pathset = set(r.get("path_nodes") or [])
            mystery = [c for c in candidates(claim) if c in ent and c not in pathset]
            if mystery:
                cats = {ent[c][1] for c in mystery}
                pathcats = {ent[x][1] for x in pathset if x in ent}
                if cats & pathcats:
                    sub_samecat["same_category"] += 1
                    this_sub_samecat = "same_category"
                else:
                    sub_samecat["different_category"] += 1
                    this_sub_samecat = "different_category"
                mv = min(ent[c][0] for c in mystery)
                if vmin:
                    sub_views_ratio.append(mv / max(vmin, 1))
        elif kind == "data_perturbation":
            ev_nums = set()
            for e in r.get("evidence_chain") or []:
                ev_nums.update(NUM.findall(e.get("fact") or ""))
            claim_nums = NUM.findall(claim)
            odd = [x for x in claim_nums if x not in ev_nums]
            rel = []
            for x in odd:
                try:
                    xv = float(x)
                except ValueError:
                    continue
                best = None
                for ev in ev_nums:
                    try:
                        d = abs(xv - float(ev)) / max(abs(float(ev)), 1e-9)
                    except ValueError:
                        continue
                    best = d if best is None else min(best, d)
                if best is not None:
                    rel.append(best)
            if rel:
                data_delta.append(min(rel))

        # --- hard subset rules (transparent, reported)
        reasons = []
        if hop >= 5:
            reasons.append("hop5")
        if vmin is not None and vmin <= 200000:
            reasons.append("obscure_endpoint<=200k")
        if rel_p and rel_p[-1] <= 2e-5:
            reasons.append("rare_relations")
        if this_chrono is not None and this_chrono <= 2:
            reasons.append("subtle_year")
        if this_sub_samecat == "same_category":
            reasons.append("same_category_substitution")
        if len(reasons) >= 2:
            hard_ids.append(r["id"])
            for x in reasons:
                hard_reasons[x] += 1

    report = {
        "records": n,
        "hops": dict(sorted(hops.items(), key=lambda x: str(x[0]))),
        "perturbation": dict(pert.most_common()),
        "nodes_missing_from_entity_table": missing_nodes,
        "nodes_total": node_total,
        "views_min_pct": {p: pct(views_min, p) for p in (10, 25, 50, 75, 90)},
        "views_mean_pct": {p: pct(views_mean, p) for p in (10, 25, 50, 75, 90)},
        "relation_prior_pct": {p: pct(rel_p, p) for p in (10, 25, 50, 75, 90)},
        "min_evidence_gap_months_pct": {p: pct(gap_list, p) for p in (10, 25, 50, 75, 90)},
        "evidence_span_months_pct": {p: pct(span_list, p) for p in (10, 25, 50, 75, 90)},
        "claim_words_pct": {p: pct(claim_words, p) for p in (10, 25, 50, 75, 90)},
        "chronological_distortion_year_delta": dict(Counter(chrono_delta).most_common(12)),
        "entity_substitution_category_match": dict(sub_samecat),
        "entity_substitution_views_ratio_pct": {p: pct(sub_views_ratio, p) for p in (10, 50, 90)},
        "data_perturbation_relative_delta_pct": {p: pct(data_delta, p) for p in (10, 25, 50, 75, 90)},
        "hard_subset_rule": ">=2 of: hop5 / min node views<=200k / mean relation prior<=2e-5 / subtle year delta<=2",
        "hard_subset_count": len(hard_ids),
        "hard_reason_counts": dict(hard_reasons),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)

    if args.hard_out:
        keep = set(hard_ids)
        with args.hard_out.open("w", encoding="utf-8") as h:
            for line in args.input.open(encoding="utf-8"):
                if not line.strip():
                    continue
                if json.loads(line)["id"] in keep:
                    h.write(line)
        print("wrote", len(hard_ids), "hard records to", args.hard_out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
