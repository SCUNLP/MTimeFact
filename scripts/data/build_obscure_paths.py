#!/usr/bin/env python3
"""Enumerate hard multi-hop paths that route through freshly harvested obscure
entities (bridges), with tight evidence dates.

Graph = old typed relation edges (resolved tails, popular entities) + new edges
from the obscure entities' own facts.  Paths are seeded at obscure nodes and
grown as simple paths; each candidate must carry >=2 dated edges with a small
minimum gap, prefer all-dated edges, rare relations and deeper hops.

Writes data/graphs/paths_obscure/paths_<n>hop.json in the same format the
synthesis pipeline expects.
"""
import argparse, json, random, re
from collections import Counter, defaultdict
from pathlib import Path


def parse_month(value):
    m = re.match(r'(\d{4})(?:-(\d{2}))?', str(value or ""))
    return (int(m.group(1)), int(m.group(2) or 1)) if m else None


def gap(a, b):
    return abs((a[0] - b[0]) * 12 + a[1] - b[1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-graph", type=Path, default=Path("data/graphs/relation_graph.json"))
    ap.add_argument("--old-triples", type=Path, default=Path("data/wiki_triples_enriched.jsonl"))
    ap.add_argument("--label-map", type=Path, default=Path("data/wiki_entities/label_qid_map.json"))
    ap.add_argument("--new-triples", type=Path, default=Path("data/wiki_triples_obscure.jsonl"))
    ap.add_argument("--obscure-entities", type=Path, default=Path("data/wiki_entities/entities_obscure.jsonl"))
    ap.add_argument("--entities", type=Path, default=Path("data/wiki_entities/entities.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/graphs/paths_obscure"))
    ap.add_argument("--per-hop", type=int, default=1200)
    ap.add_argument("--max-gap", type=int, default=24, help="max months between consecutive evidence dates")
    ap.add_argument("--max-span", type=int, default=360, help="max months covered by dated evidence")
    ap.add_argument("--min-obscure", type=int, default=1, help="obscure nodes required per path")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--report", type=Path, default=Path("data/obscure_paths_report.json"))
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    # node table
    node = {}
    census_qids = set()
    for path in (args.entities, args.obscure_entities):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            node[e["qid"]] = {"label": e.get("label") or e["title"], "category": e.get("category", "?")}
    for line in args.entities.read_text(encoding="utf-8").splitlines():
        if line.strip():
            census_qids.add(json.loads(line)["qid"])
    harvested_obscure = set()
    for line in args.obscure_entities.read_text(encoding="utf-8").splitlines():
        if line.strip():
            harvested_obscure.add(json.loads(line)["qid"])

    # label -> qid: harvested resolution map first (covers non-census entities)
    label2qid = {}
    if args.label_map.exists():
        raw = json.loads(args.label_map.read_text(encoding="utf-8"))
        for label, qid in raw.items():
            label2qid.setdefault(label, qid)
            node.setdefault(qid, {"label": label, "category": "?"})
    for qid, meta in node.items():
        label2qid.setdefault(meta["label"], qid)
    obscure_qids = {q for q in node if q not in census_qids}

    adj = defaultdict(list)

    def add_edge(head, tail, relation, date, fact_id, fact, method):
        if head == tail or head not in node or tail not in node:
            return
        meta = {"relation": relation, "date": date, "fact_id": fact_id,
                "fact": (fact or "")[:200], "method": method, "kind": "typed"}
        adj[head].append((tail, dict(meta, direction="forward")))
        adj[tail].append((head, dict(meta, direction="reverse")))

    # old edges (resolved tails only)
    old = json.loads(args.old_graph.read_text(encoding="utf-8"))
    old_n = 0
    for e in old.get("edges") or []:
        tail = e.get("tail")
        if not tail:
            continue
        # register old-graph labels when the tail is not in the node table
        if tail not in node:
            node[tail] = {"label": e.get("tail_label") or tail, "category": e.get("tail_type") or "?"}
        if e["head"] not in node:
            node[e["head"]] = {"label": e.get("head_label") or e["head"],
                               "category": e.get("head_category") or "?"}
        add_edge(e["head"], tail, e.get("relation") or "related_to", e.get("date"),
                 e.get("fact_id"), e.get("fact"), e.get("method", "rule"))
        old_n += 1
    for qid, meta in list(node.items()):
        label2qid.setdefault(meta["label"], qid)

    # old triples with object labels resolved through the harvested map:
    # these add popular->obscure edges that the earlier graph could not resolve
    old_triple_edges = resolved_old = 0
    if args.old_triples.exists():
        for line in args.old_triples.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            if t.get("object_type") != "entity":
                continue
            head = t.get("head")
            obj = (t.get("object") or "").strip()
            if not head or not obj:
                continue
            tail = obj if re.fullmatch(r"Q\d+", obj) else label2qid.get(obj)
            if not tail or tail not in node:
                continue
            add_edge(head, tail, t.get("relation") or "related_to", t.get("date"),
                     t.get("fact_id"), t.get("fact"), "llm")
            old_triple_edges += 1
            if tail in obscure_qids:
                resolved_old += 1
    print(json.dumps({"old_triple_edges": old_triple_edges,
                      "of_which_into_obscure": resolved_old}), flush=True)

    # new edges from obscure entities' triples
    new_n = unresolved = 0
    if args.new_triples.exists():
        for line in args.new_triples.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            head = t.get("head")
            obj = (t.get("object") or "").strip()
            if not head or not obj:
                continue
            tail = obj if re.fullmatch(r"Q\d+", obj) else label2qid.get(obj)
            if not tail:
                unresolved += 1
                continue
            add_edge(head, tail, t.get("relation") or "related_to", t.get("date"),
                     t.get("fact_id"), t.get("fact"), "llm")
            new_n += 1

    rel_freq = Counter()
    for src, lst in adj.items():
        for _, m in lst:
            rel_freq[m["relation"]] += 1
    rel_total = sum(rel_freq.values()) or 1
    print(json.dumps({"nodes": len(node), "obscure_nodes": len(obscure_qids),
                      "old_edges": old_n, "new_edges": new_n,
                      "new_obj_unresolved": unresolved,
                      "nodes_with_edges": len([q for q, l in adj.items() if l])}), flush=True)

    def rank(options, parent_date):
        def key(item):
            nxt, m = item
            d = parse_month(m.get("date"))
            tight = gap(d, parent_date) if (d and parent_date) else 10 ** 6
            rare = rel_freq.get(m["relation"], 0) / rel_total
            return (0 if d else 1, tight, rare)
        return sorted(options, key=key)

    seeds = [q for q in obscure_qids if adj.get(q)]
    rng.shuffle(seeds)
    print("seeds with edges:", len(seeds), flush=True)

    def enumerate_from(seed, hops, budget):
        """DFS with tightness-ordered expansion; returns list of node/edge chains."""
        results = []
        stack = [([seed], [], {seed})]
        steps = 0
        while stack and len(results) < budget and steps < budget * 30:
            steps += 1
            nodes, edges, used = stack.pop()
            if len(edges) == hops:
                results.append((nodes, edges))
                continue
            parent_date = parse_month(edges[-1]["date"]) if edges else None
            options = [(n, m) for n, m in adj.get(nodes[-1], []) if n not in used]
            for nxt, meta in rank(options, parent_date)[:6]:
                stack.append((nodes + [nxt], edges + [dict(meta, **{"from": nodes[-1], "to": nxt})], used | {nxt}))
        return results

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"nodes": len(node), "obscure_nodes": len(obscure_qids), "old_edges": old_n,
              "new_edges": new_n, "seeds_with_edges": len(seeds),
              "unresolved_object_labels": unresolved, "levels": {}}

    seed_cycle = seeds * 4
    for hops in (2, 3, 4, 5):
        collected, seen_keys = [], set()
        for seed in seed_cycle:
            if len(collected) >= args.per_hop:
                break
            for nodes, edges in enumerate_from(seed, hops, budget=max(8, args.per_hop // max(1, len(seeds)) + 4)):
                if len(collected) >= args.per_hop:
                    break
                dates = sorted(d for d in (parse_month(e.get("date")) for e in edges) if d)
                if len(dates) < 2:
                    continue
                min_gap = min(gap(dates[i], dates[i + 1]) for i in range(len(dates) - 1))
                span = gap(dates[0], dates[-1])
                if min_gap > args.max_gap or span > args.max_span:
                    continue
                obsc = sum(1 for q in nodes if q in obscure_qids)
                if obsc < args.min_obscure:
                    continue
                key = (tuple(sorted(nodes)), tuple(sorted(e["relation"] for e in edges)))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                collected.append({
                    "id": "path_obs_%dhop_%d" % (hops, len(collected)),
                    "hops": hops,
                    "nodes": [{"qid": q, "label": node[q]["label"], "category": node[q]["category"]}
                              for q in nodes],
                    "edges": edges,
                    "typed_edges": len(edges),
                    "typed_only": True,
                    "dated_edges": len(dates),
                    "obscure_nodes": obsc,
                    "dates": [e["date"] for e in edges if e.get("date")],
                    "fact_ids": [e.get("fact_id") for e in edges if e.get("fact_id")],
                    "min_gap_months": min_gap,
                })
        (out_dir / ("paths_%dhop.json" % hops)).write_text(
            json.dumps({"hops": hops, "count": len(collected), "paths": collected}, ensure_ascii=False) + "\n",
            encoding="utf-8")
        gaps = sorted(p["min_gap_months"] for p in collected)
        report["levels"]["%dhop" % hops] = {
            "paths": len(collected),
            "obscure_nodes_total": sum(p["obscure_nodes"] for p in collected),
            "min_gap_p50": gaps[len(gaps) // 2] if gaps else None,
            "all_dated_share": round(sum(1 for p in collected if p["dated_edges"] == p["hops"]) / max(1, len(collected)), 3),
        }
        print("  %d-hop: %d paths (obscure nodes %d, min_gap p50 %s)"
              % (hops, len(collected), sum(p["obscure_nodes"] for p in collected),
                 gaps[len(gaps) // 2] if gaps else None), flush=True)

    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
