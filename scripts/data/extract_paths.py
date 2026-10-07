#!/usr/bin/env python3
"""Multi-hop depth path extraction over the entity graphs.

Builds hop-stratified path libraries (2/3/4/5 hops) over the combined graph:

  * typed edges  — from relation_graph.json (rule) and, when present,
                   relation_graph_llm.json (LLM triples; typed, entity-linked)
  * mention edges — from cooccurrence_graph.json (fact co-mention), with the
                   top hub nodes excluded so paths do not route through
                   generic connectors such as "United States"

Each sampled path is a simple path (no repeated nodes).  Typed edges are
preferred during traversal; mention edges are allowed but counted separately
so consumers can filter "typed-only" chains.  Paths carry per-edge relation,
date, method and the supporting fact id.

Outputs (data/graphs/paths/):
  paths_2hop.json ... paths_5hop.json   (samples + counts per level)
  path_report.json                      (statistics per hop level)
  depth_layers.json                     (BFS depth layers from the top hub)

Usage:
  python3 scripts/data/extract_paths.py [--per-hop 3000] [--allow-mentions]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


def load_edges(path, kind):
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    edges = payload.get("edges") or []
    for edge in edges:
        edge["_kind"] = kind
    return edges


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graphs-dir", type=Path, default=Path("data/graphs"))
    parser.add_argument("--entities", type=Path, default=Path("data/wiki_entities/entities.jsonl"))
    parser.add_argument("--per-hop", type=int, default=3000, help="Path samples per hop level")
    parser.add_argument("--hubs", type=int, default=15, help="Top-N mention hubs excluded from paths")
    parser.add_argument("--allow-mentions", action="store_true", default=True)
    parser.add_argument("--typed-only", action="store_true", help="Use only typed relation edges")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    graphs = args.graphs_dir
    entities = [json.loads(l) for l in args.entities.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_qid = {e["qid"]: e for e in entities}

    typed = load_edges(graphs / "relation_graph_rule.json", "rule")
    if not typed:
        typed = load_edges(graphs / "relation_graph.json", "rule")
    llm_edges = load_edges(graphs / "relation_graph_llm.json", "llm")
    typed = [e for e in typed if e.get("tail")]
    llm_edges = [e for e in llm_edges if e.get("tail")]
    mention = load_edges(graphs / "cooccurrence_graph.json", "mention")

    # adjacency: typed edges are traversable in both directions (reverse flagged)
    adj = defaultdict(list)  # qid -> list of (neighbor, meta)
    for edge in typed + llm_edges:
        head, tail = edge["head"], edge["tail"]
        base = {"relation": edge["relation"], "method": edge.get("method", edge["_kind"]),
                "date": edge.get("date"), "fact_id": edge.get("fact_id"),
                "fact": (edge.get("fact") or "")[:160], "kind": "typed"}
        adj[head].append((tail, dict(base, direction="forward")))
        adj[tail].append((head, dict(base, direction="reverse")))

    mention_deg = Counter()
    for edge in mention:
        mention_deg[edge["source"]] += 1
        mention_deg[edge["target"]] += 1
    hub_qids = {qid for qid, _ in mention_deg.most_common(args.hubs)}
    mention_adj = defaultdict(list)
    if args.allow_mentions and not args.typed_only:
        for edge in mention:
            a, b = edge["source"], edge["target"]
            if a in hub_qids or b in hub_qids:
                continue
            meta = {"relation": "mentions", "method": "cooccurrence", "kind": "mention",
                    "mentions": edge.get("mentions"), "fact_id": edge.get("example_fact")}
            mention_adj[a].append((b, dict(meta, direction="undirected")))
            mention_adj[b].append((a, dict(meta, direction="undirected")))

    starts_typed = [qid for qid in adj if adj[qid]]
    starts_mention = [qid for qid in mention_adj if mention_adj[qid]]
    print(json.dumps({"typed_edges": len(typed) + len(llm_edges), "mention_edges": sum(len(v) for v in mention_adj.values()) // 2,
                      "typed_nodes": len(starts_typed), "mention_nodes": len(starts_mention),
                      "hub_excluded": sorted(by_qid.get(h, {}).get("label", h) for h in hub_qids)[:15]},
                     ensure_ascii=False), flush=True)

    def neighbors(qid):
        out = list(adj.get(qid, ()))
        if mention_adj:
            out.extend(mention_adj.get(qid, ()))
        return out

    def sample_path(hops, typed_bias=0.8):
        budget = 400
        while budget > 0:
            budget -= 1
            # start from a typed-connected node when possible
            start = rng.choice(starts_typed) if (starts_typed and rng.random() < typed_bias) else (
                rng.choice(starts_mention) if starts_mention else None)
            if not start:
                return None
            nodes, edges, used = [start], [], {start}
            cur = start
            ok = True
            while len(edges) < hops:
                options = [(n, m) for n, m in neighbors(cur) if n not in used]
                if not options:
                    ok = False
                    break
                typed_opts = [o for o in options if o[1]["kind"] == "typed"]
                pick_from = typed_opts if (typed_opts and rng.random() < typed_bias) else options
                nxt, meta = rng.choice(pick_from)
                used.add(nxt)
                nodes.append(nxt)
                edges.append({"from": cur, "to": nxt, **meta})
                cur = nxt
            if ok:
                return nodes, edges
        return None

    paths_dir = graphs / "paths"
    paths_dir.mkdir(parents=True, exist_ok=True)
    report_levels, seen = {}, set()
    for hops in (2, 3, 4, 5):
        collected = []
        attempts = args.per_hop * 40
        while len(collected) < args.per_hop and attempts > 0:
            attempts -= 1
            sample = sample_path(hops)
            if not sample:
                continue
            nodes, edges = sample
            key = (tuple(sorted(nodes)), tuple(sorted((e["from"], e["to"], e["relation"]) for e in edges)))
            if key in seen:
                continue
            seen.add(key)
            typed_count = sum(1 for e in edges if e["kind"] == "typed")
            dated = [e["date"] for e in edges if e.get("date")]
            collected.append({
                "id": "path_%dhop_%d" % (hops, len(collected)),
                "hops": hops,
                "nodes": [{"qid": q, "label": by_qid.get(q, {}).get("label", q),
                           "category": by_qid.get(q, {}).get("category", "?")} for q in nodes],
                "edges": edges,
                "typed_edges": typed_count,
                "typed_only": typed_count == hops,
                "dates": sorted(dated) if dated else [],
                "fact_ids": [e.get("fact_id") for e in edges if e.get("fact_id")],
            })
        (paths_dir / ("paths_%dhop.json" % hops)).write_text(
            json.dumps({"hops": hops, "count": len(collected), "paths": collected},
                       ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        typed_only = sum(1 for p in collected if p["typed_only"])
        with_dates = sum(1 for p in collected if len(p["dates"]) >= 1)
        report_levels["%dhop" % hops] = {"paths": len(collected), "typed_only": typed_only,
                                         "with_dates": with_dates,
                                         "avg_typed_edges": round(sum(p["typed_edges"] for p in collected) /
                                                                  max(1, len(collected)), 2)}
        print("  %d-hop: %d paths (typed-only %d, with dates %d)" % (hops, len(collected), typed_only, with_dates),
              flush=True)

    # depth layers from the mention-hub in the combined graph
    if starts_typed or starts_mention:
        root = (mention_deg.most_common(1)[0][0] if mention_deg else starts_typed[0])
        depth, queue = {root: 0}, deque([root])
        while queue:
            node = queue.popleft()
            for nxt, _ in neighbors(node):
                if nxt not in depth:
                    depth[nxt] = depth[node] + 1
                    queue.append(nxt)
        (graphs / "depth_layers.json").write_text(
            json.dumps({"root": root, "root_label": by_qid.get(root, {}).get("label", root),
                        "max_depth": max(depth.values(), default=0),
                        "layer_sizes": dict(Counter(depth.values())),
                        "nodes": {q: d for q, d in sorted(depth.items(), key=lambda kv: kv[1])}},
                       ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        layer_sizes = dict(sorted(Counter(depth.values()).items()))
    else:
        layer_sizes = {}
    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "typed_edges_used": len(typed) + len(llm_edges),
              "mention_edges_used": sum(len(v) for v in mention_adj.values()) // 2,
              "hubs_excluded": args.hubs, "levels": report_levels,
              "depth_layers": layer_sizes,
              "outputs": [str(p.relative_to(graphs)) for p in sorted(paths_dir.glob("*.json"))]}
    (graphs / "path_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    print(json.dumps({"status": "ok", **report_levels, "depth_layers": layer_sizes}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
