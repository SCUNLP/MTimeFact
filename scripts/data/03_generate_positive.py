
"""Sample 2–5 edge paths, realize with GPT-5, independently audit with Gemini."""


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from collections import Counter
from pathlib import Path
from mtimefact.common import ROOT, JsonLLM, require_gpt5_claim_generator, run_fingerprint, load_json, prompt, stable_id, write_json, write_jsonl
from mtimefact.data.synthesis import PathSampler, STRATEGIES, audit_accepted, path_time_window, validate_positive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--config", type=Path, default=ROOT / "config/default.json")
    p.add_argument("--cache-dir", default=".cache/llm")
    p.add_argument("--hops", nargs="+", type=int, choices=[2,3,4,5], default=[2,3,4,5])
    p.add_argument("--per-hop", type=int, default=10)
    p.add_argument("--strategy", choices=list(STRATEGIES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-attempts", type=int, default=1000, help="Per hop; bounds search and API work")
    p.add_argument("--demo", action="store_true", help="Offline plans only: no final claim or label is generated")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.per_hop < 1 or args.max_attempts < 1:
        p.error("counts must be positive")
    if args.output.exists() and not args.overwrite:
        p.error("output exists; use --overwrite or a new path")
    cfg, graph = load_json(args.config), load_json(args.graph)
    graph_demo = graph.get("metadata", {}).get("demo", False)
    if bool(args.demo) != bool(graph_demo):
        p.error("--demo must match graph metadata.demo; fixture graphs cannot become real data")
    sampler = PathSampler(graph, args.seed, cfg.get("synthesis", {}).get("anchor_degree_power", 1.0))
    generation = None if args.demo else require_gpt5_claim_generator(cfg)
    llm = None if args.demo else JsonLLM(cfg, args.cache_dir)
    threshold = cfg.get("synthesis", {}).get("audit_min_confidence", .9)
    rows, rejected, seen, counts = [], [], set(), Counter()
    sources = {s.get("id", s.get("source_id")): s for s in graph.get("sources", [])}
    for hop in dict.fromkeys(args.hops):
        for attempt in range(args.max_attempts):
            if counts[hop] >= args.per_hop:
                break
            path = sampler.sample(hop, args.strategy)
            if not path or path["id"] in seen:
                continue
            seen.add(path["id"])
            source_ids = sorted({s for e in path["edges"] for s in e["source_ids"]})
            payload = {"path": path, "strategy_description": STRATEGIES[path["strategy"]],
                       "sources": [sources[s] for s in source_ids]}
            if args.demo:
                rows.append({"id": stable_id("positive_plan_", path["id"]), "status": "draft_only_requires_gpt5",
                             "claim": None, "target_label": True, "hop_count": hop,
                             "reference_date": path["reference_date"], "strategy": path["strategy"],
                             "time_window": path_time_window(path),
                             "entity_ids": path["entity_ids"], "source_ids": source_ids, "path": path, "demo": True})
                counts[hop] += 1
                continue
            try:
                draft = llm.generate("generator", prompt("data/positive.txt"), payload)
                if draft.get("skip"):
                    raise ValueError(draft.get("reason", "generator skipped"))
                validate_positive(draft, path)
                audit = llm.generate(
                    "auditor", prompt("data/audit_claim.txt"), dict(payload, draft=draft, label=True))
                if not audit_accepted(audit, threshold):
                    raise ValueError("Auditor rejected: " + str(audit.get("reason", "checks failed")))
                rows.append({"id": stable_id("pos_", [path["id"], draft["claim"]]), "label": True,
                             "hop_count": hop, "claim": draft["claim"].strip() + " Date: " + path["reference_date"],
                             "reference_date": path["reference_date"], "strategy": path["strategy"],
                             "time_window": path_time_window(path),
                             "entity_ids": path["entity_ids"], "source_ids": source_ids, "path": path,
                             "audit": audit, "demo": False, "status": "final", "generation": generation})
                counts[hop] += 1
            except ValueError as exc:
                rejected.append({"path_id": path["id"], "reason": str(exc)})
    write_jsonl(args.output, rows)
    write_jsonl(str(args.output) + ".rejected.jsonl", rejected)
    report = {"demo": args.demo, "seed": args.seed, "requested_per_hop": args.per_hop,
              "final_claims": 0 if args.demo else len(rows), "planned_by_hop" if args.demo else "accepted": dict(counts), "shortfalls": {h: args.per_hop-counts[h] for h in args.hops},
              "unique_attempted_paths": len(seen), "rejected": len(rejected)}
    report["run"] = run_fingerprint(cfg, [args.graph], ['data/positive.txt', 'data/audit_claim.txt'])
    write_json(str(args.output) + ".report.json", report)
    print({k:v for k,v in report.items() if k != "run"})
    return 0 if all(counts[h] == args.per_hop for h in args.hops) else 2


if __name__ == "__main__":
    raise SystemExit(main())
