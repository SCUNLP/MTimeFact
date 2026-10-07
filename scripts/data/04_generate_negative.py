
"""Six evidence-audited perturbations of positive claims; fail closed on uncertainty."""


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from collections import Counter
from pathlib import Path
from mtimefact.common import ROOT, JsonLLM, require_gpt5_claim_generator, run_fingerprint, load_json, read_jsonl, prompt, stable_id, write_json, write_jsonl
from mtimefact.data.synthesis import PERTURBATIONS, audit_accepted, path_time_window, validate_negative


def negative_plan(parent, graph, strategy):
    """Only structured planning. GPT-5 must author every final negative claim."""
    edges = parent["path"]["edges"]
    time_window = parent.get("time_window") or path_time_window(parent["path"])
    if strategy == "data_perturbation":
        edges = [e for e in edges if e["relation"] == "has_value"]
    if not edges:
        return None
    return {"id": stable_id("negative_plan_", [parent["id"], strategy]), "parent_id": parent["id"],
            "status": "draft_only_requires_gpt5", "claim": None, "target_label": False,
            "hop_count": parent["hop_count"], "strategy": strategy, "path": parent["path"],
            "reference_date": parent["reference_date"], "entity_ids": parent["entity_ids"],
            "time_window": time_window,
            "source_ids": parent["source_ids"], "candidate_edge_ids": [e["id"] for e in edges],
            "candidate_fields": sorted(PERTURBATIONS[strategy]), "demo": True,
            "required_next_steps": ["GPT-5 perturbation and final claim generation", "explicit evidence contradiction", "independent audit"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True, type=Path)
    p.add_argument("--positive", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--config", type=Path, default=ROOT / "config/default.json")
    p.add_argument("--cache-dir", default=".cache/llm")
    p.add_argument("--strategies", nargs="+", choices=list(PERTURBATIONS), default=list(PERTURBATIONS))
    p.add_argument("--max-parents", type=int, default=100)
    p.add_argument("--replacement-pool-size", type=int, default=40)
    p.add_argument("--demo", action="store_true", help="Offline perturbation plans only; never final claims")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.output.exists() and not args.overwrite:
        p.error("output exists; use --overwrite or a new path")
    if args.max_parents < 1 or args.replacement_pool_size < 1:
        p.error("limits must be positive")
    cfg, graph = load_json(args.config), load_json(args.graph)
    if bool(args.demo) != bool(graph.get("metadata", {}).get("demo", False)):
        p.error("--demo must match graph metadata.demo")
    generation = None if args.demo else require_gpt5_claim_generator(cfg)
    llm = None if args.demo else JsonLLM(cfg, args.cache_dir)
    threshold = cfg.get("synthesis", {}).get("audit_min_confidence", .9)
    sources = {s.get("id", s.get("source_id")): s for s in graph["sources"]}
    rows, rejected, counts = [], [], Counter()
    seen = set()
    for i, parent in enumerate(read_jsonl(args.positive)):
        if i >= args.max_parents:
            break
        if args.demo:
            if parent.get("status") != "draft_only_requires_gpt5" or parent.get("demo") is not True or parent.get("target_label") is not True:
                raise ValueError("Demo negative planning requires a marked positive plan")
            for strategy in dict.fromkeys(args.strategies):
                plan = negative_plan(parent, graph, strategy)
                if plan:
                    rows.append(plan)
                    counts[strategy] += 1
            continue
        if parent.get("label") is not True or bool(parent.get("demo")) != args.demo:
            raise ValueError("Parent must be positive and match demo mode")
        if not audit_accepted(parent.get("audit", {}), threshold):
            raise ValueError("Parent is not independently audited")
        parent_types = {n["type"] for n in parent["path"]["nodes"]}
        pool = [n for n in graph["nodes"] if n["type"] in parent_types and n["id"] not in parent["entity_ids"]]
        for strategy in dict.fromkeys(args.strategies):
            payload = {"parent": parent, "requested_strategy": strategy,
                       "sources": [sources[s] for s in parent["source_ids"]],
                       "replacement_nodes": pool[:args.replacement_pool_size]}
            try:
                draft = llm.generate(
                    "generator", prompt("data/negative.txt"), payload)
                if draft.get("applicable") is not True:
                    raise ValueError("Not applicable / insufficient contradiction evidence")
                if draft.get("strategy") != strategy:
                    raise ValueError("Wrong requested strategy")
                entity_ids = validate_negative(draft, parent, graph)
                audit = llm.generate(
                    "auditor", prompt("data/audit_claim.txt"), dict(payload, draft=draft, label=False,
                                                              path=parent["path"], reference_date=parent["reference_date"]))
                if not audit_accepted(audit, threshold, negative=True):
                    raise ValueError("Auditor rejected: " + str(audit.get("reason", "checks failed")))
                claim = draft["claim"].strip() + " Date: " + parent["reference_date"]
                if claim in seen:
                    raise ValueError("Duplicate negative")
                seen.add(claim)
                time_window = parent.get("time_window") or path_time_window(parent["path"])
                rows.append({"id": stable_id("neg_", [parent["id"], strategy, draft["edit"]]),
                             "parent_id": parent["id"], "label": False, "hop_count": parent["hop_count"],
                             "claim": claim, "reference_date": parent["reference_date"],
                             "strategy": strategy, "positive_strategy": parent["strategy"],
                             "time_window": time_window,
                             "entity_ids": sorted(entity_ids), "source_ids": parent["source_ids"],
                             "edit": draft["edit"], "contradiction": draft["contradiction"],
                             "path": parent["path"], "audit": audit, "demo": False, "status": "final", "generation": generation})
                counts[strategy] += 1
            except ValueError as exc:
                rejected.append({"parent_id": parent["id"], "strategy": strategy, "reason": str(exc)})
    write_jsonl(args.output, rows)
    write_jsonl(str(args.output) + ".rejected.jsonl", rejected)
    report = {"demo": args.demo, "final_claims": 0 if args.demo else len(rows), "planned" if args.demo else "accepted": len(rows), "by_strategy": dict(counts), "rejected": len(rejected),
              "note": "No assumed 1:1 ratio; inapplicable/unproven negatives are discarded."}
    report["run"] = run_fingerprint(cfg, [args.graph, args.positive], ['data/negative.txt', 'data/audit_claim.txt'])
    write_json(str(args.output) + ".report.json", report)
    print({k:v for k,v in report.items() if k != "run"})
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
