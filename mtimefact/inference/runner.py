"""Four-stage inference orchestration, separate from gold-aware evaluation."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import date
import os
from pathlib import Path
import sys

from mtimefact.common import ROOT, JsonLLM, load_json, read_jsonl, run_fingerprint, write_json, write_jsonl
from mtimefact.inference import stages
from mtimefact.inference.tavily import TavilyClient


STAGES = {
    "queries": ("generate_queries", "queries.jsonl", ["inference/generate_queries.txt"]),
    "retrieval": ("retrieve_evidence", "retrieval.jsonl", ["inference/resolve_query.txt", "inference/answer_query.txt"]),
    "explanation": ("generate_explanation", "explanation.jsonl", ["inference/generate_explanation.txt"]),
    "verdict": ("predict_verdict", "predictions.jsonl", ["inference/predict_verdict.txt"]),
}
TRACE_FIELDS = ("queries", "explanation", "explanation_citations", "evidence_status", "evidence_gaps",
                "predicted_label", "verdict_reason")


def read_inputs(path, demo=False, limit=None):
    """Allow gold files as a convenience, but remove every reference/answer field."""
    rows, seen = [], set()
    for source in read_jsonl(path):
        if not isinstance(source, dict):
            raise ValueError("Each input row must be a JSON object")
        sid, claim = source.get("id"), source.get("claim")
        if not isinstance(sid, str) or not sid.strip() or sid in seen:
            raise ValueError("Input IDs must be nonempty, unique strings")
        seen.add(sid)
        if not isinstance(claim, str) or not claim.strip() or source.get("status") == "draft_only_requires_gpt5":
            raise ValueError(f"{sid}: inference requires a final claim, not a synthesis plan")
        history = source.get("inference", {})
        if not isinstance(history, dict) or not isinstance(history.get("stages", {}), dict):
            raise ValueError(f"{sid}: invalid inference metadata")
        if any(name not in STAGES or not isinstance(value, dict) or value.get("status") not in {"ok", "error", "skipped"}
               for name, value in history.get("stages", {}).items()):
            raise ValueError(f"{sid}: invalid inference stage history")
        if demo:
            if source.get("demo") is not True or source.get("fixture") is not True:
                raise ValueError("--demo requires explicitly marked demo=true, fixture=true inputs")
            if history.get("mode") == "live":
                raise ValueError("Cannot continue live inference with fixture responses")
        elif source.get("demo") is True or source.get("fixture") is True or history.get("mode") == "fixture":
            raise ValueError("Fixture inputs/outputs cannot be used for live inference")
        row = {"id": sid, "claim": claim, "queries": [], "explanation": "", "predicted_label": None}
        reference = source.get("reference_date")
        if reference is not None:
            if not isinstance(reference, str) or len(reference) != 10 or date.fromisoformat(reference).isoformat() != reference:
                raise ValueError(f"{sid}: reference_date must be a full ISO date")
            row["reference_date"] = reference
        for key in TRACE_FIELDS:
            if key in source:
                row[key] = deepcopy(source[key])
        row["inference"] = {"mode": "fixture" if demo else "live",
                            "run_id": history.get("run_id"), "stages": deepcopy(history.get("stages", {}))}
        if demo:
            row.update(demo=True, fixture=True)
        rows.append(row)
    if not rows:
        raise ValueError("Input claims must not be empty")
    return rows[:limit] if limit is not None else rows, len(rows)


class BoundReasoner:
    def __init__(self, client, sample_id, stage, run_id, fixtures=None):
        self.client, self.sample_id, self.stage = client, sample_id, stage
        self.run_id, self.fixtures, self.calls = run_id, fixtures, 0

    def generate(self, role, system, payload):
        if role != "reasoner":
            raise ValueError("Inference must use the reasoner role")
        index = self.calls
        self.calls += 1
        if self.fixtures is not None:
            values = self.fixtures.get(self.sample_id, {}).get(self.stage, [])
            if not isinstance(values, list) or index >= len(values):
                raise ValueError(f"Missing inference fixture: {self.sample_id}/{self.stage}/call{index}")
            return deepcopy(values[index])
        request = dict(payload, _inference={"sample_id": self.sample_id, "stage": self.stage,
                                           "run_id": self.run_id, "call_index": index})
        return self.client.generate(role, system, request)


class BoundSearch:
    def __init__(self, client, sample_id, run_id, fixtures=None):
        self.client, self.sample_id, self.run_id = client, sample_id, run_id
        self.fixtures, self.calls = fixtures, 0

    def search(self, query, *, max_results=5, request_context=None, **kwargs):
        index = self.calls
        self.calls += 1
        if self.fixtures is not None:
            values = self.fixtures.get(self.sample_id, {}).get("search", [])
            if not isinstance(values, list) or index >= len(values):
                raise ValueError(f"Missing search fixture: {self.sample_id}/call{index}")
            item = values[index]
            if not isinstance(item, dict) or item.get("query") != query:
                raise ValueError("Search fixture query does not match the executed search")
            response = deepcopy(item.get("response"))
            if not isinstance(response, dict) or not isinstance(response.get("documents"), list):
                raise ValueError("Search fixture requires documents")
            response["documents"] = response["documents"][:max_results]
            if not isinstance(response.get("metadata", {}), dict):
                raise ValueError("Search fixture metadata must be an object")
            response.setdefault("metadata", {}).update(provider="fixture", fixture=True)
            return response
        context = {"sample_id": self.sample_id, "run_id": self.run_id,
                   "call_index": index, "query_context": request_context}
        return self.client.search(query, max_results=max_results, request_context=context, **kwargs)


def clear_stage_outputs(record, stage):
    """Rerunning an earlier stage must invalidate any stale downstream answers."""
    index = list(STAGES).index(stage)
    if index == 0:
        record["queries"] = []
    if index <= 2:
        record["explanation"] = ""
        for key in ("explanation_citations", "evidence_status", "evidence_gaps"):
            record.pop(key, None)
    record["predicted_label"] = None
    record.pop("verdict_reason", None)
    for name in list(STAGES)[index:]:
        record["inference"]["stages"].pop(name, None)


def run_stage(stage, records, config, options, fixtures=None, client=None, search_client=None):
    if (fixtures is not None) != bool(options["demo"]):
        raise ValueError("Fixtures are required only in explicit demo mode")
    function = getattr(stages, STAGES[stage][0])
    index = list(STAGES).index(stage)
    for record in records:
        previous = list(STAGES)[:index]
        history = record["inference"]["stages"]
        failed = [s for s in previous if history.get(s, {}).get("status") in {"error", "skipped"}]
        clear_stage_outputs(record, stage)
        record["inference"]["run_id"] = options["run_id"]
        reasoner = BoundReasoner(client, record["id"], stage, options["run_id"], fixtures)
        search = BoundSearch(search_client, record["id"], options["run_id"], fixtures)
        status = {"status": "ok", "run_id": options["run_id"], "mode": "fixture" if options["demo"] else "live"}
        if failed:
            status.update(status="skipped", reason="Earlier inference stage failed: " + ", ".join(failed))
        else:
            try:
                if stage == "retrieval":
                    function(record, reasoner, search, options)
                else:
                    function(record, reasoner, options)
            except (ValueError, RuntimeError, TypeError, KeyError, OSError) as exc:
                status.update(status="error", error_type=type(exc).__name__, reason=str(exc))
        
        status.update(model_calls=reasoner.calls, search_calls=search.calls)
        record["inference"]["stages"][stage] = status
        print(f"{stage}: {record['id']} {status['status']}", flush=True)
    counts = Counter(r["inference"]["stages"][stage]["status"] for r in records)
    report = {"stage": stage, "demo": options["demo"], "n_inputs": len(records), "counts": dict(counts),
              "complete": counts["ok"] == len(records),
              "model_calls": sum(r["inference"]["stages"][stage]["model_calls"] for r in records),
              "search_calls": sum(r["inference"]["stages"][stage]["search_calls"] for r in records)}
    if stage == "verdict":
        report["abstentions"] = sum(r["predicted_label"] is None and r["inference"]["stages"][stage]["status"] == "ok" for r in records)
    return report


def add_arguments(parser, all_stages=False):
    parser.add_argument("--input", type=Path, required=True, help="Claims JSONL or preceding inference stage JSONL; gold fields are excluded")
    parser.add_argument("--output-dir" if all_stages else "--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/inference")
    parser.add_argument("--max-queries", type=int, help="Maximum generated subqueries (default: config.inference.max_queries)")
    parser.add_argument("--k", type=int, help="Results per search, 1..20 (default: config.retrieval.max_results)")
    parser.add_argument("--max-document-chars", type=int, help="Bounded evidence text per document")
    parser.add_argument("--limit", type=int, help="Process only the first N input claims")
    parser.add_argument("--run-id", default="main", help="Included in model/search cache identity")
    parser.add_argument("--demo", action="store_true", help="Replay explicitly marked, hand-authored inference fixtures; no network")
    parser.add_argument("--fixtures", type=Path, default=ROOT / "examples/inference/responses.json")
    parser.add_argument("--overwrite", action="store_true")


def context(args, selected_stages):
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if not args.run_id.strip():
        raise ValueError("--run-id must be nonempty")
    records, available = read_inputs(args.input, args.demo, args.limit)
    config = load_json(args.config)
    if not isinstance(config, dict) or any(not isinstance(config.get(key, {}), dict)
                                           for key in ("llm", "inference", "retrieval")):
        raise ValueError("Config requires object llm, inference and retrieval sections")
    if any(not isinstance(spec, dict) for spec in config.get("llm", {}).values()):
        raise ValueError("Each config.llm role must be an object")
    options = {"demo": args.demo, "run_id": args.run_id,
               "max_queries": args.max_queries if args.max_queries is not None else config.get("inference", {}).get("max_queries", 8),
               "k": args.k if args.k is not None else config.get("retrieval", {}).get("max_results", 5),
               "max_document_chars": args.max_document_chars if args.max_document_chars is not None else config.get("retrieval", {}).get("max_document_chars", 6000)}
    for key, upper in (("max_queries", 20), ("k", 20), ("max_document_chars", 100000)):
        value = options[key]
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"{key} must be an integer in 1..{upper}")
    config.setdefault("retrieval", {})["max_document_chars"] = options["max_document_chars"]
    if args.demo:
        fixtures = load_json(args.fixtures)
        if not isinstance(fixtures, dict) or any(not isinstance(fixtures.get(r["id"]), dict) for r in records):
            raise ValueError("Inference fixture file requires an object for every input sample ID")
        return records, available, config, options, fixtures, None, None
    spec = config.get("llm", {}).get("reasoner", {})
    if spec.get("provider") not in {"openai", "gemini"} or any(not isinstance(spec.get(k), str) or not spec[k].strip() for k in ("model", "base_url", "api_key_env")):
        raise ValueError("Set an explicit config.llm.reasoner provider/model/base_url/api_key_env")
    envs = [spec["api_key_env"]]
    search_client = None
    if "retrieval" in selected_stages:
        envs.append(config.get("retrieval", {}).get("api_key_env", "TAVILY_API_KEY"))
        search_client = TavilyClient(config, args.cache_dir / "search")
    missing = [name for name in envs if not os.environ.get(name)]
    if missing:
        raise ValueError("Missing API environment variables: " + ", ".join(missing) + ". Configure them for live inference; --demo only replays fixtures.")
    return records, available, config, options, None, JsonLLM(config, args.cache_dir / "llm"), search_client


def main(stage=None):
    parser = argparse.ArgumentParser(description="Run four-stage temporal fact-checking inference" if stage is None else f"Inference stage: {stage}")
    add_arguments(parser, all_stages=stage is None)
    args = parser.parse_args()
    selected = list(STAGES) if stage is None else [stage]
    try:
        targets = {name: args.output_dir / STAGES[name][1] if stage is None else args.output for name in selected}
        paths = [p for target in targets.values() for p in (target, Path(str(target) + ".report.json"))]
        if stage is None:
            paths.append(args.output_dir / "summary.json")
        protected = [args.input, args.config] + ([args.fixtures] if args.demo else [])
        if any(p.resolve() == source.resolve() for p in paths for source in protected):
            raise ValueError("Outputs must not overwrite input, config, or fixtures")
        if not args.overwrite and any(p.exists() for p in paths):
            raise ValueError("Inference output exists; use --overwrite or a new output path")
        records, available, config, options, fixtures, client, search = context(args, selected)
        summaries = {}
        for name in selected:
            report = run_stage(name, records, config, options, fixtures, client, search)
            report.update(options=options, n_available=available,
                          input_fields="id, claim, reference_date; preceding prediction trace only",
                          method="prompt_based_four_stage_baseline_not_trained_MTimeFactCheck",
                          call_count_semantics="client invocations, including cached or fixture responses")
            inputs = [args.input] + ([args.fixtures] if args.demo else [])
            report["run"] = run_fingerprint(config, inputs, STAGES[name][2])
            if name == "retrieval":
                report["retrieval_config"] = {k: v for k, v in config.get("retrieval", {}).items()
                                               if k in {"provider", "base_url", "search_depth", "topic", "max_document_chars", "timeout", "retries"}}
            write_jsonl(targets[name], records)
            write_json(str(targets[name]) + ".report.json", report)
            summaries[name] = report
        complete = all(r["complete"] for r in summaries.values())
        if stage is None:
            write_json(args.output_dir / "summary.json", {"demo": args.demo, "n_inputs": len(records), "n_available": available,
                       "complete": complete, "stages": summaries, "predictions_file": str(targets["verdict"]),
                       "note": "Fixture mode replays hand-authored examples, not measured model inference. Null verdicts are explicit abstentions and score zero under binary evaluation."})
        return 0 if complete else 2
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("Inference input error: " + str(exc), file=sys.stderr)
        return 1


def stage_cli(stage):
    return main(stage)
