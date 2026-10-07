"""Strict ID alignment, cached GPT-5 judging, error-aware aggregate evaluation."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import importlib
import math
from pathlib import Path
import sys
from mtimefact.common import ROOT, JsonLLM, load_json, read_jsonl, run_fingerprint, write_json, write_jsonl

STAGES = {
    "queries": ("mtimefact.evaluation.queries", "T-QueryScore", ["evaluation/evaluate_queries.txt"]),
    "retrieval": ("mtimefact.evaluation.retrieval", "TemporalPrecision@K", ["evaluation/evaluate_retrieval.txt"]),
    "explanation": ("mtimefact.evaluation.explanation", "LogicConsistency", ["evaluation/evaluate_alignment.txt", "evaluation/evaluate_quality.txt"]),
    "verdict": ("mtimefact.evaluation.verdict", "Accuracy", []),
}


def aligned_inputs(gold_path, predictions_path, demo=False):
    def read(path):
        result = {}
        for row in read_jsonl(path):
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
                raise ValueError(f"{path}: every row requires a nonempty string id")
            if row["id"] in result:
                raise ValueError(f"{path}: duplicate id {row['id']}")
            result[row["id"]] = row
        return result
    gold, predictions = read(gold_path), read(predictions_path)
    if not gold:
        raise ValueError("Gold dataset must not be empty")
    extra = predictions.keys() - gold.keys()
    if extra:
        raise ValueError("Prediction IDs absent from gold: " + ", ".join(sorted(extra)[:10]))
    for record in gold.values():
        if not isinstance(record.get("claim"), str) or not record["claim"].strip():
            raise ValueError(f"{record['id']}: gold requires a nonempty final claim")
        if record.get("status") == "draft_only_requires_gpt5":
            raise ValueError("Unrealized synthesis plans cannot be evaluated as final claims")
        if demo and not (record.get("fixture") is True and record.get("demo") is True):
            raise ValueError("--demo only accepts explicitly marked evaluation fixtures")
        if not demo and (record.get("demo") is True or record.get("fixture") is True):
            raise ValueError("Evaluation fixtures cannot be used as a live benchmark")
    if not demo and any(r.get("demo") is True or r.get("fixture") is True for r in predictions.values()):
        raise ValueError("Fixture predictions cannot be used in a live evaluation")
    return list(gold.values()), predictions


class BoundJudge:
    """Bind sample/stage/call identifiers so fixture and live APIs share the same interface."""
    def __init__(self, client, sample_id, stage, run_id="main", fixtures=None):
        self.client, self.sample_id, self.stage = client, sample_id, stage
        self.run_id, self.fixtures, self.calls = run_id, fixtures, 0

    def generate(self, role, prompt, payload):
        if role != "evaluator":
            raise ValueError("Evaluation uses the evaluator role, not the synthesis generator/auditor")
        index = self.calls
        self.calls += 1
        if self.fixtures is not None:
            values = self.fixtures.get(self.sample_id, {}).get(self.stage, [])
            if index >= len(values):
                raise ValueError(f"Missing test judge fixture for {self.sample_id}/{self.stage}/call{index}")
            return deepcopy(values[index])
        enriched = dict(payload, _evaluation={"sample_id": self.sample_id, "stage": self.stage,
                          "run_id": self.run_id, "call_index": index})
        return self.client.generate(role, prompt, enriched)


def summarize(rows):
    valid = [r["score"] for r in rows if r.get("score") is not None]
    count = len(rows)
    errors = count - len(valid)
    return {"n_gold": count, "n_scored": len(valid), "evaluation_errors": errors,
            "score": sum(valid)/count if count and not errors else None,
            "scored_subset_mean": sum(valid)/len(valid) if valid else None,
            "scoring_coverage": len(valid)/count if count else 0,
            "complete": bool(count) and not errors,
            "statuses": dict(Counter(r.get("status", "unknown") for r in rows))}


def evaluate_stage(stage, gold, predictions, config, options, fixtures=None):
    if fixtures is not None and not options.get("demo"):
        raise ValueError("Fixture judge responses require demo=True; cannot masquerade as live evaluation")
    if options.get("demo") and stage != "verdict" and fixtures is None:
        raise ValueError("Demo judged stages require explicit judge fixtures")
    module_name, metric, _ = STAGES[stage]
    evaluate = importlib.import_module(module_name).evaluate
    client = None if options.get("demo") or stage == "verdict" else JsonLLM(config, options.get("cache_dir", ".cache/evaluation"))
    rows = []
    for item in gold:
        judge = BoundJudge(client, item["id"], stage, options.get("run_id", "main"), fixtures)
        try:
            result = evaluate(item, predictions.get(item["id"]), judge, options)
            score = result.get("score")
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Stage returned an invalid metric; expected finite score in [0,1]")
        except (ValueError, TypeError, KeyError, RuntimeError, AttributeError) as exc:
            result = {"id": item["id"], "score": None, "status": "evaluation_error",
                      "error_type": type(exc).__name__, "reason": str(exc)}
        result.update(id=item["id"], stage=stage, hop_count=item.get("hop_count", "unknown"),
                      demo=options.get("demo", False), judge_calls=judge.calls)
        rows.append(result)
    groups = defaultdict(list)
    for result in rows:
        groups[str(result["hop_count"])].append(result)
    report = {"stage": stage, "metric": metric, "scale": "0..1", "overall": summarize(rows),
              "by_hop": {hop: summarize(group) for hop, group in sorted(groups.items())},
              "k": options.get("k", 5) if stage == "retrieval" else None,
              "demo": options.get("demo", False), "run_id": options.get("run_id", "main"),
              "missing_prediction_ids": [g["id"] for g in gold if g["id"] not in predictions],
              "notes": ["Missing/invalid model output counts as zero; the gold denominator is never reduced.",
                        "Judge or reference errors yield score=null; incomplete runs have no full-dataset metric.",
                        "scored_subset_mean is diagnostic only and must not replace the benchmark metric."]}
    if stage == "verdict":
        confusion = Counter()
        for row in rows:
            if row["score"] is not None:
                predicted = row.get("predicted_label")
                cell = ("true" if row["gold_label"] else "false") + "->" + (
                    "invalid_or_missing" if predicted is None else "true" if predicted else "false")
                confusion[cell] += 1
        report["confusion_matrix"] = dict(confusion)
    return rows, report


def add_arguments(parser, all_stages=False):
    parser.add_argument("--gold", type=Path, required=True, help="Trusted gold claims, graph, evidence, temporal windows")
    parser.add_argument("--predictions", type=Path, required=True, help="System queries/retrieval/explanation/verdict keyed by id")
    parser.add_argument("--output-dir" if all_stages else "--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"config/default.json")
    parser.add_argument("--cache-dir", default=str(ROOT/".cache/evaluation"))
    parser.add_argument("--k", type=int, default=5, help="Fixed retrieval denominator K")
    parser.add_argument("--run-id", default="main", help="Distinct IDs request independent judge calls; same ID uses cache")
    parser.add_argument("--demo", action="store_true", help="Use hand-authored judge test fixtures, not real evaluation")
    parser.add_argument("--judge-fixtures", type=Path, default=ROOT/"fixtures/evaluation/evaluation_judgements.json")
    parser.add_argument("--overwrite", action="store_true")


def context(args, need_judge=True):
    if args.k < 1:
        raise ValueError("K must be positive")
    config = load_json(args.config)
    if need_judge and not args.demo and "evaluator" not in config.get("llm", {}):
        raise ValueError("config.llm.evaluator must explicitly specify the evaluation judge (paper: GPT-5)")
    gold, predictions = aligned_inputs(args.gold, args.predictions, args.demo)
    fixtures = load_json(args.judge_fixtures) if args.demo and need_judge else None
    if fixtures is not None and not isinstance(fixtures, dict):
        raise ValueError("Judge fixtures must be a JSON object")
    options = {"demo": args.demo, "k": args.k, "run_id": args.run_id, "cache_dir": args.cache_dir}
    return config, gold, predictions, fixtures, options


def save_stage(stage, rows, report, output, args, config):
    inputs = [args.gold, args.predictions]
    if args.demo and stage != "verdict":
        inputs.append(args.judge_fixtures)
    report["run"] = run_fingerprint(config, inputs, STAGES[stage][2])
    write_jsonl(output, rows)
    write_json(str(output)+".report.json", report)


def stage_cli(stage):
    parser = argparse.ArgumentParser(description=STAGES[stage][1] + " evaluation; paper §A.3")
    add_arguments(parser)
    args = parser.parse_args()
    try:
        if not args.overwrite and any(p.exists() for p in (args.output, Path(str(args.output)+".report.json"))):
            raise ValueError("Output exists; choose another path or use --overwrite")
        config, gold, predictions, fixtures, options = context(args, need_judge=stage != "verdict")
        rows, report = evaluate_stage(stage, gold, predictions, config, options, fixtures)
        save_stage(stage, rows, report, args.output, args, config)
        print({"stage": stage, **report["overall"], "output": str(args.output)})
        return 0 if report["overall"]["complete"] else 2
    except (OSError, ValueError) as exc:
        print("Evaluation input error: " + str(exc), file=sys.stderr)
        return 1
