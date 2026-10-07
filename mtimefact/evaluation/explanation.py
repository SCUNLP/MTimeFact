"""Stage 3: GPT-5 explanation/verdict alignment, followed by grounded quality.

Paper Appendix A.3.3 leaves the aggregation of quality dimensions unspecified.
This implementation uses their arithmetic mean in [0, 1], following the continuous
score described in Section 3.3. Alignment is binary and short-circuits quality.
Verdict correctness against the gold label is evaluated separately in Stage 4.
"""
import math

from mtimefact.common import prompt
from mtimefact.evaluation.verdict import normalize_label


DIMENSIONS = ("sufficiency", "factual_correctness", "temporal_plausibility")


def _unit_score(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"explanation judge {name} must be a finite number in [0, 1]")
    if not math.isfinite(value):
        raise ValueError(f"explanation judge {name} must be a finite number in [0, 1]")
    return float(value)


def _validate_quality_reference(gold):
    evidence = gold.get("evidence")
    if (not isinstance(evidence, list) or not evidence
            or any(not isinstance(item, dict) or not isinstance(item.get("text"), str)
                   or not item["text"].strip() for item in evidence)):
        raise ValueError("explanation quality requires nonempty gold evidence text")
    graph = gold.get("reasoning_graph")
    if (not isinstance(graph, dict)
            or any(not isinstance(graph.get(key), list) or not graph[key]
                   or any(not isinstance(item, dict) for item in graph[key])
                   for key in ("nodes", "edges"))):
        raise ValueError("explanation quality requires nonempty gold reasoning_graph nodes and edges")


def _result(gold, status, reason):
    return {"id": gold.get("id"), "stage": "explanation", "score": 0.0,
            "status": status, "reason": reason,
            "components": {"i_align": None, "i_quality": None,
                           **{name: None for name in DIMENSIONS},
                           "quality_evaluated": False, "gold_sufficient": None},
            "judgement": {"alignment": None, "quality": None}}


def evaluate(gold, prediction, judge, options=None):
    """Evaluate one example using ``judge.generate('evaluator', ...)``.

    ``gold`` is a normalized evaluation example with claim, evidence and
    reasoning_graph. ``prediction`` contains explanation, predicted_label and
    queries with retrieved_documents. ``options`` is accepted for compatibility
    with the other stages; no additional Stage 3 options are needed.

    Missing/invalid model outputs score zero without using the judge. Malformed
    judge outputs and insufficient reference evidence raise ValueError, allowing
    the runner to report evaluation failures separately from model failures.
    """
    if prediction is None:
        return _result(gold, "missing", "No prediction was supplied.")
    if not isinstance(prediction, dict):
        return _result(gold, "invalid_output", "Prediction must be a JSON object.")
    explanation = prediction.get("explanation")
    if explanation is None or isinstance(explanation, str) and not explanation.strip():
        return _result(gold, "missing", "No non-empty explanation was supplied.")
    if not isinstance(explanation, str):
        return _result(gold, "invalid_output", "Explanation must be a string.")
    try:
        predicted_label = normalize_label(prediction.get("predicted_label"))
    except ValueError as exc:
        return _result(gold, "invalid_output", str(exc))
    queries = prediction.get("queries", [])
    if not isinstance(queries, list) or any(not isinstance(q, dict) for q in queries):
        return _result(gold, "invalid_output", "Queries must be a list of JSON objects.")
    if any(not isinstance(q.get("retrieved_documents", []), list) for q in queries):
        return _result(gold, "invalid_output", "retrieved_documents must be a list.")
    if any(not isinstance(document, dict) for query in queries
           for document in query.get("retrieved_documents", [])):
        return _result(gold, "invalid_output", "Retrieved documents must be JSON objects.")

    result = _result(gold, "ok", "")
    context = {name: gold.get(name) for name in ("claim", "reference_date", "time_window")}
    
    
    alignment = judge.generate("evaluator", prompt("evaluation/evaluate_alignment.txt"), {
        **context, "explanation": explanation.strip(), "predicted_label": predicted_label})
    if not isinstance(alignment, dict):
        raise ValueError("explanation alignment judge must return a JSON object")
    aligned = alignment.get("i_align")
    if type(aligned) is not int or aligned not in (0, 1):
        raise ValueError("explanation judge i_align must be the integer 0 or 1")
    result["judgement"]["alignment"] = alignment
    result["components"]["i_align"] = aligned
    if not aligned:
        result["reason"] = "Alignment gate failed; quality was not evaluated."
        return result

    _validate_quality_reference(gold)
    quality = judge.generate("evaluator", prompt("evaluation/evaluate_quality.txt"), {
        **context, "explanation": explanation.strip(), "predicted_label": predicted_label,
        "supplied_retrieval": queries, "gold_evidence": gold.get("evidence", []),
        "gold_reasoning_graph": gold.get("reasoning_graph", {})})
    if not isinstance(quality, dict):
        raise ValueError("explanation quality judge must return a JSON object")
    if type(quality.get("gold_sufficient")) is not bool:
        raise ValueError("explanation judge gold_sufficient must be a boolean")
    scores = {name: _unit_score(quality.get(name), name) for name in DIMENSIONS}
    if not quality["gold_sufficient"]:
        raise ValueError("explanation quality cannot be evaluated: insufficient gold evidence/path")
    quality_score = sum(scores.values()) / len(DIMENSIONS)
    result["components"].update(scores)
    result["components"].update(i_quality=quality_score, quality_evaluated=True,
                                gold_sufficient=True)
    result["judgement"]["quality"] = quality
    result["score"] = aligned * quality_score
    result["reason"] = "Aligned explanation; quality is the arithmetic mean of three dimensions."
    return result
