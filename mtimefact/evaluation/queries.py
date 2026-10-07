"""Stage 1: T-QueryScore from paper Appendix A.3.1.

The paper specifies 0.3*Sdc + 0.3*Ssr + 0.4*Stp. The judge assesses the
whole query set for decomposition and semantic sufficiency, and each query
for temporal precision. Averaging temporal credit across queries, penalizing
an incorrect date, and requiring an ordered dependency trace are explicit
project choices, because the published metric omits these details.
"""
from __future__ import annotations

import math

from mtimefact.common import prompt


TEMPORAL_CREDIT = {"absolute": 1.0, "relative": 0.5, "missing": 0.0}


def _nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def _validate_gold(gold):
    """Missing annotations are evaluation errors, never a model failure."""
    if not isinstance(gold, dict) or not _nonempty_text(gold.get("id")):
        raise ValueError("query evaluation requires a gold record with a nonempty id")
    if not _nonempty_text(gold.get("claim")):
        raise ValueError("query evaluation requires the gold claim")
    graph = gold.get("reasoning_graph")
    if (not isinstance(graph, dict)
            or not isinstance(graph.get("nodes"), list) or not graph["nodes"]
            or not isinstance(graph.get("edges"), list) or not graph["edges"]
            or any(not isinstance(item, dict) for item in graph["nodes"] + graph["edges"])):
        raise ValueError("query evaluation requires annotated reasoning_graph nodes and edges")
    evidence = gold.get("evidence")
    if (not isinstance(evidence, list) or not evidence
            or any(not isinstance(item, dict) or not _nonempty_text(item.get("text"))
                   for item in evidence)):
        raise ValueError("query evaluation requires nonempty gold evidence text")


def _query_schema_error(queries):
    """Reject malformed traces that cannot be mapped to per-query judgements."""
    ids = []
    for query in queries:
        if not isinstance(query, dict):
            return "each query must be an object"
        if not _nonempty_text(query.get("id")):
            return "each query requires a nonempty id"
        if not _nonempty_text(query.get("text")):
            return "each query requires nonempty text"
        deps = query.get("depends_on")
        if not isinstance(deps, list) or any(not _nonempty_text(dep) for dep in deps):
            return "each query requires an explicit depends_on list (use [] for a root)"
        ids.append(query["id"])
    if len(set(ids)) != len(ids):
        return "duplicate query ids"
    return None


def _query_topology_error(queries):
    """Topology errors only invalidate Sdc; Ssr and Stp remain independently scored."""
    ids = [query["id"] for query in queries]
    all_ids = set(ids)
    for query in queries:
        if len(set(query["depends_on"])) != len(query["depends_on"]):
            return "duplicate query dependencies"
        if query["id"] in query["depends_on"]:
            return "self dependency in query graph"
        if any(dep not in all_ids for dep in query["depends_on"]):
            return "unknown query dependency"

    
    degree = {query["id"]: len(query["depends_on"]) for query in queries}
    children = {query_id: [] for query_id in ids}
    for query in queries:
        for dep in query["depends_on"]:
            children[dep].append(query["id"])
    ready = [query_id for query_id, count in degree.items() if count == 0]
    visited = 0
    while ready:
        query_id = ready.pop()
        visited += 1
        for child in children[query_id]:
            degree[child] -= 1
            if degree[child] == 0:
                ready.append(child)
    if visited != len(queries):
        return "cycle in query dependency graph"

    preceding = set()
    for query in queries:
        if any(dep not in preceding for dep in query["depends_on"]):
            return "query list is not in topological generation order"
        preceding.add(query["id"])
    return None


def _unit_score(value, name, binary=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"query judge {name} must be a numeric score")
    try:
        valid = math.isfinite(value) and 0 <= value <= 1
    except OverflowError:
        valid = False
    if not valid or (binary and value not in (0, 1)):
        interval = "0 or 1" if binary else "finite and in [0, 1]"
        raise ValueError(f"query judge {name} must be {interval}")
    return float(value)


def _validate_judgement(judgement, query_ids):
    if not isinstance(judgement, dict):
        raise ValueError("query judge must return an object")
    if type(judgement.get("gold_sufficient")) is not bool:
        raise ValueError("query judge requires boolean gold_sufficient")
    if not judgement["gold_sufficient"]:
        raise ValueError("query judge found insufficient gold annotations: "
                         + str(judgement.get("reason", "no reason supplied")))
    if not _nonempty_text(judgement.get("reason")):
        raise ValueError("query judge requires an assessment reason")
    s_dc = _unit_score(judgement.get("s_dc"), "s_dc", binary=True)
    s_sr = _unit_score(judgement.get("s_sr"), "s_sr")
    ratings = judgement.get("temporal_ratings")
    if not isinstance(ratings, list) or len(ratings) != len(query_ids):
        raise ValueError("query judge requires exactly one temporal rating for every query")
    by_id = {}
    for rating in ratings:
        if not isinstance(rating, dict):
            raise ValueError("query judge temporal ratings must be objects")
        query_id = rating.get("query_id")
        if not _nonempty_text(query_id) or query_id not in query_ids or query_id in by_id:
            raise ValueError("query judge temporal ratings have unknown or duplicate query ids")
        kind = rating.get("kind")
        if not isinstance(kind, str) or kind not in TEMPORAL_CREDIT:
            raise ValueError("query judge temporal kind must be absolute, relative, or missing")
        if type(rating.get("correct")) is not bool:
            raise ValueError("query judge temporal correctness must be boolean")
        if kind == "missing" and rating["correct"]:
            raise ValueError("query judge missing time must have correct=false")
        if not _nonempty_text(rating.get("reason")):
            raise ValueError("query judge requires a reason for each temporal rating")
        by_id[query_id] = TEMPORAL_CREDIT[kind] if rating["correct"] else 0.0
    s_tp = sum(by_id.values()) / len(query_ids)
    return {"s_dc": s_dc, "s_sr": s_sr, "s_tp": s_tp}


def _zero(gold, status, reason):
    return {"id": gold["id"], "stage": "query", "status": status, "score": 0.0,
            "components": {"s_dc": 0.0, "s_sr": 0.0, "s_tp": 0.0},
            "reason": reason, "judgement": None}


def evaluate(gold: dict, prediction: dict | None, judge, options: dict) -> dict:
    """Evaluate one trace; invalid judge output raises ValueError for the runner.

    ``judge`` must expose ``generate(role, prompt, payload)``. Production uses
    the unified GPT-5 evaluator, while offline examples inject a fixture judge.
    ``options`` is reserved for the shared runner; no demo scoring is hidden here.
    """
    _validate_gold(gold)
    if prediction is None:
        return _zero(gold, "missing_prediction", "no prediction for this gold id")
    if not isinstance(prediction, dict):
        return _zero(gold, "invalid_output", "prediction must be an object")
    if "id" in prediction and prediction["id"] != gold["id"]:
        return _zero(gold, "invalid_output", "prediction id does not match gold id")
    queries = prediction.get("queries")
    if queries is None or queries == []:
        return _zero(gold, "empty_output", "no generated queries")
    if not isinstance(queries, list):
        return _zero(gold, "invalid_output", "queries must be a list")
    error = _query_schema_error(queries)
    if error:
        return _zero(gold, "invalid_output", error)
    topology_error = _query_topology_error(queries)
    topology_check = {"valid": topology_error is None, "reason": topology_error}

    
    
    payload = {"task": "query_generation_accuracy", "id": gold["id"],
               "claim": gold["claim"], "hop_count": gold.get("hop_count"),
               "reference_date": gold.get("reference_date"),
               "time_window": gold.get("time_window"),
               "reasoning_graph": gold["reasoning_graph"],
               "evidence": gold["evidence"],
               "queries": [{key: query[key] for key in ("id", "text", "depends_on")}
                           for query in queries],
               "topology_check": topology_check,
               "temporal_scope": "required_for_all_queries"}
    judgement = judge.generate(role="evaluator", prompt=prompt("evaluation/evaluate_queries.txt"), payload=payload)
    components = _validate_judgement(judgement, {query["id"] for query in queries})
    if topology_error:
        
        
        components["s_dc"] = 0.0
    score = 0.3 * components["s_dc"] + 0.3 * components["s_sr"] + 0.4 * components["s_tp"]
    return {"id": gold["id"], "stage": "query", "status": "scored", "score": score,
            "components": components, "judgement": judgement,
            "topology_check": topology_check,
            "aggregation": "mean temporal credit over all generated queries"}
