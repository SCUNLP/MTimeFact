"""A.3.2 Temporal Precision@K: GPT-5 relevance AND a gold-window check.

Paper-defined: TP@K divides by K and requires both semantic and temporal hits.
Project choices (the paper does not specify these): inclusive calendar-day
windows, the calendar date in an explicitly supplied timestamp's own offset,
duplicate results waste a rank, and a macro mean over generated queries. Dates
must be supplied by the dataset/retriever; publication/event dates are not guessed.
"""
import re
from datetime import date, datetime
from urllib.parse import urlsplit, urlunsplit

from mtimefact.common import prompt


_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"(?:[Zz]|[+-]\d{2}:\d{2})\Z"
)


def _parse_date(value):
    """Return (source calendar day, parsed value), never silently assume a zone."""
    if not isinstance(value, str):
        raise ValueError("timestamp must be a full ISO date or a timezone-aware timestamp")
    try:
        if _DATE.fullmatch(value):
            parsed = date.fromisoformat(value)
            return parsed, parsed
        if _TIMESTAMP.fullmatch(value):
            normalized = value.replace("t", "T")
            if normalized[-1:] in ("Z", "z"):
                normalized = normalized[:-1] + "+00:00"
            
            if normalized[-3:-2] == ":" and int(normalized[-2:]) >= 60:
                raise ValueError("invalid timezone minute")
            parsed = datetime.fromisoformat(normalized)
            if parsed.utcoffset() is not None:
                return parsed.date(), parsed
    except (ValueError, OverflowError) as exc:
        raise ValueError("invalid ISO timestamp") from exc
    raise ValueError("timestamp must be a full ISO date or a timezone-aware timestamp")


def _window(value, name):
    if not isinstance(value, dict) or "start" not in value or "end" not in value:
        raise ValueError(f"{name}: gold window requires start and end")
    try:
        start, raw_start = _parse_date(value["start"])
        end, raw_end = _parse_date(value["end"])
    except ValueError as exc:
        raise ValueError(f"{name}: invalid gold window: {exc}") from exc
    if start > end or (isinstance(raw_start, datetime) and
                       isinstance(raw_end, datetime) and raw_start > raw_end):
        raise ValueError(f"{name}: reversed gold window")
    return start, end


def _canonical_url(value):
    """Conservative URL canonicalization; query/path contents remain unchanged."""
    try:
        parts = urlsplit(value)
        if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
            raise ValueError("document URL must be absolute HTTP(S)")
        if parts.username is not None or parts.password is not None:
            raise ValueError("document URL must not contain credentials")
        scheme = parts.scheme.lower()
        host = parts.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        port = parts.port
        if port is not None and (scheme, port) not in (("http", 80), ("https", 443)):
            host += f":{port}"
        return urlunsplit((scheme, host, parts.path or "/", parts.query, ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid document URL: {exc}") from exc


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _validate_prediction(prediction, gold, k):
    """Validate before scoring so malformed rows never disappear from a mean."""
    if prediction is None:
        return [], "missing_output", None
    if not isinstance(prediction, dict):
        return [], "invalid_output", "prediction must be an object"
    if "id" in prediction and prediction["id"] != gold.get("id"):
        return [], "invalid_output", "prediction id does not match gold id"
    if "queries" not in prediction:
        return [], "missing_output", None
    queries = prediction["queries"]
    if not isinstance(queries, list):
        return [], "invalid_output", "queries must be an array"
    if not queries:
        return [], "missing_output", None
    seen_queries = set()
    for query in queries:
        if not isinstance(query, dict) or not _nonempty_string(query.get("id")):
            return [], "invalid_output", "each query requires a nonempty string id"
        if query["id"] in seen_queries:
            return [], "invalid_output", "duplicate query id"
        seen_queries.add(query["id"])
        if not _nonempty_string(query.get("text")):
            return [], "invalid_output", "each query requires nonempty text"
        dependencies = query.get("depends_on", [])
        if not isinstance(dependencies, list) or any(not _nonempty_string(x) for x in dependencies):
            return [], "invalid_output", "depends_on must be an array of query ids"
        documents = query.get("retrieved_documents", [])
        if not isinstance(documents, list):
            return [], "invalid_output", "retrieved_documents must be an array"
        for document in documents[:k]:
            if not isinstance(document, dict) or not _nonempty_string(document.get("text")):
                return [], "invalid_output", "each document requires nonempty text"
            identity, url = document.get("id"), document.get("url")
            if identity is not None and not _nonempty_string(identity):
                return [], "invalid_output", "document id must be a nonempty string"
            if url is not None:
                if not _nonempty_string(url):
                    return [], "invalid_output", "document URL must be a nonempty string"
                try:
                    _canonical_url(url)
                except ValueError as exc:
                    return [], "invalid_output", str(exc)
            if identity is None and url is None:
                return [], "invalid_output", "each document requires an id or URL"
    return queries, "ok", None


def _relevance(judge, payload):
    response = judge.generate("evaluator", prompt("evaluation/evaluate_retrieval.txt"), payload)
    expected = {document["rank_id"] for document in payload["documents"]}
    if not isinstance(response, dict) or set(response) != {"relevance"}:
        raise ValueError("retrieval evaluator: response must contain only relevance")
    rows = response["relevance"]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("retrieval evaluator: incomplete relevance judgments")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"rank_id", "relevant"}:
            raise ValueError("retrieval evaluator: invalid relevance judgment")
        rank_id = row["rank_id"]
        if not isinstance(rank_id, str) or rank_id not in expected or rank_id in result:
            raise ValueError("retrieval evaluator: unknown or duplicate rank_id")
        if type(row["relevant"]) is not bool:
            raise ValueError("retrieval evaluator: relevant must be a JSON boolean")
        result[rank_id] = row["relevant"]
    if set(result) != expected:
        raise ValueError("retrieval evaluator: missing rank_id")
    return result


def evaluate(gold, prediction, judge, options=None):
    """Evaluate one claim. Invalid gold/judge output raises ValueError.

    Missing candidate output scores zero. Malformed candidate output scores zero
    with ``invalid_output``; it must not be silently omitted by the caller.
    ``options['k']`` defaults to 5 and stays the denominator for short result lists.
    """
    options = {} if options is None else options
    if not isinstance(options, dict):
        raise ValueError("retrieval options must be an object")
    k = options.get("k", 5)
    if type(k) is not int or k <= 0:
        raise ValueError("retrieval k must be a positive integer")
    if not isinstance(gold, dict) or not _nonempty_string(gold.get("claim")):
        raise ValueError("retrieval gold requires a nonempty claim")
    global_window = _window(gold.get("time_window"), "time_window")
    overrides = gold.get("query_time_windows", {})
    if not isinstance(overrides, dict) or any(not _nonempty_string(key) for key in overrides):
        raise ValueError("query_time_windows must map query ids to gold windows")
    parsed_overrides = {key: _window(value, f"query_time_windows.{key}")
                        for key, value in overrides.items()}
    metadata = {
        "aggregation": "macro_mean_over_generated_queries",
        "timestamp_convention": "explicit_timestamp_source_calendar_day",
        "window_boundaries": "inclusive",
        "window_authority": "gold_only",
        "duplicate_policy": "same_id_or_canonical_url_wastes_rank_no_backfill",
        "url_canonicalization": "lowercase_scheme_host_default_ports_empty_path_fragment_only",
        "missing_ranks": "zero_with_fixed_k_denominator",
        "project_choices": ["calendar_day_comparison", "duplicate_policy", "cross_query_macro_mean"],
    }
    queries, status, reason = _validate_prediction(prediction, gold, k)
    result = {"score": 0.0, "tp_at_k": 0.0, "k": k, "status": status,
              "query_scores": [], "metadata": metadata}
    if reason:
        result["reason"] = reason
    if status != "ok":
        return result

    for query_index, query in enumerate(queries):
        query_id = query["id"]
        window = parsed_overrides.get(query_id, global_window)
        window_source = "query_time_windows" if query_id in parsed_overrides else "time_window"
        all_documents = query.get("retrieved_documents", [])
        documents = all_documents[:k]
        ranks, judge_documents = [], []
        seen_ids, seen_urls = set(), set()
        for rank, document in enumerate(documents, 1):
            identity = document.get("id")
            url = _canonical_url(document["url"]) if document.get("url") else None
            duplicate = (identity is not None and identity in seen_ids) or (url is not None and url in seen_urls)
            if identity is not None:
                seen_ids.add(identity)
            if url is not None:
                seen_urls.add(url)
            reasons = []
            temporal_valid = False
            source_date = None
            timestamp = document.get("timestamp")
            if timestamp is None or timestamp == "":
                reasons.append("missing_timestamp")
            else:
                try:
                    source_date, _ = _parse_date(timestamp)
                    temporal_valid = window[0] <= source_date <= window[1]
                    if not temporal_valid:
                        reasons.append("outside_gold_window")
                except ValueError:
                    reasons.append("invalid_timestamp")
            rank_id = f"q{query_index}:r{rank}"
            record = {"rank": rank, "rank_id": rank_id, "document_id": identity,
                      "semantic_relevant": None, "temporal_valid": temporal_valid,
                      "source_calendar_day": source_date.isoformat() if source_date else None,
                      "hit": False, "reasons": reasons}
            if duplicate:
                reasons.append("duplicate_document")
            else:
                judge_documents.append({"rank_id": rank_id, "id": identity,
                                        "text": document["text"], "url": document.get("url")})
            ranks.append(record)
        relevance = {}
        if judge_documents:
            payload = {"claim": gold["claim"], "reference_date": gold.get("reference_date"),
                       "query": {"id": query_id, "text": query["text"],
                                 "depends_on": query.get("depends_on", [])},
                       "gold_evidence": gold.get("evidence", []),
                       "reasoning_graph": gold.get("reasoning_graph", {}),
                       "documents": judge_documents}
            relevance = _relevance(judge, payload)
        for record in ranks:
            if record["rank_id"] not in relevance:
                continue
            record["semantic_relevant"] = relevance[record["rank_id"]]
            if not record["semantic_relevant"]:
                record["reasons"].append("semantically_irrelevant")
            record["hit"] = record["semantic_relevant"] and record["temporal_valid"]
        hits = sum(record["hit"] for record in ranks)
        result["query_scores"].append({
            "query_id": query_id, "score": hits / k, "tp_at_k": hits / k, "hits": hits,
            "k": k, "status": "ok" if documents else "missing_output", "ranks": ranks,
            "window": {"start": window[0].isoformat(), "end": window[1].isoformat()},
            "window_source": window_source, "missing_rank_count": max(k - len(documents), 0),
            "ignored_beyond_k": max(len(all_documents) - k, 0),
        })
    result["score"] = result["tp_at_k"] = sum(row["score"] for row in result["query_scores"]) / len(queries)
    return result
