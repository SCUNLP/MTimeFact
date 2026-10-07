"""Four observable inference stages for a prompt-based fact-checking baseline.

This implements the protocol of paper §3.3 with web retrieval (§C), not the
trained MTIMEFACTCHECK decomposer/hybrid solver/SFT-RL answer generator.
Every model payload is built from explicit allowlists. Gold annotations are
never used by these stages, including when a caller supplies a dataset row.
"""
import re

from mtimefact.common import prompt


STANCES = {"supported", "refuted", "insufficient_evidence"}
RETRIEVAL_STATUSES = {"answered", "insufficient_evidence", "skipped_dependency"}
_DOC_FIELDS = ("id", "url", "title", "text", "timestamp", "rank", "timestamp_kind", "timestamp_source")
_DOC_PROVENANCE_FIELDS = ("published_date_raw", "retrieved_at", "score", "text_truncated",
                          "text_original_length", "content_kind", "truncated_characters")


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _base(record):
    if not isinstance(record, dict) or not _text(record.get("id")) or not _text(record.get("claim")):
        raise ValueError("inference requires nonempty string id and claim")
    result = {"id": record["id"], "claim": record["claim"]}
    if record.get("reference_date") is not None:
        if not _text(record["reference_date"]):
            raise ValueError("reference_date must be a nonempty string when supplied")
        result["reference_date"] = record["reference_date"]
    return result


def _options(options):
    options = {} if options is None else options
    if not isinstance(options, dict):
        raise ValueError("inference options must be an object")
    result = {}
    for name, default, maximum in (("max_queries", 8, 32), ("k", 5, 20),
                                   ("max_document_chars", 12000, 1000000)):
        value = options.get(name, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer in [1, {maximum}]")
        result[name] = value
    return result


def _generate(llm, name, payload):
    result = llm.generate("reasoner", prompt("inference/" + name + ".txt"),
                          {"task": name, **payload})
    if not isinstance(result, dict):
        raise ValueError(f"{name}: model response must be a JSON object")
    return result


def _queries(value, maximum):
    if not isinstance(value, list) or not 1 <= len(value) <= maximum:
        raise ValueError(f"queries must contain between 1 and {maximum} items")
    seen, result = set(), []
    for item in value:
        if not isinstance(item, dict) or not _text(item.get("id")) or not _text(item.get("text")):
            raise ValueError("each query requires nonempty string id and text")
        identity, dependencies = item["id"], item.get("depends_on")
        if identity in seen:
            raise ValueError("query ids must be unique")
        if (not isinstance(dependencies, list) or any(not _text(dep) for dep in dependencies)
                or len(set(dependencies)) != len(dependencies)):
            raise ValueError("depends_on must be an explicit array of unique query ids")
        if any(dep not in seen for dep in dependencies):
            raise ValueError("query dependencies must refer to preceding queries; cycles are invalid")
        seen.add(identity)
        result.append({"id": identity, "text": item["text"], "depends_on": list(dependencies)})
    return result


def _documents(value, limit=None, preserve_metadata=False):
    if not isinstance(value, list):
        raise ValueError("retrieved_documents must be an array")
    result, identity_text = [], {}
    for document in value:
        if (not isinstance(document, dict) or not _text(document.get("id"))
                or not _text(document.get("text"))):
            raise ValueError("every retrieved document requires a nonempty id and text")
        identity = document["id"]
        if identity in identity_text and identity_text[identity] != document["text"]:
            raise ValueError("document id is ambiguous: different texts use the same id")
        identity_text[identity] = document["text"]
        fields = _DOC_FIELDS + _DOC_PROVENANCE_FIELDS if preserve_metadata else _DOC_FIELDS
        clean = {key: document[key] for key in fields if key in document}
        for key in ("url", "title", "timestamp", "timestamp_kind", "timestamp_source"):
            if clean.get(key) is not None and not isinstance(clean[key], str):
                raise ValueError(f"document {key} must be a string or null")
        if "rank" in clean and (type(clean["rank"]) is not int or clean["rank"] < 1):
            raise ValueError("document rank must be a positive integer")
        if limit is not None:
            clean["text"] = clean["text"][:limit]
        result.append(clean)
    return result


def _citations(value, documents, query_id=None):
    if not isinstance(value, list):
        raise ValueError("citations must be an array")
    by_id = {document["id"]: document["text"] for document in documents}
    result = []
    for item in value:
        if not isinstance(item, dict) or not _text(item.get("document_id")) or not _text(item.get("quote")):
            raise ValueError("each citation requires document_id and a nonempty exact quote")
        if item["document_id"] not in by_id or item["quote"] not in by_id[item["document_id"]]:
            raise ValueError("citation must reference a retrieved document and an exact substring")
        clean = {"document_id": item["document_id"], "quote": item["quote"]}
        if query_id is not None:
            if item.get("query_id") != query_id:
                raise ValueError("citation query_id does not match its retrieved document")
            clean["query_id"] = query_id
        result.append(clean)
    return result


def _answer(result, documents):
    if not all(key in result for key in ("status", "answer", "citations")):
        raise ValueError("subanswer requires explicit status, answer, and citations")
    status = result.get("status")
    if status not in ("answered", "insufficient_evidence"):
        raise ValueError("subanswer status must be answered or insufficient_evidence")
    citations = _citations(result.get("citations"), documents)
    answer = result.get("answer")
    if status == "answered" and (not _text(answer) or not citations):
        raise ValueError("an answered query requires a nonempty answer and exact evidence citations")
    if status == "insufficient_evidence" and answer is not None:
        raise ValueError("insufficient evidence requires answer=null, not a guessed answer")
    return answer, citations, status


def _unresolved(text, query_ids):
    
    
    if re.search(r"\{[^}]*\}|\{\{|\}\}|<[^>]+>|\[[^\]]+\]|\$\{|\b(?:TODO|TBD|UNKNOWN)\b", text, re.I):
        return True
    return any(re.search(r"(?<![A-Za-z0-9_])" + re.escape(identity) + r"(?![A-Za-z0-9_])", text)
               for identity in query_ids)


def generate_queries(record, llm, options=None):
    """Generate a validated, topologically ordered query plan from the claim."""
    settings = _options(options)
    result = _generate(llm, "generate_queries", {**_base(record), "max_queries": settings["max_queries"]})
    record["queries"] = _queries(result.get("queries"), settings["max_queries"])
    return record


def retrieve_evidence(record, llm, search, options=None):
    """Execute queries in order and retain ranked evidence plus cited subanswers.

    Dependencies are resolved using ancestors' evidence-backed answers. An
    unresolved branch is explicitly skipped. Empty retrieval is insufficient
    evidence, never a refutation. Partial documents survive a later model error.
    """
    settings, base = _options(options), _base(record)
    queries = _queries(record.get("queries"), settings["max_queries"])
    record["queries"] = queries
    by_id = {query["id"]: query for query in queries}
    for query in queries:
        query.update(executed_search_query=None, answer=None, answer_citations=[],
                     status="pending", retrieved_documents=[], search_metadata={})
        ancestors = set(query["depends_on"])
        pending = list(ancestors)
        while pending:
            for parent in by_id[pending.pop()]["depends_on"]:
                if parent not in ancestors:
                    ancestors.add(parent)
                    pending.append(parent)
        dependencies = [item for item in queries if item["id"] in ancestors]
        if any(item["status"] != "answered" for item in dependencies):
            query["status"] = "skipped_dependency"
            query["search_metadata"] = {"skip_reason": "an ancestor has insufficient or unresolved evidence"}
            continue
        query_spec = {key: query[key] for key in ("id", "text", "depends_on")}
        search_query = query["text"]
        if dependencies:
            dependency_payload = []
            for item in dependencies:
                documents = _documents(item["retrieved_documents"], settings["max_document_chars"])
                cited_ids = {citation["document_id"] for citation in item["answer_citations"]}
                dependency_payload.append({"id": item["id"], "answer": item["answer"],
                                           "citations": item["answer_citations"],
                                           "documents": [doc for doc in documents if doc["id"] in cited_ids]})
            resolved = _generate(llm, "resolve_query", {**base, "query": query_spec,
                                                        "dependencies": dependency_payload})
            if (type(resolved.get("resolved")) is not bool or not _text(resolved.get("reason"))
                    or "search_query" not in resolved):
                raise ValueError("query resolution requires boolean resolved and nonempty reason")
            if not resolved["resolved"]:
                if resolved.get("search_query") is not None:
                    raise ValueError("unresolved dependency requires search_query=null")
                query["status"] = "skipped_dependency"
                query["search_metadata"] = {"skip_reason": resolved["reason"]}
                continue
            search_query = resolved.get("search_query")
        if not _text(search_query) or _unresolved(search_query, by_id):
            raise ValueError("search query is empty or still contains unresolved placeholders/query ids")
        query["executed_search_query"] = search_query
        query["status"] = "searching"
        found = search.search(search_query, max_results=settings["k"],
                              request_context={"id": base["id"], "query_id": query["id"]})
        if not isinstance(found, dict) or not isinstance(found.get("metadata"), dict):
            raise ValueError("search must return documents and an object metadata")
        documents = _documents(found.get("documents"), preserve_metadata=True)
        
        if len(documents) > settings["k"]:
            raise ValueError("search returned more than the requested k documents")
        query["retrieved_documents"] = documents
        query["search_metadata"] = found["metadata"]
        query["status"] = "retrieved"
        if not documents:
            query["status"] = "insufficient_evidence"
            continue
        visible = _documents(documents, settings["max_document_chars"])
        response = _generate(llm, "answer_query", {**base, "query": query_spec,
                                                   "executed_search_query": search_query,
                                                   "documents": visible})
        answer, citations, status = _answer(response, visible)
        query.update(answer=answer, answer_citations=citations, status=status)
    return record


def _trace(record, settings):
    plans = _queries(record.get("queries"), settings["max_queries"])
    result = []
    for plan, raw in zip(plans, record["queries"]):
        if not all(key in raw for key in ("status", "answer", "answer_citations", "executed_search_query",
                                          "retrieved_documents")):
            raise ValueError("explanation requires a complete preceding retrieval trace")
        status = raw.get("status")
        if not isinstance(status, str) or status not in RETRIEVAL_STATUSES:
            raise ValueError("explanation requires completed retrieval or explicit insufficient evidence")
        documents = _documents(raw.get("retrieved_documents"))
        answer, citations, _ = _answer({"status": "answered" if status == "answered" else "insufficient_evidence",
                                       "answer": raw.get("answer"),
                                       "citations": raw.get("answer_citations")}, documents)
        executed = raw.get("executed_search_query")
        if status == "skipped_dependency":
            if executed is not None or documents or answer is not None or citations:
                raise ValueError("skipped dependency cannot have executed search or evidence")
        elif not _text(executed):
            raise ValueError("completed search requires executed_search_query")
        result.append({**plan, "executed_search_query": executed, "status": status,
                       "answer": answer, "answer_citations": citations,
                       "retrieved_documents": _documents(documents, settings["max_document_chars"])})
    return result


def _explanation(value, trace):
    if (not _text(value.get("explanation")) or not isinstance(value.get("stance"), str)
            or value["stance"] not in STANCES):
        raise ValueError("explanation requires text and supported/refuted/insufficient_evidence stance")
    gaps = value.get("evidence_gaps")
    if not isinstance(gaps, list) or any(not _text(gap) for gap in gaps):
        raise ValueError("evidence_gaps must be an array of nonempty strings")
    raw_citations = value.get("citations")
    if not isinstance(raw_citations, list):
        raise ValueError("explanation citations must be an array")
    by_id = {query["id"]: query for query in trace}
    citations = []
    for item in raw_citations:
        if not isinstance(item, dict) or not _text(item.get("query_id")) or item["query_id"] not in by_id:
            raise ValueError("explanation citation requires a known query_id")
        identity = item["query_id"]
        citations.extend(_citations([item], by_id[identity]["retrieved_documents"], identity))
    if value["stance"] != "insufficient_evidence" and not citations:
        raise ValueError("supported/refuted explanations require retrieved evidence citations")
    if value["stance"] == "supported" and (gaps or any(query["status"] != "answered" for query in trace)):
        raise ValueError("supported explanation cannot have evidence gaps or unresolved query branches")
    if value["stance"] == "insufficient_evidence" and not gaps:
        raise ValueError("insufficient evidence requires an explicit evidence gap")
    return {"explanation": value["explanation"], "explanation_citations": citations,
            "evidence_status": value["stance"], "evidence_gaps": list(gaps)}


def generate_explanation(record, llm, options=None):
    """Generate a concise cited verification summary, not private reasoning."""
    settings, base = _options(options), _base(record)
    trace = _trace(record, settings)
    result = _generate(llm, "generate_explanation", {**base, "queries": trace})
    record.update(_explanation(result, trace))
    return record


def predict_verdict(record, llm, options=None):
    """Classify the verified claim; explicit evidence gaps produce abstention."""
    settings, base = _options(options), _base(record)
    trace = _trace(record, settings)
    explanation = _explanation({"explanation": record.get("explanation"),
                                "stance": record.get("evidence_status"),
                                "citations": record.get("explanation_citations"),
                                "evidence_gaps": record.get("evidence_gaps")}, trace)
    result = _generate(llm, "predict_verdict", {**base, **explanation})
    label = result.get("predicted_label")
    if "predicted_label" not in result or (label is not None and type(label) is not bool):
        raise ValueError("predicted_label must be a JSON boolean or null abstention")
    expected = {"supported": True, "refuted": False, "insufficient_evidence": None}[explanation["evidence_status"]]
    if label is not expected:
        raise ValueError("verdict disagrees with cited explanation stance; insufficient evidence must abstain")
    if not _text(result.get("reason")):
        raise ValueError("verdict requires a nonempty concise reason")
    record.update(predicted_label=label, verdict_reason=result["reason"])
    return record
