"""Conservative, source-grounded implementation of paper §§A.1.1 and A.4.

The paper does not publish its complete schema or prompts. This implementation
therefore records project choices and rejects unsupported model output.
It does not establish that a fact is objectively true: accepted edges remain
silver-standard, audited extractions whose exact source quotes are retained.
"""
from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from mtimefact.common import ROOT, prompt as read_prompt
MONTHS = ["", "January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
VERDICTS = {
    "true": "true", "correct": "true", "real": "true", "mostly true": "true",
    "真实": "true", "真": "true", "属实": "true",
    "false": "false", "fake": "false", "incorrect": "false", "mostly false": "false",
    "谣言": "false", "假": "false", "不实": "false", "虚假": "false",
    "mixed": "mixed", "mixture": "mixed", "misleading": "mixed", "partly true": "mixed",
    "部分属实": "mixed", "误导": "mixed",
}
AUDIT_FIELDS = ("entity_accuracy", "temporal_grounding", "verdict_consistent", "evidence_support")


def stable_id(prefix: str, value: Any) -> str:
    serial = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return prefix + hashlib.sha256(serial.encode("utf-8")).hexdigest()[:20]


def verdict(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return VERDICTS.get(re.sub(r"[\s_-]+", " ", value.strip().lower()))


def iso_date(value: Any) -> dt.date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("date must be a bounded ISO YYYY-MM-DD value")
    return dt.date.fromisoformat(value)


def time_window(label: Any) -> tuple:
    """Resolve an explicitly represented calendar anchor, never an inferred date."""
    if not isinstance(label, str):
        raise ValueError("time-node label must be a strict ISO year/month/day")
    if re.fullmatch(r"\d{4}", label):
        year = int(label)
        return dt.date(year, 1, 1), dt.date(year, 12, 31), "year"
    if re.fullmatch(r"\d{4}-\d{2}", label):
        year, month = map(int, label.split("-"))
        return dt.date(year, month, 1), dt.date(year, month, calendar.monthrange(year, month)[1]), "month"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", label):
        day = iso_date(label)
        return day, day, "day"
    raise ValueError("time-node label must be a strict ISO year/month/day")


def _absolute_grounded(date: dt.date, precision: str, text: str) -> bool:
    """Recognize unambiguous ISO, English named-month, and Chinese dates.

    Numerical locale-dependent dates such as 03/04/24 are deliberately rejected.
    This narrow deterministic gate supplements, rather than replaces, the audit.
    """
    year, month, day = date.year, date.month, date.day
    if precision == "year":
        return bool(re.search(r"(?<!\d)" + str(year) + r"(?!\d)", text))
    name = MONTHS[month]
    abbreviations = rf"{name[:3]}\.?" if month != 9 else r"Sept?\.?"
    month_name = rf"(?:{name}|{abbreviations})"
    if precision == "month":
        patterns = [rf"(?<!\d){year}-{month:02d}(?!\d)",
                    rf"\b{month_name}\s+{year}\b", rf"{year}年\s*0?{month}月"]
    else:
        patterns = [rf"(?<!\d){date.isoformat()}(?!\d)",
                    rf"\b{month_name}\s+0?{day}(?:st|nd|rd|th)?[,]?\s+{year}\b",
                    rf"\b0?{day}(?:st|nd|rd|th)?\s+{month_name}[,]?\s+{year}\b",
                    rf"{year}年\s*0?{month}月\s*0?{day}[日号]"]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def validate_interval(fact: Dict[str, Any], source_text: str) -> Dict[str, Any]:
    start, end = iso_date(fact.get("start")), iso_date(fact.get("end"))
    if start > end:
        raise ValueError("interval start is later than end")
    precision = fact.get("time_precision")
    if precision not in {"day", "month", "year"}:
        raise ValueError("time_precision must be day, month, or year")
    if precision == "year" and (start.month != 1 or start.day != 1 or end.month != 12 or end.day != 31):
        raise ValueError("year precision requires complete boundary years")
    if precision == "month" and (start.day != 1 or end.day != calendar.monthrange(end.year, end.month)[1]):
        raise ValueError("month precision requires complete boundary months")
    quote = fact.get("evidence_quote")
    if not isinstance(quote, str) or not quote.strip() or quote not in source_text:
        raise ValueError("fact evidence quote is not an exact nonempty source substring")
    time_quotes = fact.get("time_evidence_quotes", [])
    if not isinstance(time_quotes, list) or any(not isinstance(q, str) or not q.strip() or q not in source_text for q in time_quotes):
        raise ValueError("time evidence quotes must be exact nonempty source substrings")
    grounding = "\n".join([quote] + time_quotes)
    grounded = _absolute_grounded(start, precision, grounding) and _absolute_grounded(end, precision, grounding)
    anchor = fact.get("temporal_anchor")
    if not grounded and isinstance(anchor, dict) and precision == "day" and start == end:
        anchor_quote = anchor.get("quote")
        anchor_date = iso_date(anchor.get("date"))
        offset = anchor.get("offset_days")
        if (isinstance(anchor_quote, str) and anchor_quote.strip() and anchor_quote in source_text
                and isinstance(offset, int) and not isinstance(offset, bool)
                and _absolute_grounded(anchor_date, "day", anchor_quote)
                and start == anchor_date + dt.timedelta(days=offset)):
            
            numbers = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                       6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
            n = abs(offset)
            number_pattern = rf"(?:{n}|{numbers[n]})" if n in numbers else str(n)
            direction = r"(?:after|later|following)" if offset >= 0 else r"(?:before|earlier|prior)"
            relative = rf"\b{number_pattern}\s+days?\s+{direction}\b"
            chinese = rf"{n}天(?:后|之后)" if offset >= 0 else rf"{n}天(?:前|之前)"
            grounded = bool(re.search(relative, quote, re.I) or re.search(chinese, quote))
            if grounded:
                time_quotes = list(dict.fromkeys(time_quotes + [anchor_quote]))
    if not grounded:
        raise ValueError("dates are not grounded in quoted evidence; ambiguous/unanchored dates rejected")
    return {"start": start.isoformat(), "end": end.isoformat(), "time_precision": precision,
            "time_evidence_quotes": time_quotes,
            "temporal_semantics": "inclusive_bounded_interval" if precision == "day" else "uncertain_within_bounds"}


def canonical_node_id(entity: Dict[str, Any], raw: Dict[str, Any]) -> str:
    """Trust a QID only when the raw record supplies a supporting entity link.

    Otherwise avoid fuzzy name matching. Repeated exact entity keys can merge
    only with an explicit disambiguation, or remain scoped to the source.
    """
    if entity.get("type") == "time":
        time_window(entity.get("label"))  
        return "temporal:" + entity["label"]
    qid = entity.get("canonical_id", "")
    links = raw.get("entity_links", {})
    if isinstance(links, dict) and re.fullmatch(r"Q[1-9]\d*", str(qid)):
        if links.get(entity.get("label")) == qid or links.get(entity.get("id")) == qid:
            return "wikidata:" + qid
    disambiguation = str(entity.get("disambiguation", "")).strip()
    scope = disambiguation if disambiguation else raw["id"]
    return stable_id("entity:", [entity.get("type"), raw.get("language", "und"), entity.get("label"), scope])


def audit_passes(audit: Any, threshold: float) -> bool:
    if not isinstance(audit, dict) or audit.get("accepted") is not True:
        return False
    confidence = audit.get("confidence")
    return (isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
            and threshold <= confidence <= 1 and all(audit.get(k) is True for k in AUDIT_FIELDS))


def load_relations(path: Optional[Path] = None) -> Dict[str, Any]:
    return json.loads((path or ROOT / "config" / "relations.json").read_text(encoding="utf-8"))


def _prompt(name: str) -> str:
    return read_prompt("data/" + name)


def _source(raw: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(raw)
    clean["id"] = str(raw.get("id") or raw.get("source_id") or "").strip()
    if not clean["id"]:
        raise ValueError("record requires an id/source_id")
    if not isinstance(raw.get("text"), str) or not raw["text"].strip():
        raise ValueError("record requires nonempty source text")
    return clean


def wikipedia_context(node: Dict[str, Any], language: str, llm: Any,
                      cache_dir: Path, threshold: float, user_agent: str) -> Optional[Dict[str, Any]]:
    """Retrieve and audit optional context. Never convert a summary into edges."""
    lang = "zh" if language.startswith("zh") else "en"
    params = {"action": "query", "format": "json", "formatversion": "2", "redirects": "1",
              "titles": node["label"], "prop": "extracts|pageprops|revisions", "exintro": "1",
              "explaintext": "1", "rvprop": "ids|timestamp", "rvlimit": "1"}
    url = "https://" + lang + ".wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / (stable_id("wiki_", url) + ".json")
    if cache.exists():
        snapshot = json.loads(cache.read_text(encoding="utf-8"))
    else:
        request = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
        snapshot = {"retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(), "response": data}
        cache.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    pages = snapshot["response"].get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing") or "disambiguation" in pages[0].get("pageprops", {}):
        return None
    page = pages[0]
    extract = page.get("extract", "")
    if not extract:
        return None
    revision = (page.get("revisions") or [{}])[0]
    source = {"text": extract, "url": "https://" + lang + ".wikipedia.org/?curid=" + str(page["pageid"]),
              "page_id": page["pageid"], "title": page.get("title"), "revision_id": revision.get("revid"),
              "revision_at": revision.get("timestamp"), "retrieved_at": snapshot.get("retrieved_at"),
              "wikidata_id": page.get("pageprops", {}).get("wikibase_item")}
    refined = llm.generate("generator", _prompt("wiki_refine.txt"), {"entity": node, "source": source})
    quotes = refined.get("evidence_quotes", []) if isinstance(refined, dict) else []
    if (not refined.get("entity_match") or not isinstance(refined.get("summary"), str)
            or not refined["summary"].strip() or not isinstance(quotes, list) or not quotes
            or any(not isinstance(q, str) or not q or q not in extract for q in quotes)):
        return None
    check = llm.generate("auditor", _prompt("wiki_refine.txt") + "\nAUDIT MODE: return {accepted: boolean, confidence: number, entity_match: boolean, summary_supported: boolean}. Independently reject incorrect entity identity or any unsupported summary detail.",
                         {"entity": node, "source": source, "proposed_context": refined})
    if not (check.get("accepted") is True and check.get("entity_match") is True
            and check.get("summary_supported") is True and isinstance(check.get("confidence"), (int, float))
            and threshold <= check["confidence"] <= 1):
        return None
    return {"summary": refined["summary"], "evidence_quotes": quotes, "source": source,
            "audit": check, "usage": "background_context_only_not_temporal_graph_edges"}


def build_graph(records: Iterable[Dict[str, Any]], llm: Any = None, *, demo: bool = False,
                config: Optional[Dict[str, Any]] = None, cache_dir: Optional[Path] = None,
                enrich_wikipedia: bool = False) -> tuple:
    config = config or {}
    graph_config = config.get("graph", {})
    threshold = float(graph_config.get("audit_min_confidence", 0.9))
    if not 0 <= threshold <= 1:
        raise ValueError("audit_min_confidence must be within [0, 1]")
    if not demo and llm is None:
        raise ValueError("live graph construction requires an LLM client")
    if demo and enrich_wikipedia:
        raise ValueError("--demo is offline; Wikipedia enrichment requires a live run")
    schema = load_relations()
    allowed_relations = {r["id"] for r in schema["relations"]}
    nodes, edges, sources = {}, {}, []
    report = {"input_records": 0, "accepted_records": 0, "time_robust_records": [],
              "rejections": [], "audits": [], "wikipedia_errors": [], "contradictions": []}
    seen_sources = set()
    for original in records:
        report["input_records"] += 1
        source_id = str(original.get("id") or original.get("source_id") or "unknown")
        try:
            raw = _source(original)
            source_id = raw["id"]
            if source_id in seen_sources:
                raise ValueError("duplicate source id")
            seen_sources.add(source_id)
            
            sources.append(raw)
            scraped_verdict = raw.get("source_verdict") or raw.get("verdict")
            raw_verdict = verdict(scraped_verdict)
            if scraped_verdict and raw_verdict is None:
                raise ValueError("unknown/unsupported source verdict")
            if demo:
                if raw.get("fixture") is not True and raw.get("is_fixture") is not True:
                    raise ValueError("--demo only accepts explicitly marked fixture records")
                draft = raw.get("pre_extracted")
                if not isinstance(draft, dict):
                    raise ValueError("demo fixture requires pre_extracted")
            else:
                draft = llm.generate("generator", _prompt("extract.txt"), {"source": raw, "relation_schema": schema})
            if not isinstance(draft, dict):
                raise ValueError("extraction is not an object")
            if raw_verdict is None:
                raw_verdict = verdict(draft.get("source_verdict"))
                verdict_quote = draft.get("verdict_evidence_quote")
                if (raw_verdict is None or not isinstance(verdict_quote, str) or not verdict_quote.strip()
                        or verdict_quote not in raw["text"]):
                    raise ValueError("missing source verdict requires an exact article-body verdict evidence quote")
                raw["extracted_verdict"] = {"value": raw_verdict, "quote": verdict_quote,
                                            "status": "pending_independent_audit"}
            elif verdict(draft.get("source_verdict")) != raw_verdict:
                raise ValueError("extracted source verdict disagrees with scraped source verdict")
            if draft.get("time_sensitive") is False:
                audit = raw.get("pre_audit") if demo else llm.generate("auditor", _prompt("audit_raw.txt"),
                            {"source": raw, "extraction": draft, "validated_edges": []})
                report["audits"].append({"source_id": source_id, "audit": audit, "demo": demo})
                if not audit_passes(audit, threshold):
                    raise ValueError("time-robust classification/verdict audit failed")
                if "extracted_verdict" in raw:
                    raw["extracted_verdict"]["status"] = "audited"
                report["time_robust_records"].append({"source_id": source_id, "claim": draft.get("claim"),
                                                    "source_verdict": raw_verdict,
                                                    "classification": "time_robust",
                                                    "source": raw})
                continue
            if draft.get("time_sensitive") is not True:
                raise ValueError("missing explicit time-sensitive classification")
            entities = draft.get("entities")
            facts = draft.get("facts")
            if not isinstance(entities, list) or not isinstance(facts, list) or not facts:
                raise ValueError("extraction requires entities and nonempty verified facts")
            local_nodes, local_map, candidate_edges = {}, {}, []
            for ent in entities:
                if not isinstance(ent, dict) or any(not isinstance(ent.get(k), str) or not ent[k].strip() for k in ("id", "label", "type", "domain")):
                    raise ValueError("invalid entity id/label/type/domain")
                if ent["id"] in local_map:
                    raise ValueError("duplicate local entity id")
                key = canonical_node_id(ent, raw)
                node = {"id": key, "label": ent["label"], "type": ent["type"], "domain": ent["domain"],
                        "domains": [ent["domain"]], "language": raw.get("language", "und"),
                        "source_ids": [source_id], "disambiguation": ent.get("disambiguation", "")}
                if key.startswith("wikidata:"):
                    node["canonical_id"] = key.split(":", 1)[1]
                if node["type"] == "time":
                    window_start, window_end, window_precision = time_window(node["label"])
                    node.update({"start": window_start.isoformat(), "end": window_end.isoformat(),
                                 "time_precision": window_precision, "language": "und"})
                local_nodes[key] = node
                local_map[ent["id"]] = key
            for index, fact in enumerate(facts):
                if not isinstance(fact, dict):
                    raise ValueError("fact must be an object")
                if fact.get("head") not in local_map or fact.get("tail") not in local_map:
                    raise ValueError("fact endpoint does not refer to an extracted entity")
                if fact.get("relation") not in allowed_relations:
                    raise ValueError("unknown relation: " + str(fact.get("relation")))
                if fact.get("polarity") not in {"positive", "negative"}:
                    raise ValueError("fact polarity must be explicit positive or negative")
                if fact.get("temporal_kind", "event") not in {"event", "state"}:
                    raise ValueError("temporal_kind must be event or state")
                interval = validate_interval(fact, raw["text"])
                head, tail = local_map[fact["head"]], local_map[fact["tail"]]
                head_node, tail_node = local_nodes[head], local_nodes[tail]
                if fact["relation"] == "occurred_during":
                    if (head_node["type"] != "event" or tail_node["type"] != "time"
                            or fact.get("temporal_kind", "event") != "event"
                            or fact["polarity"] != "positive"):
                        raise ValueError("occurred_during requires a positive event occurrence pointing to a time node")
                    if not (tail_node["start"] <= interval["start"] <= interval["end"] <= tail_node["end"]):
                        raise ValueError("occurred_during fact bounds lie outside the represented time-node window")
                elif head_node["type"] == "time" or tail_node["type"] == "time":
                    raise ValueError("time nodes can only be grounded as occurred_during tails")
                edge = {"id": stable_id("edge:", [head, fact["relation"], tail, interval["start"], interval["end"], interval["time_precision"], fact["polarity"], fact.get("temporal_kind", "event")]),
                        "head": head, "relation": fact["relation"], "tail": tail,
                        **interval, "polarity": fact["polarity"], "source_ids": [source_id],
                        "temporal_kind": fact.get("temporal_kind", "event"),
                        "evidence": [{"source_id": source_id, "quote": fact["evidence_quote"], "kind": "fact"}],
                        "source_verdict": raw_verdict, "source_verdicts": {source_id: raw_verdict}}
                edge["evidence"].extend({"source_id": source_id, "quote": q, "kind": "temporal"} for q in interval["time_evidence_quotes"])
                if fact.get("temporal_anchor"):
                    edge["temporal_anchor"] = fact["temporal_anchor"]
                candidate_edges.append(edge)
            audit = raw.get("pre_audit") if demo else llm.generate("auditor", _prompt("audit_raw.txt"),
                        {"source": raw, "extraction": draft, "validated_edges": candidate_edges})
            report["audits"].append({"source_id": source_id, "audit": audit, "demo": demo})
            if not audit_passes(audit, threshold):
                raise ValueError("independent raw-data audit failed or confidence is below threshold")
            if "extracted_verdict" in raw:
                raw["extracted_verdict"]["status"] = "audited"
            for key, node in local_nodes.items():
                if key not in nodes:
                    nodes[key] = node
                else:
                    for field in ("source_ids", "domains"):
                        nodes[key][field] = sorted(set(nodes[key][field] + node[field]))
            for edge in candidate_edges:
                key = edge["id"]
                if key not in edges:
                    edges[key] = edge
                else:
                    edges[key]["source_ids"] = sorted(set(edges[key]["source_ids"] + edge["source_ids"]))
                    edges[key]["source_verdicts"].update(edge["source_verdicts"])
                    edges[key]["evidence"].extend(e for e in edge["evidence"] if e not in edges[key]["evidence"])
            report["accepted_records"] += 1
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            report["rejections"].append({"source_id": source_id, "reason": str(exc)})
    
    
    by_triple = {}
    quarantine = set()
    for edge in edges.values():
        triple = (edge["head"], edge["relation"], edge["tail"])
        for previous in by_triple.get(triple, []):
            if (edge["polarity"] != previous["polarity"]
                    and max(edge["start"], previous["start"]) <= min(edge["end"], previous["end"])):
                quarantine.update([edge["id"], previous["id"]])
                report["contradictions"].append({"reason": "opposite polarity with overlapping bounded dates",
                                                 "edges": [previous, edge]})
        by_triple.setdefault(triple, []).append(edge)
    edges = {key: edge for key, edge in edges.items() if key not in quarantine}
    if enrich_wikipedia:
        for node in nodes.values():
            if node["type"] == "time":
                continue
            try:
                context = wikipedia_context(node, node["language"], llm,
                          (cache_dir or ROOT / ".cache" / "llm") / "wikipedia", threshold,
                          config.get("wikipedia", {}).get("user_agent", "MTimeFact/1.0 (research; https://github.com/SCUNLP/MTimeFact)"))
                if context:
                    node["wikipedia"] = context
            except Exception as exc:
                report["wikipedia_errors"].append({"node_id": node["id"], "reason": str(exc)})
    used = {edge[endpoint] for edge in edges.values() for endpoint in ("head", "tail")}
    graph = {"schema_version": 1,
             "metadata": {"implementation": "independent_paper_based", "demo": demo,
                          "relation_schema": schema["name"], "audit_min_confidence": threshold,
                          "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                          "date_bounds": "inclusive", "broad_precision": "uncertainty_not_continuous_validity",
                          "evidence_standard": "silver_standard_with_source_quotes", "wikipedia_enriched": enrich_wikipedia,
                          "quarantined_edges": len(quarantine),
                          "prompt_hashes": {name: hashlib.sha256(_prompt(name).encode()).hexdigest()
                                            for name in ("extract.txt", "audit_raw.txt", "wiki_refine.txt")},
                          "relation_schema_hash": hashlib.sha256(json.dumps(schema, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                          "models": {role: {k: v for k, v in spec.items() if k in {"provider", "model", "base_url", "max_output_tokens"}}
                                     for role, spec in config.get("llm", {}).items()},
                          "accepted_records": report["accepted_records"], "input_records": report["input_records"]},
             "nodes": [nodes[key] for key in sorted(used)],
             "edges": [edges[key] for key in sorted(edges)], "sources": sources}
    return graph, report
