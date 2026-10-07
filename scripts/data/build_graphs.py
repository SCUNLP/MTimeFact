#!/usr/bin/env python3
"""Build entity graphs from the Wikipedia entity/fact dataset.

Deterministic first layer (no LLM), producing multiple graphs:

  1. co-mention graph  — edge A→B whenever a fact ABOUT A mentions entity B
                         (every entity gets connected; edges carry fact evidence)
  2. relation graph    — typed temporal triples extracted by curated English
                         patterns: (head, relation, tail, date, fact_id)
                         relations: born_on/born_year/died_on/died_year/is_a/
                         founded/co_founded/member_of/married_to/child_of/
                         parent_of/sibling_of/acted_in/directed/wrote/released/
                         won/nominated_for/leader_of/worked_for/educated_at/
                         located_in/occurred_in/served_as
  3. type subgraphs    — person_person / person_org / person_work / place /
                         event ... exported separately plus one merged graph
  4. analytics         — components, degrees, relation counts, cross-category
                         edges, most-connected entities

Outputs (data/graphs/):
  cooccurrence_graph.json, relation_graph.json, graphs_by_relation/*.json,
  graphs_by_type/*.json, entity_graph.json (merged), graph_report.json

Usage:
  python3 scripts/data/build_graphs.py \
      --entities data/wiki_entities/entities.jsonl \
      --facts data/wiki_facts.jsonl \
      --output-dir data/graphs
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9''\-\.]*")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")
MONTH_RE = "|".join(MONTHS)

STOP_SINGLE = {"from", "this", "that", "when", "while", "after", "before", "during",
               "there", "here", "these", "those", "their", "they", "what", "which",
               "where", "indian", "american", "chinese", "english", "french", "german",
               "russian", "japanese", "korean", "italian", "spanish", "mexican", "canadian",
               "australian", "european", "african", "asian", "national", "international",
               "college", "university", "city", "state", "world", "life", "war", "man",
               "woman", "king", "queen", "love", "home", "work", "game", "film", "music",
               "art", "history", "time", "people", "party", "company", "general", "black",
               "white", "great", "little", "young", "long", "short", "high", "low", "first",
               "united", "states", "county", "island", "river", "mountain", "house"}

OBJ = r"([A-Z][A-Za-z''\-\. ]{2,55}?)"
OBJ_END = (r"(?=[,;()]|\s+(?:and|but|or|who|which|where|when|in|on|at|from|by|for|with|as|of|to)\s"
           r"|[.!?]|$)")

RULES = [
    # (regex, relation, object_source, extra_key)
    (r"\bborn on ((?:%s) \d{1,2},? \d{4})" % MONTH_RE, "born_on", "date", None),
    (r"\bborn in (\d{4})", "born_year", "value", None),
    (r"\bdied on ((?:%s) \d{1,2},? \d{4})" % MONTH_RE, "died_on", "date", None),
    (r"\bdied in (\d{4})", "died_year", "value", None),
    (r"\b(?:is|was) an? ([a-z][a-z \-]{2,50}?)(?:[.,;]| who| and )", "is_a", "value", None),
    (r"\b(?:founded|co-founded|established|created) (?:the )?" + OBJ + OBJ_END, "founded", "entity", None),
    (r"\b(?:member of|joined|belonged to) (?:the )?" + OBJ + OBJ_END, "member_of", "entity", None),
    (r"\bmarried " + OBJ + OBJ_END, "married_to", "entity", None),
    (r"\b(?:son|daughter) of " + OBJ + OBJ_END, "child_of", "entity", None),
    (r"\b(?:father|mother) of " + OBJ + OBJ_END, "parent_of", "entity", None),
    (r"\b(?:brother|sister) of " + OBJ + OBJ_END, "sibling_of", "entity", None),
    (r"\b(?:played|portrayed|starred as) ([^.,;()]{2,50}?) in " + OBJ + OBJ_END, "acted_in", "entity", "role"),
    (r"\bdirected (?:the )?" + OBJ + OBJ_END, "directed", "entity", None),
    (r"\b(?:wrote|authored) (?:the )?" + OBJ + OBJ_END, "wrote", "entity", None),
    (r"\breleased (?:the )?" + OBJ + OBJ_END, "released", "entity", None),
    (r"\bwon (?:the )?" + OBJ + OBJ_END, "won", "entity", None),
    (r"\bnominated for (?:the )?" + OBJ + OBJ_END, "nominated_for", "entity", None),
    (r"\b(?:CEO|chairman|chairwoman|president|leader|head|director) of (?:the )?" + OBJ + OBJ_END, "leader_of", "entity", None),
    (r"\b(?:worked for|employed by|worked at) " + OBJ + OBJ_END, "worked_for", "entity", None),
    (r"\bgraduated from " + OBJ + OBJ_END, "educated_at", "entity", None),
    (r"\b(?:is |was |are |were )?(?:located|based|headquartered) in " + OBJ + OBJ_END, "located_in", "entity", None),
    (r"\b(?:took place|occurred) in " + OBJ + OBJ_END, "occurred_in", "entity", None),
    (r"\bserved as (?:the )?([a-z][^.,;]{3,60}?)(?:[.,;]| and | from |$)", "served_as", "value", None),
]


def normalize_name(value):
    text = value or ""
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)  # drop "(film)" style suffixes
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def name_variants(entity):
    variants = set()
    for key in ("label", "title"):
        value = entity.get(key) or ""
        base = re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()
        if len(base) >= 4 and base.lower() not in {"list", "main page"}:
            variants.add(base)
    return variants


class Matcher:
    """Case-sensitive token n-gram lookup against entity names.

    Single-token names are only used when reasonably distinctive (length >= 6
    and not a common sentence-start/word token), which prevents extreme false
    positives such as matching the TV series "From" inside the word "from".
    """

    def __init__(self, entities):
        self.map = defaultdict(list)
        for entity in entities:
            for name in name_variants(entity):
                key = re.sub(r"\s+", " ", name).strip()
                tokens = key.split()
                if len(tokens) == 1:
                    if len(key) < 6 or key.lower() in STOP_SINGLE:
                        continue
                elif len(key) < 4:
                    continue
                self.map[key].append(entity)
        self.map = {key: sorted({e["qid"]: e for e in value}.values(),
                                key=lambda e: -e.get("monthly_views_max", 0))
                    for key, value in self.map.items()}
        self.max_tokens = max((len(key.split()) for key in self.map), default=1)

    def find(self, text, exclude_qid=None):
        tokens = [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
        hits = []
        for i in range(len(tokens)):
            for n in range(1, self.max_tokens + 1):
                if i + n > len(tokens):
                    break
                phrase = " ".join(t[0] for t in tokens[i:i + n])
                entries = self.map.get(phrase)
                if not entries:
                    continue
                entry = entries[0]  # highest-view candidate; ambiguity flagged downstream
                if entry["qid"] == exclude_qid:
                    continue
                hits.append({"qid": entry["qid"], "label": entry["label"],
                             "start": tokens[i][1], "end": tokens[i + n - 1][2],
                             "phrase": text[tokens[i][1]:tokens[i + n - 1][2]],
                             "ambiguous": len(entries) > 1})
        # keep longest non-overlapping matches
        hits.sort(key=lambda h: (h["start"], -(h["end"] - h["start"])))
        kept, last_end = [], -1
        for hit in hits:
            if hit["start"] >= last_end:
                kept.append(hit)
                last_end = hit["end"]
        return kept


def date_of(text, fallback, precision_fallback):
    for match in re.finditer(r"\b((?:%s) \d{1,2},? \d{4})\b" % MONTH_RE, text):
        try:
            parsed = datetime.strptime(match.group(1).replace(",", ""), "%B %d %Y")
            return parsed.date().isoformat(), "day"
        except ValueError:
            continue
    years = re.findall(r"\b(1[89]\d{2}|20\d{2})\b", text)
    if years:
        return years[0], "year"
    return fallback, precision_fallback


def build_relation_edges(facts, matcher):
    edges, per_relation = [], Counter()
    for fact in facts:
        text = fact.get("fact") or ""
        entity = fact["entity"]
        for pattern, relation, source, extra_key in RULES:
            match = re.search(pattern, text)
            if not match:
                continue
            if relation == "acted_in":
                role, obj = match.group(1), match.group(2)
                extra = {"role": role.strip()} if extra_key else {}
            else:
                obj = match.group(1)
                extra = {}
            obj = obj.strip().strip('"').strip()
            if len(obj) < 3:
                continue
            tail_qid, tail_label, tail_type = None, obj, ("value" if source == "value" else "other")
            if source != "value":
                resolved = matcher.find(obj)
                if resolved:
                    tail_qid, tail_label = resolved[0]["qid"], resolved[0]["label"]
                    tail_type = "entity"
            date, precision = (fact.get("date"), fact.get("date_precision")) \
                if relation not in ("born_on", "died_on") else (None, None)
            if relation in ("born_on", "died_on"):
                date, precision = date_of(obj, fact.get("date"), fact.get("date_precision"))
            elif relation in ("born_year", "died_year"):
                date, precision = obj, "year"
            edges.append({
                "head": entity["qid"], "head_label": entity["label"],
                "head_category": entity["category"],
                "relation": relation,
                "tail": tail_qid, "tail_label": tail_label, "tail_type": tail_type,
                "date": date, "date_precision": precision,
                "fact_id": fact["id"], "fact": text[:200],
                **extra,
            })
            per_relation[relation] += 1
            break  # one relation per fact (first match wins)
    return edges, per_relation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entities", type=Path, default=Path("data/wiki_entities/entities.jsonl"))
    parser.add_argument("--facts", type=Path, default=Path("data/wiki_facts.jsonl"))
    parser.add_argument("--triples", type=Path, default=Path("data/wiki_triples.jsonl"),
                        help="Optional LLM triples JSONL (extract_triples_llm.py); merged when present")
    parser.add_argument("--output-dir", type=Path, default=Path("data/graphs"))
    args = parser.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    entities = [json.loads(l) for l in args.entities.read_text(encoding="utf-8").splitlines() if l.strip()]
    facts = [json.loads(l) for l in args.facts.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(json.dumps({"entities": len(entities), "facts": len(facts)}, ensure_ascii=False), flush=True)

    matcher = Matcher(entities)
    by_qid = {e["qid"]: e for e in entities}

    # ---- 1) co-mention graph
    mention_counts = Counter()
    mention_examples = {}
    for fact in facts:
        head = fact["entity"]["qid"]
        for hit in matcher.find(fact.get("fact") or "", exclude_qid=head):
            key = (head, hit["qid"])
            mention_counts[key] += 1
            mention_examples.setdefault(key, fact["id"])
    co_edges = [{"source": a, "target": b, "mentions": count,
                 "example_fact": mention_examples[(a, b)]} for (a, b), count in mention_counts.items()]
    co_nodes = sorted({qid for pair in mention_counts for qid in pair})
    print(json.dumps({"cooccurrence_nodes": len(co_nodes), "cooccurrence_edges": len(co_edges)},
                     ensure_ascii=False), flush=True)

    # ---- 2) relation graph (rules) + optional LLM triples merge
    rel_edges, per_relation = build_relation_edges(facts, matcher)
    for edge in rel_edges:
        edge["method"] = "rule"
    llm_edges = []
    if args.triples and args.triples.exists():
        for line in args.triples.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            triple = json.loads(line)
            obj = (triple.get("object") or "").strip()
            if not obj:
                continue
            tail_qid, tail_label, tail_type = None, obj, triple.get("object_type", "other")
            resolved = matcher.find(obj)
            if resolved:
                tail_qid, tail_label, tail_type = resolved[0]["qid"], resolved[0]["label"], "entity"
            llm_edges.append({
                "head": triple["head"], "head_label": triple["head_label"],
                "head_category": triple.get("head_category", "other"),
                "relation": triple["relation"],
                "tail": tail_qid, "tail_label": tail_label, "tail_type": tail_type,
                "date": triple.get("date"), "date_precision": triple.get("date_precision", "none"),
                "fact_id": triple.get("fact_id"), "fact": (triple.get("fact") or "")[:200],
                "method": "llm",
            })
    print(json.dumps({"relation_edges_rule": len(rel_edges), "relation_edges_llm": len(llm_edges),
                      "by_relation_rule": dict(per_relation.most_common(20))}, ensure_ascii=False), flush=True)
    all_rel_edges = rel_edges + llm_edges

    # ---- 3) type subgraphs + merged graph
    def edge_types(edge):
        head_cat = edge.get("head_category", "other")
        tail_cat = by_qid.get(edge.get("tail") or "", {}).get("category", edge.get("tail_type", "other"))
        return head_cat, tail_cat

    by_type = defaultdict(list)
    for edge in all_rel_edges:
        head_cat, tail_cat = edge_types(edge)
        by_type["%s_%s" % (head_cat, tail_cat)].append(edge)
    by_relation = defaultdict(list)
    for edge in all_rel_edges:
        by_relation[edge["relation"]].append(edge)

    def write_json(path, payload):
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    type_dir = out / "graphs_by_type"
    type_dir.mkdir(exist_ok=True)
    for name, edges in sorted(by_type.items()):
        write_json(type_dir / (name + ".json"),
                   {"graph": name, "edge_count": len(edges), "edges": edges[:20000]})
    rel_dir = out / "graphs_by_relation"
    rel_dir.mkdir(exist_ok=True)
    for name, edges in sorted(by_relation.items()):
        write_json(rel_dir / (name + ".json"),
                   {"graph": name, "edge_count": len(edges), "edges": edges[:20000]})

    write_json(out / "cooccurrence_graph.json",
               {"graph": "cooccurrence", "nodes": co_nodes, "edge_count": len(co_edges),
                "edges": sorted(co_edges, key=lambda e: -e["mentions"])[:30000]})
    write_json(out / "relation_graph.json",
               {"graph": "relations", "edge_count": len(all_rel_edges), "edges": all_rel_edges})
    write_json(out / "relation_graph_rule.json",
               {"graph": "relations_rule", "edge_count": len(rel_edges), "edges": rel_edges})
    if llm_edges:
        write_json(out / "relation_graph_llm.json",
                   {"graph": "relations_llm", "edge_count": len(llm_edges), "edges": llm_edges})
    merged_nodes = []
    used_qids = {e["head"] for e in all_rel_edges} | {e["tail"] for e in all_rel_edges if e["tail"]} | set(co_nodes)
    for qid in sorted(used_qids):
        entity = by_qid.get(qid)
        if entity:
            merged_nodes.append({"id": qid, "label": entity["label"], "category": entity["category"]})
    write_json(out / "entity_graph.json",
               {"graph": "merged", "node_count": len(merged_nodes),
                "nodes": merged_nodes,
                "relation_edges": len(all_rel_edges), "cooccurrence_edges": len(co_edges),
                "note": "Full edge lists live in relation_graph.json / cooccurrence_graph.json"})

    # ---- 4) analytics
    def components(edges):
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b in edges:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        return Counter(find(x) for x in parent)

    co_comp = components((a, b) for (a, b) in mention_counts)
    rel_undirected = set()
    for e in all_rel_edges:
        if e["tail"]:
            rel_undirected.add((e["head"], e["tail"]) if e["head"] < e["tail"] else (e["tail"], e["head"]))
    rel_comp = components(rel_undirected)
    degree = Counter()
    for (a, b) in mention_counts:
        degree[a] += 1
        degree[b] += 1
    top = [{"qid": qid, "label": by_qid.get(qid, {}).get("label", qid), "degree": count}
           for qid, count in degree.most_common(25)]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entities": len(entities), "facts": len(facts),
        "cooccurrence": {"nodes": len(co_nodes), "edges": len(co_edges),
                         "components": len(co_comp),
                         "largest_component": max(co_comp.values(), default=0)},
        "relations": {"edges": len(all_rel_edges), "rule_edges": len(rel_edges), "llm_edges": len(llm_edges),
                      "with_entity_tail": sum(1 for e in all_rel_edges if e["tail"]),
                      "with_date": sum(1 for e in all_rel_edges if e["date"]),
                      "by_relation": dict(Counter(e["relation"] for e in all_rel_edges).most_common(60)),
                      "components": len(rel_comp),
                      "largest_component": max(rel_comp.values(), default=0)},
        "type_pairs": {name: len(edges) for name, edges in sorted(by_type.items(), key=lambda kv: -len(kv[1]))[:20]},
        "top_connected": top,
        "outputs": [str(p.relative_to(out)) for p in sorted(out.rglob("*.json"))],
    }
    write_json(out / "graph_report.json", report)
    print(json.dumps({"status": "ok", "cooccurrence": report["cooccurrence"],
                      "relations": {k: report["relations"][k] for k in ("edges", "with_entity_tail", "components")},
                      "output": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
