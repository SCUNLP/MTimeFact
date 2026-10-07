"""Evidence-preserving simple-path sampling and claim validation.

A hop is one evidence-backed edge, not an invented semantic shortcut. Walks may
use incoming edges; stored subject/relation/object directions never change.
"""
from collections import defaultdict
from datetime import date
import math
import random
import re
from decimal import Decimal, InvalidOperation
from mtimefact.common import stable_id

STRATEGIES = {
    "temporal_synchronization": "Time anchor: explicitly compare facts in the same stated time window.",
    "person_identity_fusion": "Person anchor: connect a person's distinct documented roles.",
    "geospatial_coincidence": "Location anchor: connect facts through the same verified place.",
    "institutional_affiliation": "Organization bridge: connect through a documented organization affiliation.",
    "artifact_adaptation": "Creative-work bridge: connect an original and its adaptation.",
    "genealogical_linkage": "Family bridge: connect through documented marriage or kinship.",
    "event_causality": "Causal bridge: causality must be explicit in evidence, not inferred from order.",
    "role_parallelism": "Parallel roles: compare explicitly corresponding roles, without implying causation."
}
PERTURBATIONS = {
    "entity_hallucination": {"head", "tail"},
    "chronological_distortion": {"time"},
    "relational_corruption": {"relation"},
    "contextual_contradiction": {"premise"},
    "data_perturbation": {"value"},
    "sentiment_inversion": {"polarity"},
}


def interval(edge):
    start, end = date.fromisoformat(edge["start"]), date.fromisoformat(edge["end"])
    if end < start:
        raise ValueError("Reversed interval")
    return start.toordinal(), end.toordinal() + 1  


def path_time_window(path):
    """Return the inclusive bounding window covering every edge in a path.

    The window is the union envelope of the path's evidence intervals. It is
    deliberately distinct from ``reference_date``: sequential or independent
    paths may contain facts on different dates, while evaluation still needs a
    window that covers all source evidence.
    """
    if not isinstance(path, dict) or not isinstance(path.get("edges"), list) or not path["edges"]:
        raise ValueError("Path must contain at least one edge")
    spans = [interval(edge) for edge in path["edges"]]
    start = date.fromordinal(min(span[0] for span in spans))
    end = date.fromordinal(max(span[1] for span in spans) - 1)
    return {"start": start.isoformat(), "end": end.isoformat()}


def temporal_valid(edges, mode):
    spans = [interval(e) for e in edges]
    if mode == "reverse_sequence":
        return all(spans[i][0] >= spans[i + 1][0] for i in range(len(spans) - 1))
    if mode == "sequence":
        return all(spans[i][0] <= spans[i + 1][0] for i in range(len(spans) - 1))
    if mode == "parallel":
        
        if any(e.get("time_precision") != "day" for e in edges):
            return False
        return all(max(a[0], b[0]) < min(a[1], b[1]) for a, b in zip(spans, spans[1:]))
    return mode == "independent"


def relation_temporal_mode(edges, walk_nodes):
    """Chronological direction follows relation semantics, not a randomly selected tag.

    A--caused_by-->B goes back in causal order. Walking an incoming edge goes
    the other way. Conflicting order directions are rejected conservatively.
    """
    directions = set()
    for i, edge in enumerate(edges):
        rel = edge["relation"]
        if rel in {"causes", "predecessor_of", "caused_by", "subsequent_to"}:
            semantic = -1 if rel in {"caused_by", "subsequent_to"} else 1
            traversal = 1 if walk_nodes[i] == edge["head"] else -1
            directions.add(semantic * traversal)
    if len(directions) > 1:
        return None
    if any(e["relation"] == "parallel_to" for e in edges):
        if not temporal_valid(edges, "parallel"):
            return None
    if directions:
        mode = "sequence" if 1 in directions else "reverse_sequence"
        return mode if temporal_valid(edges, mode) else None
    return "parallel" if any(e["relation"] == "parallel_to" for e in edges) else "independent"


def strategy_candidates(nodes, edges):
    """Conservative candidate tags; semantic applicability is still independently audited."""
    types = {n.get("type", "").lower() for n in nodes}
    relations = {e["relation"] for e in edges}
    found = []
    if len({e["start"][:4] for e in edges}) == 1 and all(e["start"][:4] == e["end"][:4] for e in edges):
        found.append("temporal_synchronization")
    if types & {"person", "human"}:
        found.append("person_identity_fusion")
    if types & {"location", "place", "city", "country"}:
        found.append("geospatial_coincidence")
    if relations & {"member_of", "headquartered_in", "employed_by", "founded"}:
        found.append("institutional_affiliation")
    if "adapted_from" in relations:
        found.append("artifact_adaptation")
    if relations & {"married_to", "relative_of"}:
        found.append("genealogical_linkage")
    if relations & {"caused_by", "causes"}:
        found.append("event_causality")
    if "parallel_to" in relations:
        found.append("role_parallelism")
    return found


class PathSampler:
    def __init__(self, graph, seed=42, degree_power=1.0):
        self.nodes = {n["id"]: n for n in graph["nodes"]}
        self.adj = defaultdict(list)
        self.rng = random.Random(seed)
        if not math.isfinite(degree_power) or degree_power < 0:
            raise ValueError("degree_power must be finite and nonnegative")
        self.degree_power = degree_power
        edge_ids = set()
        for edge in graph["edges"]:
            if edge["id"] in edge_ids:
                raise ValueError("Duplicate graph edge ID")
            edge_ids.add(edge["id"])
            interval(edge)
            if not edge.get("evidence") or not edge.get("source_ids"):
                raise ValueError("Every graph edge must have evidence and sources")
            if edge["head"] not in self.nodes or edge["tail"] not in self.nodes:
                raise ValueError("Dangling graph edge")
            if edge["head"] == edge["tail"]:
                continue
            self.adj[edge["head"]].append((edge["tail"], edge))
            self.adj[edge["tail"]].append((edge["head"], edge))
        self.anchors = sorted(self.adj)
        self.weights = [len(self.adj[n]) ** degree_power for n in self.anchors]

    def sample(self, hops, strategy=None, search_budget=10000):
        if hops not in (2, 3, 4, 5):
            raise ValueError("hops must be 2..5")
        if not self.anchors:
            return None
        start = self.rng.choices(self.anchors, weights=self.weights)[0]
        visited, path, budget = [start], [], [search_budget]

        def walk(node):
            if budget[0] <= 0:
                return None
            budget[0] -= 1
            if len(path) == hops:
                nodes = [self.nodes[n] for n in visited]
                choices = strategy_candidates(nodes, path)
                if strategy and strategy not in choices:
                    return None
                if not choices:
                    return None
                selected = strategy or self.rng.choice(choices)
                mode = relation_temporal_mode(path, visited)
                if mode is None or not temporal_valid(path, mode):
                    return None
                return {"id": stable_id("path_", sorted(e["id"] for e in path)),
                        "hop_count": hops, "entity_ids": visited[:], "nodes": nodes,
                        "edges": path[:], "strategy": selected, "temporal_mode": mode,
                        "reference_date": max(e["end"] for e in path)}
            options = self.adj[node][:]
            self.rng.shuffle(options)
            for neighbor, edge in options:
                if neighbor in visited:
                    continue
                visited.append(neighbor)
                path.append(edge)
                result = walk(neighbor)
                if result:
                    return result
                path.pop()
                visited.pop()
            return None
        return walk(start)


def validate_positive(draft, path):
    if not isinstance(draft, dict):
        raise ValueError("Positive draft must be an object")
    for field in ("entity_ids", "used_edge_ids"):
        if not isinstance(draft.get(field), list) or any(not isinstance(x, str) for x in draft[field]):
            raise ValueError(field + " must be a list of strings")
        if len(set(draft[field])) != len(draft[field]):
            raise ValueError(field + " must have no duplicates")
    if not isinstance(draft.get("claim"), str) or not draft["claim"].strip():
        raise ValueError("Empty claim")
    if set(draft.get("used_edge_ids", [])) != {e["id"] for e in path["edges"]}:
        raise ValueError("Claim must use every path edge exactly as support")
    if set(draft.get("entity_ids", [])) != set(path["entity_ids"]):
        raise ValueError("Claim entity set differs from path")
    if "Date:" in draft["claim"]:
        raise ValueError("Reference Date is appended by the script, not generated")


def audit_accepted(audit, confidence, negative=False):
    if not isinstance(audit, dict):
        return False
    required = ["faithfulness", "temporal_consistency", "fluency", "evidence_sufficiency",
                "label_consistency", "multi_hop_necessary", "strategy_correct"]
    if negative:
        required.append("explicit_contradiction")
    return (audit.get("valid") is True and isinstance(audit.get("confidence"), (int, float))
            and not isinstance(audit["confidence"], bool) and confidence <= audit["confidence"] <= 1
            and all(audit.get(k) is True for k in required))


def parse_quantity(value):
    if not isinstance(value, str):
        raise ValueError("Quantity must be a numeric string, optionally followed by a unit")
    match = re.fullmatch(r"\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*([^\d]*)", value)
    if not match:
        raise ValueError("Invalid or nonfinite quantity")
    return Decimal(match.group(1).replace(",", "")), match.group(2).strip().lower()


def validate_negative(draft, parent, graph):
    if not isinstance(draft, dict) or not isinstance(draft.get("strategy"), str):
        raise ValueError("Negative draft requires a strategy string")
    if not isinstance(draft.get("edit"), dict):
        raise ValueError("Negative edit must be an object")
    if not isinstance(draft.get("entity_ids"), list) or any(not isinstance(x, str) for x in draft["entity_ids"]):
        raise ValueError("Negative entity_ids must be a list of strings")
    if not isinstance(draft.get("contradiction"), dict):
        raise ValueError("Negative contradiction must be an object")
    edit = draft["edit"]
    if not all(k in edit for k in ("edge_id", "field", "before", "after")):
        raise ValueError("Missing structured edit fields")
    if not isinstance(edit["edge_id"], str) or not isinstance(edit["field"], str):
        raise ValueError("Edit edge_id and field must be strings")
    strategy = draft["strategy"]
    if strategy not in PERTURBATIONS:
        raise ValueError("Unknown perturbation")
    edit = draft["edit"]
    if edit["field"] not in PERTURBATIONS[strategy]:
        raise ValueError("Field does not implement requested perturbation")
    original = next((e for e in parent["path"]["edges"] if e["id"] == edit["edge_id"]), None)
    if original is None:
        raise ValueError("Edited edge is not in parent path")
    field = edit["field"]
    expected = ({"start": original["start"], "end": original["end"]} if field == "time" else
                "" if field == "premise" else
                next(n["label"] for n in graph["nodes"] if n["id"] == original["tail"]) if field == "value" else
                original.get(field, "positive"))
    if edit["before"] != expected or edit["before"] == edit["after"]:
        raise ValueError("Edit must start with actual original value and change it")
    if field == "time":
        if not isinstance(edit["after"], dict) or any(not isinstance(edit["after"].get(k), str) for k in ("start", "end")):
            raise ValueError("New time must have ISO start/end strings")
        interval(edit["after"])
    elif not isinstance(edit["after"], str) or not edit["after"].strip():
        raise ValueError("New field value must be a nonempty string")
    if field == "polarity" and {edit["before"], edit["after"]} != {"positive", "negative"}:
        raise ValueError("Polarity must reverse a factual assertion")
    if field == "value":
        before_value, before_unit = parse_quantity(edit["before"])
        after_value, after_unit = parse_quantity(edit["after"])
        if before_value == after_value or before_unit != after_unit:
            raise ValueError("Data perturbation must change number while preserving unit")
    nodes = {n["id"]: n for n in graph["nodes"]}
    expected_entities = set(parent["entity_ids"])
    if field in ("head", "tail"):
        if edit["after"] not in nodes:
            raise ValueError("Replacement entity must have a known graph ID")
        if nodes[edit["before"]]["type"] != nodes[edit["after"]]["type"]:
            raise ValueError("Entity replacement must preserve entity type")
        expected_entities.add(edit["after"])
    if set(draft.get("entity_ids", [])) != expected_entities:
        raise ValueError("All original and introduced entity IDs must be retained for splitting")
    contradiction = draft.get("contradiction", {})
    sources = {s.get("id", s.get("source_id")): s for s in graph["sources"]}
    if not isinstance(contradiction.get("source_id"), str):
        raise ValueError("Contradiction source_id must be a string")
    src = sources.get(contradiction.get("source_id"))
    quote = contradiction.get("quote", "")
    if (not src or not isinstance(quote, str) or not quote.strip() or quote not in src.get("text", "")
            or not isinstance(contradiction.get("reason"), str) or not contradiction["reason"].strip()):
        raise ValueError("Negative requires an exact source quote and contradiction rationale")
    claim = draft.get("claim", "")
    if not isinstance(claim, str) or not claim.strip() or "Date:" in claim:
        raise ValueError("Invalid negative claim")
    if claim.strip() == parent["claim"].split(" Date:")[0].strip():
        raise ValueError("Unchanged claim")
    return expected_entities
