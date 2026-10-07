
"""Entity-disjoint split and honest human-review export (Python standard library)."""

from __future__ import annotations


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HOPS = (2, 3, 4, 5)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                records.append(value)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class UnionFind:
    def __init__(self, size: int):
        self.parents = list(range(size))
        self.sizes = [1] * size

    def find(self, value: int) -> int:
        while self.parents[value] != value:
            self.parents[value] = self.parents[self.parents[value]]
            value = self.parents[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.sizes[left] < self.sizes[right]:
            left, right = right, left
        self.parents[right] = left
        self.sizes[left] += self.sizes[right]


def build_components(records: list[dict[str, Any]]) -> tuple[list[list[int]], list[str]]:
    """Every shared entity and every positive/negative family stays together.

    Entity IDs are authoritative; callers must include replacements and bridge
    entities. No entity is removed to make the requested split easier.
    """
    ids: dict[str, int] = {}
    for index, record in enumerate(records):
        if record.get("status") == "draft_only_requires_gpt5" or record.get("demo") is True:
            raise ValueError("Draft/legacy template demo records are not final GPT-5 claims and cannot be split as a dataset")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"Record {index + 1}: missing nonempty string id")
        if record_id in ids:
            raise ValueError(f"Duplicate sample id: {record_id}")
        ids[record_id] = index
        entities = record.get("entity_ids")
        if not isinstance(entities, list) or not entities or any(
            not isinstance(entity, str) or not entity.strip() for entity in entities
        ):
            raise ValueError(f"{record_id}: entity_ids must be a nonempty list of strings")
        if type(record.get("hop_count")) is not int or record["hop_count"] not in HOPS:
            raise ValueError(f"{record_id}: hop_count must be an integer in {HOPS}")
        if record.get("parent_id") is not None and not isinstance(record["parent_id"], str):
            raise ValueError(f"{record_id}: parent_id must be a string or null")
    union = UnionFind(len(records))
    entity_owner: dict[str, int] = {}
    family_owner: dict[str, int] = {}
    missing_parents: set[str] = set()
    for index, record in enumerate(records):
        for entity in set(record["entity_ids"]):
            if entity in entity_owner:
                union.union(index, entity_owner[entity])
            else:
                entity_owner[entity] = index
        
        
        family_tokens = [record["id"]]
        if record.get("parent_id"):
            family_tokens.append(record["parent_id"])
            if record["parent_id"] not in ids:
                missing_parents.add(record["parent_id"])
        for family in family_tokens:
            if family in family_owner:
                union.union(index, family_owner[family])
            else:
                family_owner[family] = index
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        grouped[union.find(index)].append(index)
    return list(grouped.values()), sorted(missing_parents)


class SearchPath:
    __slots__ = ("index", "previous")

    def __init__(self, index: int, previous: SearchPath | None):
        self.index = index
        self.previous = previous


def _score(counts: tuple[int, ...]) -> tuple[int, int]:
    
    return sum(counts), -sum(value * value for value in counts)


def choose_test_components(
    vectors: list[tuple[int, ...]],
    quota: int,
    seed: int,
    max_states: int = 20_000,
    max_transitions: int = 500_000,
) -> tuple[set[int], dict[str, Any]]:
    """Bounded multidimensional subset sum, with seeded greedy warm starts.

    A failed bounded search is not proof of impossibility. Components larger
    than any corresponding quota cannot enter test without violating the cap.
    """
    target = (quota,) * len(HOPS)
    zero = (0,) * len(HOPS)
    eligible = [index for index, vector in enumerate(vectors) if all(v <= quota for v in vector)]
    rng = random.Random(seed)
    best_counts, best_selected = zero, set()
    metadata: dict[str, Any] = {
        "algorithm": "seeded greedy starts + bounded sparse multidimensional subset sum",
        "max_states": max_states,
        "max_transitions": max_transitions,
        "transitions": 0,
        "states_pruned": False,
        "transition_budget_exhausted": False,
        "eligible_components": len(eligible),
    }
    if quota == 0:
        metadata["exact"] = True
        return set(), metadata
    for attempt in range(32):
        order = eligible[:]
        rng.shuffle(order)
        if attempt == 0:
            order.sort(key=lambda index: sum(vectors[index]), reverse=True)
        elif attempt == 1:
            order.sort(key=lambda index: sum(vectors[index]))
        counts, selected = zero, set()
        for index in order:
            candidate = tuple(left + right for left, right in zip(counts, vectors[index]))
            if all(value <= quota for value in candidate):
                counts = candidate
                selected.add(index)
        if _score(counts) > _score(best_counts):
            best_counts, best_selected = counts, selected
        if counts == target:
            metadata.update(exact=True, found_by="greedy")
            return selected, metadata

    order = eligible[:]
    rng.shuffle(order)
    states: dict[tuple[int, ...], SearchPath | None] = {zero: None}
    best_path: SearchPath | None = None
    exact_path: SearchPath | None = None
    for index in order:
        additions: dict[tuple[int, ...], SearchPath] = {}
        for counts, path in list(states.items()):
            if metadata["transitions"] >= max_transitions:
                metadata["transition_budget_exhausted"] = True
                break
            metadata["transitions"] += 1
            candidate = tuple(left + right for left, right in zip(counts, vectors[index]))
            if any(value > quota for value in candidate) or candidate in states or candidate in additions:
                continue
            next_path = SearchPath(index, path)
            additions[candidate] = next_path
            if _score(candidate) > _score(best_counts):
                best_counts, best_path = candidate, next_path
            if candidate == target:
                exact_path = next_path
                break
        states.update(additions)
        if exact_path is not None or metadata["transition_budget_exhausted"]:
            break
        if len(states) > max_states:
            metadata["states_pruned"] = True
            retained = sorted(states, key=_score, reverse=True)[:max_states - 1]
            states = {counts: states[counts] for counts in retained}
            states[zero] = None
    if exact_path is not None:
        best_path = exact_path
    if best_path is not None:
        best_selected = set()
        while best_path is not None:
            best_selected.add(best_path.index)
            best_path = best_path.previous
    metadata["exact"] = best_counts == target
    metadata["found_by"] = "subset_sum" if exact_path is not None else "best_partial_assignment"
    metadata["search_exhaustive"] = not metadata["states_pruned"] and not metadata["transition_budget_exhausted"]
    return best_selected, metadata


def split_records(
    records: list[dict[str, Any]], quota: int, seed: int = 42,
    max_states: int = 20_000, max_transitions: int = 500_000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if quota < 0 or max_states < 2 or max_transitions < 1:
        raise ValueError("quota >= 0, max_states >= 2, and max_transitions >= 1 are required")
    components, missing_parents = build_components(records)
    vectors = [tuple(sum(records[i]["hop_count"] == hop for i in component) for hop in HOPS)
               for component in components]
    selected, search = choose_test_components(vectors, quota, seed, max_states, max_transitions)
    test_indices = {index for component_id in selected for index in components[component_id]}
    train = [record for index, record in enumerate(records) if index not in test_indices]
    test = [record for index, record in enumerate(records) if index in test_indices]
    train_entities = {entity for record in train for entity in record["entity_ids"]}
    test_entities = {entity for record in test for entity in record["entity_ids"]}
    train_families = {token for record in train for token in (record["id"], record.get("parent_id")) if token}
    test_families = {token for record in test for token in (record["id"], record.get("parent_id")) if token}
    entity_overlap = sorted(train_entities & test_entities)
    family_overlap = sorted(train_families & test_families)
    if entity_overlap or family_overlap:
        raise AssertionError("Internal error: entity/family leakage in proposed split")
    test_counts = Counter(record["hop_count"] for record in test)
    train_counts = Counter(record["hop_count"] for record in train)
    eligible_vectors = [vector for vector in vectors if all(count <= quota for count in vector)]
    capacities = {str(hop): sum(vector[position] for vector in eligible_vectors)
                  for position, hop in enumerate(HOPS)}
    deficits = {str(hop): quota - test_counts[hop] for hop in HOPS}
    if not any(deficits.values()):
        status = "exact_quotas_met"
    elif any(capacities[str(hop)] < quota for hop in HOPS):
        status = "quota_shortfall_insufficient_eligible_capacity"
    elif search["search_exhaustive"]:
        status = "quota_shortfall_no_exact_component_assignment"
    else:
        status = "quota_shortfall_no_exact_solution_found_within_search_budget"
    largest = sorted(enumerate(components), key=lambda item: len(item[1]), reverse=True)[:10]
    report = {
        "status": status,
        "seed": seed,
        "quota_unit": "individual claims, including positive and negative claims",
        "requested_test_per_hop": {str(hop): quota for hop in HOPS},
        "train_per_hop": {str(hop): train_counts[hop] for hop in HOPS},
        "test_per_hop": {str(hop): test_counts[hop] for hop in HOPS},
        "test_deficits": deficits,
        "eligible_test_capacity_per_hop": capacities,
        "total_input": len(records), "train_total": len(train), "test_total": len(test),
        "unassigned_total": 0,
        "component_count": len(components),
        "components_too_large_for_test_quota": len(components) - len(eligible_vectors),
        "largest_component_fraction": max((len(c) for c in components), default=0) / max(1, len(records)),
        "giant_component_warning": any(len(c) > len(records) / 2 for c in components) and bool(records),
        "largest_components": [
            {"component_id": index, "size": len(component),
             "counts_per_hop": dict(zip(map(str, HOPS), vectors[index])),
             "example_sample_ids": [records[i]["id"] for i in component[:5]]}
            for index, component in largest
        ],
        "missing_parent_ids": missing_parents,
        "leakage_checks": {
            "entity_id_overlap_count": len(entity_overlap),
            "family_token_overlap_count": len(family_overlap),
            "all_input_assigned_once": len(train) + len(test) == len(records),
            "pass": not entity_overlap and not family_overlap,
            "scope": "Provided entity IDs and parent IDs; upstream entity linking determines semantic coverage.",
        },
        "search": search,
        "notes": [
            "Whole connected components are assigned. Entity IDs, including replacements, are never removed.",
            "A bounded-search shortfall is not proof that an exact split is mathematically impossible.",
            "Quota shortfalls do not weaken entity or positive/negative-family disjointness.",
        ],
    }
    return train, test, report


def export_human_review(
    path: Path, synthetic: list[dict[str, Any]], raw: list[dict[str, Any]],
    test_ids: set[str], size: int, seed: int,
) -> dict[str, Any]:
    """Seeded, stratified sample; annotator cells intentionally remain empty."""
    if size < 0:
        raise ValueError("human sample size must be nonnegative")
    strata: dict[tuple[str, str, str], list[tuple[str, int, dict[str, Any]]]] = defaultdict(list)
    for kind, records in (("synthetic", synthetic), ("raw", raw)):
        for index, record in enumerate(records):
            group = str(record.get("hop_count", record.get("temporal_category", record.get("temporal_type", "unknown"))))
            label = str(record.get("label", record.get("source_verdict", record.get("verdict", "unknown"))))
            strata[(kind, group, label)].append((kind, index, record))
    rng = random.Random(seed)
    for entries in strata.values():
        rng.shuffle(entries)
    keys = sorted(strata)
    rng.shuffle(keys)
    selected = []
    
    
    while len(selected) < size:
        changed = False
        for key in keys:
            if strata[key] and len(selected) < size:
                selected.append(strata[key].pop())
                changed = True
        if not changed:
            break
    rng.shuffle(selected)
    fields = ["sample_id", "data_type", "split", "hop_count", "label", "claim", "reference_date",
              "entity_ids", "source_url", "audit", "annotator_1", "annotator_2", "annotator_3", "notes"]
    counts = Counter()
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for kind, index, record in selected:
            sample_id = record.get("id", f"raw-row-{index + 1}")
            counts[kind] += 1
            writer.writerow({
                "sample_id": sample_id, "data_type": kind,
                "split": ("test" if sample_id in test_ids else "train") if kind == "synthetic" else "raw",
                "hop_count": record.get("hop_count", ""),
                "label": record.get("label", record.get("source_verdict", record.get("verdict", ""))),
                "claim": record.get("claim", record.get("title", record.get("text", ""))),
                "reference_date": record.get("reference_date", record.get("published_at", "")),
                "entity_ids": json.dumps(record.get("entity_ids", []), ensure_ascii=False),
                "source_url": record.get("source_url", record.get("url", "")),
                "audit": json.dumps(record.get("audit", {}), ensure_ascii=False),
                "annotator_1": "", "annotator_2": "", "annotator_3": "", "notes": "",
            })
    return {"requested": size, "exported": len(selected), "raw": counts["raw"],
            "synthetic": counts["synthetic"], "strata": len(strata),
            "method": "seeded balanced round-robin over data type, hop/category, and label strata",
            "annotation_status": "pending; no pass rate or agreement statistic has been computed"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Repeat for positive/negative JSONL files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--test-per-hop", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--human-sample-size", type=int, default=900)
    parser.add_argument("--raw-input", action="append", default=[], help="Optional raw JSONL for human review")
    parser.add_argument("--max-search-states", type=int, default=20_000)
    parser.add_argument("--max-search-transitions", type=int, default=500_000)
    args = parser.parse_args(argv)
    if args.human_sample_size < 0:
        parser.error("--human-sample-size must be nonnegative")
    output_dir = Path(args.output_dir)
    if not args.overwrite and any((output_dir / name).exists() for name in ("train.jsonl", "test.jsonl", "split_report.json", "human_review.csv")):
        parser.error("split output exists; use --overwrite or a new directory")
    records = [record for path in args.input for record in read_jsonl(path)]
    if len({bool(r.get("demo", False)) for r in records}) > 1:
        parser.error("cannot mix demo and real samples")
    raw = [record for path in args.raw_input for record in read_jsonl(path)]
    train, test, report = split_records(records, args.test_per_hop, args.seed,
                                        args.max_search_states, args.max_search_transitions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "test.jsonl", test)
    report["human_review"] = export_human_review(
        output_dir / "human_review.csv", records, raw, {record["id"] for record in test},
        args.human_sample_size, args.seed,
    )
    (output_dir / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "train": len(train), "test": len(test),
                      "deficits": report["test_deficits"], "report": str(output_dir / "split_report.json")},
                     ensure_ascii=False))
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
