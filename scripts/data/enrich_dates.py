#!/usr/bin/env python3
"""Enrich triple dates from Wikidata structured claims (month precision focus).

Targets triples whose dates are year-only or missing and whose relation maps
to a Wikidata property on the head entity:

  born_* -> P569, died_* -> P570, founded -> P571, released* -> P577,
  won -> P166, nominated_for -> P1411, member_of -> P463, educated_at -> P69,
  married_to -> P26, leader_of/served_as/position_held -> P39,
  worked_for -> P108, located_in -> P131, occurred_in -> P276,
  sibling_of -> P3373, child/parent_of -> P40

For creation relations (acted_in / directed / wrote / stars ...) the WORK's
P577 (publication date) is used as the relation's date.

Wikidata time values carry precision (year/month/day); qualifiers P580/P582
give statement time ranges.  A date is applied only when the improvement is
strict (day > month > year > none) and the claim matches the triple tail
(by QID, or is an unambiguous single claim).

Outputs:
  data/wiki_triples_enriched.jsonl  — full triples with upgraded dates
  data/date_enrichment.jsonl        — per-fact upgrades with provenance
  data/wiki_triples_enriched.report.json

Usage:
  python3 scripts/data/enrich_dates.py --proxy http://your-proxy:7890
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clash_rotate import NodeRotator  # noqa: E402
from wikipedia_entities import Wiki  # noqa: E402

PROPERTY_MAP = {
    "born_on": "P569", "born_year": "P569", "born_in": "P569",
    "died_on": "P570", "died_year": "P570",
    "founded": "P571", "founded_by": "P571", "established": "P571",
    "released": "P577", "released_on": "P577", "streaming_debut_on": "P577",
    "won": "P166", "nominated_for": "P1411",
    "member_of": "P463", "educated_at": "P69", "married_to": "P26",
    "leader_of": "P39", "served_as": "P39", "position_held": "P39",
    "worked_for": "P108", "occurred_in": "P276", "located_in": "P131",
    "sibling_of": "P3373", "child_of": "P40", "parent_of": "P40",
}
SINGLETON_PROPERTIES = {"P569", "P570", "P571", "P577", "P1411"}
TAIL_P577_RELATIONS = {"acted_in", "stars", "starred_in", "directed", "directed_by",
                       "wrote", "written_by", "cast_includes", "has_cast_member",
                       "features_cast", "appeared_in"}
PRECISION_RANK = {"none": 0, "year": 1, "month": 2, "day": 3}
WIKIDATA_TIME = re.compile(r"^([+-])(\d{4,5})-(\d{2})-(\d{2})T")


def parse_wikidata_time(value):
    """Wikidata time +1929-05-12T00:00:00Z + precision -> (iso, precision)."""
    if not isinstance(value, dict):
        return None, "none"
    match = WIKIDATA_TIME.match(value.get("time") or "")
    if not match:
        return None, "none"
    year, month, day = match.group(2), int(match.group(3)), int(match.group(4))
    precision = value.get("precision")
    if precision == 11 and month and day:
        return "%s-%02d-%02d" % (year, month, day), "day"
    if precision == 10 and month:
        return "%s-%02d" % (year, month), "month"
    if precision == 9:
        return year, "year"
    if month and day:
        return "%s-%02d-%02d" % (year, month, day), "day"
    if month:
        return "%s-%02d" % (year, month), "month"
    return year, "year"


def statement_date(claim):
    """(date, precision, tail_qid) for a claim; qualifiers P580/P582 preferred."""
    mainsnak = (claim.get("mainsnak") or {}).get("datavalue") or {}
    if mainsnak.get("type") != "wikibase-entityid":
        return None, "none", None
    tail_qid = (mainsnak.get("value") or {}).get("id")
    for key in ("P580", "P582"):
        for qualifier in (claim.get("qualifiers") or {}).get(key) or []:
            date, precision = parse_wikidata_time((qualifier.get("datavalue") or {}).get("value"))
            if date and precision != "none":
                return date, precision, tail_qid
    snak = ((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value")
    if isinstance(snak, dict) and "time" in snak:
        date, precision = parse_wikidata_time(snak)
        return date, precision, tail_qid
    return None, "none", tail_qid


def property_date(claims, property_id):
    """Earliest mainsnak time for singleton properties (P569/P570/P571/P577)."""
    best = None
    for claim in claims.get(property_id) or []:
        snak = ((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value")
        if not isinstance(snak, dict) or "time" not in snak:
            continue
        date, precision = parse_wikidata_time(snak)
        if date and (best is None or date < best[0]):
            best = (date, precision)
    return best or (None, "none")


def fetch_claims(wiki, cache_dir, qids):
    """QID -> claims dict, batched via wbgetentities (cached on disk)."""
    result = {}
    for start in range(0, len(qids), 40):
        chunk = sorted(set(qids))[start:start + 40]
        key = "claims_" + hashlib.sha256("|".join(chunk).encode()).hexdigest()[:24]
        params = urlencode({"action": "wbgetentities", "format": "json",
                           "ids": "|".join(chunk), "props": "claims"})
        data = wiki.get("https://www.wikidata.org/w/api.php?" + params, cache_dir, key)
        for qid, entity in (data.get("entities") or {}).items():
            result[qid] = entity.get("claims") or {}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--triples", type=Path, default=Path("data/wiki_triples.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/wiki_triples_enriched.jsonl"))
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/wiki"))
    parser.add_argument("--clash-controller", default=None)
    parser.add_argument("--clash-group", default="🔰 选择节点")
    parser.add_argument("--rotate-every", type=int, default=25)
    parser.add_argument("--no-rotate", action="store_true")
    args = parser.parse_args(argv)
    triples = [json.loads(l) for l in args.triples.read_text(encoding="utf-8").splitlines() if l.strip()]
    head_targets = [t["head"] for t in triples
                    if t["relation"] in PROPERTY_MAP
                    and PRECISION_RANK.get(t.get("date_precision") or "none", 0) < 3]
    tail_targets = [t["tail"] for t in triples
                    if t["relation"] in TAIL_P577_RELATIONS and t.get("tail")
                    and PRECISION_RANK.get(t.get("date_precision") or "none", 0) < 3]
    print(json.dumps({"triples": len(triples), "head_entities": len(set(head_targets)),
                      "tail_entities": len(set(tail_targets))}, ensure_ascii=False), flush=True)
    rotator = None
    if not args.no_rotate:
        rotator = NodeRotator(controller=args.clash_controller, group=args.clash_group,
                              rotate_every=args.rotate_every)
        atexit.register(rotator.restore)
    wiki = Wiki(args.proxy, rotator=rotator)
    claims = fetch_claims(wiki, args.cache_dir, head_targets + tail_targets)
    print("claims fetched:", len(claims), flush=True)

    output_triples, enriched, upgrades = [], [], Counter()
    for triple in triples:
        old_precision = triple.get("date_precision") or "none"
        old_rank = PRECISION_RANK.get(old_precision, 0)
        new_date, new_precision, source = None, "none", None
        relation = triple["relation"]
        if relation in PROPERTY_MAP and triple["head"] in claims:
            property_id = PROPERTY_MAP[relation]
            if property_id in SINGLETON_PROPERTIES:
                new_date, new_precision = property_date(claims[triple["head"]], property_id)
                source = "wikidata:" + property_id
            else:
                candidates = claims[triple["head"]].get(property_id) or []
                matched = [statement_date(c) for c in candidates]
                if triple.get("tail"):
                    matched = [m for m in matched if m[2] == triple["tail"]]
                elif len(candidates) == 1:
                    matched = matched[:1]
                usable = [m for m in matched if m[0] and m[1] != "none"]
                if len(usable) == 1:
                    new_date, new_precision = usable[0][0], usable[0][1]
                    source = "wikidata:" + property_id
        elif relation in TAIL_P577_RELATIONS and triple.get("tail") and triple["tail"] in claims:
            new_date, new_precision = property_date(claims[triple["tail"]], "P577")
            source = "wikidata:P577"
        if new_date and PRECISION_RANK.get(new_precision, 0) > old_rank:
            upgrades["%s->%s" % (old_precision, new_precision)] += 1
            output_triples.append(dict(triple, date=new_date, date_precision=new_precision,
                                       date_source=source))
            enriched.append({"fact_id": triple.get("fact_id"), "relation": relation,
                             "old_precision": old_precision, "new_date": new_date,
                             "new_precision": new_precision, "source": source})
        else:
            output_triples.append(triple)

    out = args.output
    temporary = out.with_suffix(out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for triple in output_triples:
            handle.write(json.dumps(triple, ensure_ascii=False) + "\n")
    temporary.replace(out)
    Path("data/date_enrichment.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in enriched), encoding="utf-8")
    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "triples": len(triples), "enriched": len(enriched),
              "upgrade_paths": dict(upgrades.most_common(20)),
              "precisions_before": dict(Counter((t.get("date_precision") or "none") for t in triples)),
              "precisions_after": dict(Counter((t.get("date_precision") or "none") for t in output_triples)),
              "claims_fetched": len(claims), "output": str(out),
              "notes": ["Dates come from Wikidata structured claims (QIDs already known); "
                        "qualifiers P580/P582 provide statement ranges. Upgrades are strict "
                        "(day > month > year > none) and tail-matched.",
                        "data/date_enrichment.jsonl carries per-fact provenance."]}
    Path(str(out) + ".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps({"status": "ok", "enriched": len(enriched),
                      "upgrade_paths": report["upgrade_paths"],
                      "precisions_before": report["precisions_before"],
                      "precisions_after": report["precisions_after"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
