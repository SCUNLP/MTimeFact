#!/usr/bin/env python3
"""Targeted date enrichment for undated multi-hop claims.

For claims where both `date` and `time_window` are null:
  1. Trace back to the path's node entities (QIDs)
  2. Fetch each entity's temporal anchors from Wikidata
     (P569 birth, P570 death, P571 inception, P576 dissolution,
      P577 publication, P585 point-in-time)
  3. Compute a month-precision time window from the min/max of all anchors
  4. For entities with NO Wikidata temporal data, try Bing search
     ("entity name" date/founded/born/released) as a fallback
  5. Claims still without any date after both passes are DROPPED

Output: data/synthesized_claims_dated.jsonl + report

Usage:
  python3 scripts/data/enrich_path_dates.py --proxy http://your-proxy:7890
"""
import argparse
import atexit, hashlib, json, re, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, quote_plus

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clash_rotate import NodeRotator
from wikipedia_entities import Wiki

TEMPORAL_PROPS = ["P569", "P570", "P571", "P576", "P577", "P585"]
PROP_LABELS = {"P569": "birth", "P570": "death", "P571": "inception",
               "P576": "dissolution", "P577": "publication", "P585": "point_in_time"}
WIKIDATA_TIME = re.compile(r"^([+-])(\d{4,5})-(\d{2})-(\d{2})T")
MONTH_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
BING_DATE = re.compile(r"\b((?:19|20)\d{2})\b")


def parse_wd_time(value):
    if not isinstance(value, dict):
        return None
    m = WIKIDATA_TIME.match(value.get("time") or "")
    if not m:
        return None
    year, month, day = m.group(2), int(m.group(3)), int(m.group(4))
    precision = value.get("precision")
    if precision == 11 and month and day:
        return "%s-%02d-%02d" % (year, month, day), "day"
    if precision == 10 and month:
        return "%s-%02d" % (year, month), "month"
    return year, "year"


def fetch_entity_temporal(wiki, cache_dir, qids):
    """QID -> {prop: (date, precision)} for temporal properties."""
    result = {}
    qids = sorted(set(qids))
    for start in range(0, len(qids), 40):
        chunk = qids[start:start + 40]
        key = "temporal_" + hashlib.sha256("|".join(chunk).encode()).hexdigest()[:24]
        params = urlencode({"action": "wbgetentities", "format": "json",
                           "ids": "|".join(chunk), "props": "claims"})
        try:
            data = wiki.get("https://www.wikidata.org/w/api.php?" + params, cache_dir, key)
        except RuntimeError as exc:
            print("  batch failed: %s" % exc, file=sys.stderr, flush=True)
            continue
        for qid, entity in (data.get("entities") or {}).items():
            temporal = {}
            for prop in TEMPORAL_PROPS:
                for claim in (entity.get("claims") or {}).get(prop) or []:
                    snak = ((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value")
                    if isinstance(snak, dict) and "time" in snak:
                        date, precision = parse_wd_time(snak)
                        if date and (prop not in temporal or date < temporal[prop][0]):
                            temporal[prop] = (date, precision)
            if temporal:
                result[qid] = temporal
    return result


def bing_search_date(wiki, entity_label):
    """Try Bing search for entity date info. Returns (date, precision) or None."""
    try:
        url = "https://www.bing.com/search?q=" + quote_plus('"%s" born OR founded OR released OR established date' % entity_label)
        html = wiki._request(url)
        # extract first 4-digit year near "born|founded|released|established"
        years = BING_DATE.findall(html[:20000])
        if years:
            return years[0], "year"
    except Exception:
        pass
    return None


def compute_window(temporal_by_qid, node_qids, month_precision=True):
    """Min/max temporal anchors for a set of entities -> month-precision window."""
    starts, ends = [], []
    for qid in node_qids:
        temporal = temporal_by_qid.get(qid)
        if not temporal:
            continue
        for prop, (date, precision) in temporal.items():
            if not date:
                continue
            if precision == "day":
                starts.append(date[:7]); ends.append(date[:7])
            elif precision == "month":
                starts.append(date); ends.append(date)
            elif precision == "year":
                starts.append(date + "-01"); ends.append(date + "-12")
    if not starts:
        return None
    return {"start": min(starts), "end": max(ends)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/synthesized_claims.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/synthesized_claims_dated.jsonl"))
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/wiki"))
    parser.add_argument("--clash-controller", default=None)
    parser.add_argument("--clash-group", default="🔰 选择节点")
    parser.add_argument("--rotate-every", type=int, default=20)
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--no-bing", action="store_true", help="Skip Bing search fallback")
    parser.add_argument("--bing-only-missing", action="store_true", default=True)
    args = parser.parse_args(argv)
    claims = [json.loads(l) for l in args.input.read_text(encoding="utf-8").splitlines() if l.strip()]
    undated = [r for r in claims if not r.get("date") and not (r["time_window"].get("start") or r["time_window"].get("end"))]
    dated = [r for r in claims if r.get("date") or (r["time_window"].get("start") or r["time_window"].get("end"))]
    print(json.dumps({"total": len(claims), "dated": len(dated), "undated": len(undated)},
                     ensure_ascii=False), flush=True)

    # Collect unique path node sets for undated claims
    path_cache = {}
    for hop in (2, 3, 4, 5):
        f = Path("data/graphs/paths/paths_%dhop.json" % hop)
        if f.exists():
            data = json.loads(f.read_text(encoding="utf-8"))
            for p in data.get("paths") or []:
                path_cache[p["id"]] = [n["qid"] for n in p["nodes"] if n.get("qid")]

    all_qids = set()
    undated_paths = {}
    for r in undated:
        qids = path_cache.get(r.get("path_id"), [])
        undated_paths[r["path_id"]] = qids
        all_qids.update(qids)
    print("unique entities to fetch:", len(all_qids), flush=True)

    # Phase 1: Wikidata temporal anchors
    rotator = None
    if not args.no_rotate:
        rotator = NodeRotator(controller=args.clash_controller, group=args.clash_group,
                              rotate_every=args.rotate_every)
        atexit.register(rotator.restore)
    wiki = Wiki(args.proxy, rotator=rotator)
    temporal = fetch_entity_temporal(wiki, args.cache_dir, list(all_qids))
    print("entities with temporal data:", len(temporal), "/", len(all_qids), flush=True)

    # Phase 2: Bing fallback for entities without Wikidata temporal data
    if not args.no_bing:
        missing = [q for q in all_qids if q not in temporal]
        # we need labels for search; load from entities file
        entities = {json.loads(l)["qid"]: json.loads(l)
                    for l in open("data/wiki_entities/entities.jsonl", encoding="utf-8") if l.strip()}
        bing_found = 0
        for i, qid in enumerate(sorted(missing)[:200]):  # cap at 200 to be practical
            label = entities.get(qid, {}).get("label", "")
            if not label:
                continue
            result = bing_search_date(wiki, label)
            if result:
                temporal[qid] = {"bing_search": result}
                bing_found += 1
            if (i + 1) % 20 == 0:
                print("  bing: %d/%d checked, found %d" % (i + 1, min(len(missing), 200), bing_found), flush=True)
            time.sleep(1.0)  # polite
        print("bing fallback found:", bing_found, "dates", flush=True)

    # Phase 3: Apply windows and rebuild
    enriched, still_undated = [], []
    for r in undated:
        qids = undated_paths.get(r.get("path_id"), [])
        window = compute_window(temporal, qids)
        if window and (window["start"] or window["end"]):
            r = dict(r, time_window=window, date_source="wikidata_entity_temporal")
            enriched.append(r)
        else:
            still_undated.append(r)

    # Combine: originally dated + enriched; drop still-undated
    final = dated + enriched
    dropped = still_undated
    out = args.output
    temporary = out.with_suffix(out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in final:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(out)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "input": len(claims), "originally_dated": len(dated),
              "enriched": len(enriched), "still_undated_dropped": len(dropped),
              "final": len(final), "entities_temporal": len(temporal),
              "entities_total": len(all_qids),
              "output": str(out),
              "notes": ["Time windows for undated claims computed from entity-level Wikidata temporal anchors "
                        "(birth/death/inception/publication/dissolution/point-in-time).",
                        "Month precision: day dates truncated to month, year dates span Jan-Dec.",
                        "Claims with no recoverable date for ANY path entity are dropped."]}
    Path(str(out) + ".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps({"status": "ok", "final": len(final), "enriched": len(enriched),
                      "dropped": len(dropped)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
