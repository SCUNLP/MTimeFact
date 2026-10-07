#!/usr/bin/env python3
"""Restore entity-anchor time windows that refresh_windows.py wiped.

Those windows came from Wikidata temporal anchors (P569/P570/P571/P576/P577/P585)
of the path's node entities and are marked date_source=wikidata_entity_temporal.
The Wikidata responses are cached on disk, so the windows can be rebuilt offline
with exactly the original logic.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enrich_path_dates import TEMPORAL_PROPS, parse_wd_time, compute_window  # noqa: E402


def build_temporal_map(cache_dir):
    by_qid = {}
    for f in sorted(Path(cache_dir).glob("temporal_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
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
                by_qid[qid] = temporal
    return by_qid


def path_nodes_by_id(paths_dir):
    out = {}
    for f in sorted(Path(paths_dir).glob("paths_*hop.json")):
        payload = json.loads(f.read_text(encoding="utf-8"))
        for p in payload.get("paths") or []:
            out[p["id"]] = [n["qid"] for n in p.get("nodes") or [] if n.get("qid")]
    return out


def main():
    temporal = build_temporal_map("data/cache/wiki")
    print("temporal qids from cache:", len(temporal), flush=True)
    node_map = path_nodes_by_id("data/graphs/paths")
    print("paths in library:", len(node_map), flush=True)

    for path in ("data/synthesized_claims_final_en.jsonl", "data/synthesized_claims_final.jsonl"):
        p = Path(path)
        rows, restored, still_null, no_path = [], 0, 0, 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tw = r.get("time_window") or {}
            if not (tw.get("start") or tw.get("end")):
                qids = node_map.get(r.get("path_id"))
                if not qids:
                    no_path += 1
                else:
                    window = compute_window(temporal, qids)
                    if window:
                        r["time_window"] = window
                        r["date_source"] = "wikidata_entity_temporal"
                        restored += 1
                if not (r.get("time_window") or {}).get("start"):
                    still_null += 1
            rows.append(r)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as h:
            for r in rows:
                h.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(p)
        print(json.dumps({"file": str(p), "restored": restored,
                          "still_null_window": still_null, "path_id_missing": no_path}), flush=True)


if __name__ == "__main__":
    main()
