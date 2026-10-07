#!/usr/bin/env python3
"""Recompute time_window from evidence_chain dates (the deterministic rule the
synthesis pipeline documents) and overwrite stale windows.

The main set's evidence dates were enriched after synthesis (undated edges
recovered via Wikidata), so 29.9% of stored windows no longer reflect their
evidence chains.  This refreshes both the Chinese and English files so they
stay field-identical.
"""
import json, re, shutil, sys
from pathlib import Path


def window_from_chain(chain):
    starts, ends = [], []
    for e in chain or []:
        v = e.get("date")
        if not isinstance(v, str):
            continue
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            starts.append(v[:7]); ends.append(v[:7])
        elif re.fullmatch(r"\d{4}-\d{2}", v):
            starts.append(v); ends.append(v)
        elif re.fullmatch(r"\d{4}", v):
            starts.append(v + "-01"); ends.append(v + "-12")
    if not starts:
        return {"start": None, "end": None}
    return {"start": min(starts), "end": max(ends)}


def refresh(path):
    p = Path(path)
    rows, changed, widened, narrowed = [], 0, 0, 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        new = window_from_chain(r.get("evidence_chain"))
        old = r.get("time_window") or {}
        if {"start": old.get("start"), "end": old.get("end")} != new:
            changed += 1
            if new["start"] and old.get("start"):
                if (new["start"], new.get("end")) < (old["start"], old.get("end")):
                    widened += 1
                else:
                    narrowed += 1
            r["time_window"] = new
        rows.append(r)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)
    print(json.dumps({"file": str(p), "records": len(rows), "windows_refreshed": changed,
                      "widened": widened, "narrowed": narrowed}), flush=True)


if __name__ == "__main__":
    for f in sys.argv[1:] or [
            "data/synthesized_claims_final.jsonl",
            "data/synthesized_claims_final_en.jsonl",
            "data/synthesized_hard.jsonl",
            "data/synthesized_hard_en.jsonl",
            "data/synthesized_obscure.jsonl",
            "data/synthesized_obscure_en.jsonl"]:
        refresh(f)
