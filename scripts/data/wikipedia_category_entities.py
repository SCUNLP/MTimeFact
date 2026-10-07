#!/usr/bin/env python3
"""Supplement the pageviews-ranked entity pool with Wikipedia category members.

The 48-month pageviews union yields ~12k candidate articles (≈6-7k classified
entities); categories fill the gap to the 10k entity goal.  Each source is
specified as ``name=type:quota[:sub]``:

  --source '在世人物=person:3000'          direct page members of Category:在世人物
  --source '各國城市=place:1200:sub'       page members of every subcategory

For each harvested page title the script resolves the Wikidata QID, fetches the
zh label and P31 types, and verifies (or, when P31 is missing, assumes) the
declared category.  Output records match wikipedia_entities.py's census schema
with monthly_views_max=0, so popularity-ranked core entities always sort first.

Usage:
  python3 scripts/data/wikipedia_category_entities.py \
      --proxy http://your-proxy:7890 \
      --source '在世人物=person:3000' \
      --source '各國城市=place:1200:sub' \
      --output data/wiki_entities/supplement.jsonl
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clash_rotate import NodeRotator  # noqa: E402
from wikipedia_entities import (Wiki, CATEGORY_MAP, CATEGORY_LABELS, resolve_qids,  # noqa: E402
                                fetch_types)


def category_members(wiki, cache_dir, title, cmtype="page", limit=5000, api_host="en.wikipedia.org"):
    """Yield member titles (with continuation)."""
    collected, cont = [], None
    while len(collected) < limit:
        params = {"action": "query", "format": "json", "formatversion": "2",
                  "list": "categorymembers", "cmtitle": title,
                  "cmlimit": "500", "cmtype": cmtype}
        if cont:
            params["cmcontinue"] = cont
        key = "cat_%s_%s_%s_%s" % (api_host.split(".")[0], cmtype,
                                   hashlib.sha256(title.encode()).hexdigest()[:12],
                                   hashlib.sha256(str(cont).encode()).hexdigest()[:12])
        data = wiki.get("https://%s/w/api.php?%s" % (api_host, urlencode(params)), cache_dir, key)
        collected.extend(m["title"] for m in data.get("query", {}).get("categorymembers", [])
                         if m.get("title"))
        cont = (data.get("continue") or {}).get("cmcontinue")
        if not cont:
            break
    return collected[:limit]


def is_leaf_category(title):
    """Keep country-level leaf categories ('Cities in X' / 'Companies of X'),
    drop recursive containers ('... by country', '... by year', lists)."""
    if " by " in title:
        return False
    if not (" in " in title or " of " in title):
        return False
    return True


def classify_assumed(p31_ids, assumed):
    for category, wanted in CATEGORY_MAP.items():
        if any(qid in wanted for qid in p31_ids):
            return category
    return assumed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--wiki-lang", default="en", help="Wikipedia language edition (en default)")
    parser.add_argument("--output", type=Path, default=Path("data/wiki_entities/supplement.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/wiki"))
    parser.add_argument("--source", action="append", required=True,
                        help="name=type:quota[:sub] — e.g. '在世人物=person:3000' or '各國城市=place:1200:sub'")
    parser.add_argument("--subcats", type=int, default=45,
                        help="Max leaf subcategories consumed per 'sub' source")
    parser.add_argument("--clash-controller", default=None)
    parser.add_argument("--clash-group", default="🔰 选择节点")
    parser.add_argument("--rotate-every", type=int, default=40)
    parser.add_argument("--no-rotate", action="store_true")
    args = parser.parse_args(argv)
    sources = []
    for spec in args.source:
        name_part, rest = spec.split("=", 1)
        parts = rest.split(":")
        source_type, quota = parts[0], int(parts[1])
        mode = parts[2] if len(parts) > 2 else None
        sources.append({"category": "Category:" + name_part, "type": source_type,
                        "quota": quota, "sub": mode == "sub"})
    from wikipedia_entities import resolve_qids, fetch_types  # shared helpers
    rotator = None
    if not args.no_rotate:
        rotator = NodeRotator(controller=args.clash_controller, group=args.clash_group,
                              rotate_every=args.rotate_every)
        atexit.register(rotator.restore)
    wiki = Wiki(args.proxy, rotator=rotator)
    api_host = "%s.wikipedia.org" % args.wiki_lang
    titles, by_type = [], {}
    for source in sources:
        category = source["category"]
        try:
            if source["sub"]:
                subcats = category_members(wiki, args.cache_dir, category, cmtype="subcat",
                                           limit=500, api_host=api_host)
                subcats = [s for s in subcats if is_leaf_category(s)][:args.subcats]
                per_sub = max(30, source["quota"] // max(1, len(subcats)))
                members = []
                for sub in subcats:
                    members.extend(category_members(wiki, args.cache_dir, sub, limit=per_sub,
                                                    api_host=api_host))
            else:
                members = category_members(wiki, args.cache_dir, category,
                                           limit=source["quota"] * 2, api_host=api_host)
            print("%s -> %d member titles" % (category, len(members)), flush=True)
        except RuntimeError as exc:
            print("%s failed: %s" % (category, exc), file=sys.stderr, flush=True)
            continue
        for title in members:
            by_type[title] = source["type"]
            titles.append(title)
    qid_by_title = resolve_qids(wiki, args.cache_dir, list(dict.fromkeys(titles)), api_host=api_host)
    print("resolved QIDs: %d" % len(qid_by_title), flush=True)
    types = fetch_types(wiki, args.cache_dir, sorted(set(qid_by_title.values())))
    quota_by_type = {}
    for source in sources:
        quota_by_type[source["type"]] = quota_by_type.get(source["type"], 0) + source["quota"]
    entities, counts, seen = [], {}, set()
    for title, assumed in by_type.items():
        qid = qid_by_title.get(title)
        if not qid or qid in seen or qid not in types:
            continue
        label, p31, label_zh = types[qid]
        category = classify_assumed(p31, assumed)
        if counts.get(category, 0) >= quota_by_type.get(category, 0):
            continue
        seen.add(qid)
        counts[category] = counts.get(category, 0) + 1
        entities.append({"qid": qid, "title": title, "label": label, "label_zh": label_zh or None,
                         "category": category, "category_label": CATEGORY_LABELS.get(category, category),
                         "types_p31": p31, "monthly_views_max": 0,
                         "wiki_url": "https://%s/wiki/%s" % (api_host, title.replace(" ", "_")),
                         "wiki_lang": args.wiki_lang, "source_kind": "category_members"})
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for entity in entities:
            handle.write(json.dumps(entity, ensure_ascii=False) + "\n")
    temporary.replace(out)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "sources": sources,
              "member_titles": len(titles), "resolved": len(qid_by_title),
              "supplement_entities": len(entities), "by_category": counts,
              "wiki_stats": {k: v for k, v in wiki.stats.items()},
              "rotations": rotator.rotations if rotator else 0, "output": str(out)}
    Path(str(out) + ".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps({"status": "ok", **{k: report[k] for k in
                      ("member_titles", "resolved", "supplement_entities", "by_category")}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
