#!/usr/bin/env python3
"""Harvest genuinely obscure entities to extend the knowledge graph downward.

Source: entity-type object labels from the extracted triples that never matched
the popularity census (i.e. pages that never entered the monthly top lists).

Pipeline:
  1. resolve labels -> Wikidata QIDs via the en.wikipedia API (batched, cached)
  2. per-article monthly pageviews (12 months) -> max monthly views
  3. keep pages with max views <= --max-views (obscure) and a known category
  4. write entities_obscure.jsonl in the same schema as entities.jsonl
"""
import argparse, atexit, hashlib, json, sys, time
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import quote, urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikipedia_entities import Wiki, resolve_qids, fetch_types, classify, CATEGORY_LABELS  # noqa: E402
from clash_rotate import NodeRotator  # noqa: E402

MONTHS = 12


def last_months(count):
    today = date.today()
    y, m = today.year, today.month - 1
    if m == 0:
        y, m = y - 1, 12
    out = []
    for _ in range(count):
        out.append("%04d-%02d" % (y, m))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return list(reversed(out))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", type=Path, default=Path("data/wiki_triples_enriched.jsonl"))
    ap.add_argument("--entities", type=Path, default=Path("data/wiki_entities/entities.jsonl"))
    ap.add_argument("--census-dir", type=Path, default=Path("data/wiki_entities"))
    ap.add_argument("--output", type=Path, default=Path("data/wiki_entities/entities_obscure.jsonl"))
    ap.add_argument("--report", type=Path, default=Path("data/wiki_entities/entities_obscure.report.json"))
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--cache-dir", type=Path, default=Path(".cache/obscure_harvest"))
    ap.add_argument("--max-label-candidates", type=int, default=12000)
    ap.add_argument("--max-views", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=3000, help="how many obscure entities to keep")
    ap.add_argument("--min-bytes", type=int, default=2000, help="minimum enwiki article length")
    ap.add_argument("--min-per-category", type=int, default=150)
    ap.add_argument("--clash-controller", default=None)
    ap.add_argument("--clash-group", default="🔰 选择节点")
    ap.add_argument("--clash-nodes", default=None)
    ap.add_argument("--rotate-every", type=int, default=200)
    ap.add_argument("--no-rotate", action="store_true")
    args = ap.parse_args(argv)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    rotator = None
    if not args.no_rotate:
        nodes = [n.strip() for n in args.clash_nodes.split(",")] if args.clash_nodes else None
        rotator = NodeRotator(controller=args.clash_controller, group=args.clash_group,
                              nodes=nodes, rotate_every=args.rotate_every)
        atexit.register(rotator.restore)
    wiki = Wiki(proxy=args.proxy, rotator=rotator)

    # existing census (popular entities to exclude)
    known_titles, known_qids = set(), set()
    for name in ("census_en.jsonl", "supplement.jsonl", "entities.jsonl"):
        p = args.census_dir / name
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    known_titles.add(e["title"]); known_qids.add(e["qid"])

    # candidate labels: entity-type objects not in the census
    mentions = Counter()
    for line in args.triples.open(encoding="utf-8"):
        if not line.strip():
            continue
        t = json.loads(line)
        if t.get("object_type") == "entity":
            label = (t.get("object") or "").strip()
            if label and label not in known_titles and 1 < len(label) < 80:
                mentions[label] += 1
    labels = [lb for lb, _ in mentions.most_common(args.max_label_candidates)]
    print(json.dumps({"candidate_labels": len(mentions), "using": len(labels)}), flush=True)

    resolved = resolve_qids(wiki, args.cache_dir, labels)
    resolved = {t: q for t, q in resolved.items() if q not in known_qids}
    print(json.dumps({"resolved_qids": len(resolved)}), flush=True)

    months = last_months(MONTHS)
    start, end = months[0], months[-1]
    # pageviews REST API is rate-limited from this network; use Wikidata sitelinks
    # (fewer sitelinks = more obscure) plus article length as obscurity signals.
    from urllib.parse import urlencode as _urlencode

    def fetch_sitelinks_and_types(qids):
        out = {}
        for i in range(0, len(qids), 50):
            chunk = qids[i:i + 50]
            params = _urlencode({"action": "wbgetentities", "format": "json",
                                 "ids": "|".join(chunk),
                                 "props": "labels|claims|sitelinks", "languages": "en|zh"})
            key = "wd_%s" % hashlib.sha256("|".join(chunk).encode()).hexdigest()[:24]
            data = wiki.get("https://www.wikidata.org/w/api.php?" + params, args.cache_dir, key)
            for qid, ent in (data.get("entities") or {}).items():
                if not isinstance(ent, dict) or ent.get("missing"):
                    continue
                labels = ent.get("labels") or {}
                en = (labels.get("en") or {}).get("value", "")
                zh = (labels.get("zh") or {}).get("value", "")
                p31 = []
                for claim in (ent.get("claims") or {}).get("P31", []):
                    v = ((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value")
                    if isinstance(v, dict) and v.get("id"):
                        p31.append(v["id"])
                out[qid] = {"label": en or zh, "label_zh": zh, "p31": p31,
                            "sitelinks": len(ent.get("sitelinks") or {})}
            if i % 2500 == 0:
                print("wikidata %d/%d" % (i + 50, len(qids)), flush=True)
        return out

    obscure_qids = [q for t, q in sorted(resolved.items())]
    wd = fetch_sitelinks_and_types(obscure_qids)
    print(json.dumps({"wikidata_fetched": len(wd)}), flush=True)

    # article length + disambiguation flag (action API; not rate-limited)
    info = {}
    titles_sorted = sorted(resolved.items())
    for i in range(0, len(titles_sorted), 50):
        chunk = [t for t, _ in titles_sorted[i:i + 50]]
        params = _urlencode({"action": "query", "format": "json", "formatversion": "2",
                             "prop": "info|pageprops", "ppprop": "disambiguation",
                             "titles": "|".join(chunk)})
        key = "len_%s" % hashlib.sha256("|".join(chunk).encode()).hexdigest()[:24]
        try:
            data = wiki.get("https://en.wikipedia.org/w/api.php?" + params, args.cache_dir, key)
        except RuntimeError:
            continue
        for page in (data.get("query", {}).get("pages") or []):
            if page.get("missing"):
                continue
            info[page["title"]] = {"bytes": page.get("length") or 0,
                                   "disambig": bool(page.get("pageprops", {}).get("disambiguation"))}
        if i % 2000 == 0:
            print("pagelen %d/%d" % (i + 50, len(titles_sorted)), flush=True)

    views = {}

    # category + obscurity (sitelinks as proxy), drop unknown types / disambiguations
    records = []
    for title, qid in sorted(resolved.items()):
        meta = wd.get(qid)
        if not meta:
            continue
        cat = classify(meta["p31"])
        if not cat:
            continue
        page = info.get(title) or {}
        if page.get("disambig") or page.get("bytes", 0) < args.min_bytes:
            continue
        records.append({"qid": qid, "title": title, "label": meta["label"] or title,
                        "label_zh": meta["label_zh"], "category": cat,
                        "category_label": CATEGORY_LABELS.get(cat, cat),
                        "types_p31": meta["p31"], "monthly_views_max": 0,
                        "sitelinks": meta["sitelinks"], "article_bytes": page.get("bytes", 0),
                        "mentions": mentions[title],
                        "obscurity_note": "never in monthly top-view lists; sitelinks=%d" % meta["sitelinks"],
                        "wiki_url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
                        "wiki_lang": "en"})

    # quota by category, ranking: fewest sitelinks first (most obscure), then most mentions
    by_cat = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda r: (r["sitelinks"], -r["mentions"]))
    picked, seen = [], set()
    for cat, rows in sorted(by_cat.items()):
        for r in rows[:args.min_per_category]:
            picked.append(r); seen.add(r["qid"])
    rest = sorted((r for r in records if r["qid"] not in seen),
                  key=lambda r: (r["sitelinks"], -r["mentions"]))
    for r in rest:
        if len(picked) >= args.limit:
            break
        picked.append(r)
    picked = picked[:args.limit]

    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked),
                           encoding="utf-8")
    report = {
        "candidate_labels": len(mentions), "labels_used": len(labels),
        "resolved_qids": len(resolved), "wikidata_fetched": len(wd),
        "with_known_category": len(records),
        "picked": len(picked),
        "picked_by_category": dict(Counter(r["category"] for r in picked)),
        "picked_sitelinks_pct": _pct(sorted(r["sitelinks"] for r in picked)),
        "picked_article_bytes_pct": _pct(sorted(r["article_bytes"] for r in picked)),
        "note": "monthly_views_max=0: pageviews REST API rate-limited from this network; "
                "obscurity proxied by Wikidata sitelink count (never in top-view lists by construction).",
        "output": str(args.output),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)
    return 0


def _pct(vals):
    if not vals:
        return {}
    return {p: vals[min(len(vals) - 1, int(len(vals) * p))] for p in (0.1, 0.5, 0.9)}


if __name__ == "__main__":
    raise SystemExit(main())
