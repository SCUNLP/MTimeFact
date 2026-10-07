#!/usr/bin/env python3
"""Harvest common entities (people, places, organizations, works, events) from
Chinese Wikipedia / Wikidata, ranked by real popularity.

Strategy (WDQS SPARQL was unreachable through the local proxy):
  1. fetch the monthly top-viewed zh.wikipedia articles from the Wikimedia
     pageviews REST API (most viewed ≈ most common entities)
  2. resolve titles to Wikidata items via the zh.wikipedia API (pageprops)
  3. fetch P31 (instance-of) claims via the Wikidata API and classify each
     entity into person / country / place / organization / work / event
  4. keep the top --quota-* entities per category by view rank

All Wikipedia/Wikidata requests go through --proxy; the output feeds
wikipedia_facts.py which extracts 实体/事实/日期 records.

Usage:
  python3 scripts/data/wikipedia_entities.py \
      --proxy http://your-proxy:7890 \
      --output data/wiki_entities/entities.jsonl
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import ProxyHandler, Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clash_rotate import NodeRotator  # noqa: E402

UA = "MTimeFactResearch/0.1 (academic dataset construction; contact: research)"
SKIP_PREFIXES = ("Wikipedia:", "Special:", "Portal:", "Template:", "Category:", "Help:",
                 "File:", "Talk:", "User:", "Draft:", "Module:", "MediaWiki:", "维基百科:",
                 "特别:", "分类:", "模板:", "帮助:", "文件:", "讨论:", "用户:")
SKIP_TITLES = {"Wikipedia:首页", "首页", "Main Page", "Special:Search", "Wikipedia:搜索"}

CATEGORY_MAP = {
    "person": {"Q5"},
    "country": {"Q6256", "Q3624078", "Q7275"},
    "place": {"Q515", "Q1549591", "Q3957", "Q486972", "Q5119", "Q15239622", "Q23442",
              "Q82794", "Q1637706", "Q35657", "Q13221722", "Q10864048"},
    "organization": {"Q4830453", "Q6881511", "Q3918", "Q38723", "Q43229", "Q31855",
                     "Q2385804", "Q1364904", "Q210167", "Q163740", "Q327333", "Q79913"},
    "work": {"Q11424", "Q571", "Q7725634", "Q482994", "Q7366", "Q215380", "Q5398426",
             "Q732577", "Q8261", "Q47461344", "Q15416", "Q24862", "Q580922"},
    "event": {"Q1190554", "Q178561", "Q40231", "Q1656682", "Q3839081", "Q198",
              "Q131569", "Q350604", "Q21480300", "Q16738832", "Q8065", "Q1864448"},
}
CATEGORY_LABELS = {"person": "人物", "country": "国家", "place": "地点",
                   "organization": "机构", "work": "作品", "event": "事件"}


class Wiki:
    def __init__(self, proxy, timeout=45, delay=0.5, retries=4, rotator=None):
        handlers = [ProxyHandler({"http": proxy, "https": proxy})] if proxy else []
        self.opener = build_opener(*handlers)
        self.timeout, self.delay, self.retries = timeout, delay, retries
        self.last = 0.0
        self.rotator = rotator
        self.stats = {"requests": 0, "cache_hits": 0, "rotations_on_error": 0}

    def get(self, url, cache_dir=None, key=None):
        if cache_dir and key:
            cached = Path(cache_dir) / (key + ".json")
            if cached.exists():
                self.stats["cache_hits"] += 1
                return json.loads(cached.read_text(encoding="utf-8"))
        wait = self.delay - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        self.stats["requests"] += 1
        request = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
        last = None
        for attempt in range(self.retries + 1):
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    data = json.load(response)
                if cache_dir and key:
                    Path(cache_dir).mkdir(parents=True, exist_ok=True)
                    (Path(cache_dir) / (key + ".json")).write_text(
                        json.dumps(data, ensure_ascii=False), encoding="utf-8")
                if self.rotator:
                    self.rotator.maybe_rotate()
                return data
            except HTTPError as exc:
                last = exc
                if exc.code in (429, 403):
                    # rate limited or datacenter-IP throttled: switch exit IP
                    rotated = False
                    if self.rotator:
                        self.rotator.rotate()
                        self.stats["rotations_on_error"] += 1
                        rotated = True
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        # after an IP switch the old IP's penalty window does not
                        # apply to the fresh exit, so pause only briefly
                        pause = (min(float(retry_after), 60) if (retry_after and not rotated)
                                 else (8 if rotated else 30))
                    except (TypeError, ValueError):
                        pause = 8 if rotated else 30
                    print("HTTP %d; rotating exit IP, sleeping %.0fs" % (exc.code, pause),
                          file=sys.stderr, flush=True)
                    time.sleep(pause)
                else:
                    time.sleep(min(2 ** attempt, 10))
            except (URLError, TimeoutError, OSError, ValueError) as exc:
                last = exc
                # dead proxy node or connection reset: switch exit IP and retry
                if self.rotator and attempt < self.retries:
                    self.rotator.rotate()
                    self.stats["rotations_on_error"] += 1
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError("wiki request failed: %s" % last)


def latest_months(count):
    today = datetime.now(timezone.utc).date()
    year, month = today.year, today.month
    months = []
    for _ in range(count):
        month -= 1
        if month == 0:
            year, month = year - 1, 12
        months.append("%04d/%02d" % (year, month))
    return months


def fetch_pageviews(wiki, cache_dir, months, limit, project="en.wikipedia"):
    ranked = {}
    slug = project.replace(".", "-")
    for month in months:
        url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/top/%s/"
               "all-access/%s/all-days" % (project, month))
        try:
            data = wiki.get(url, cache_dir, "pv_%s_%s" % (slug, month.replace("/", "-")))
        except RuntimeError as exc:
            # a month can transiently 404 through some exit nodes; skipping one
            # or two months out of 48 changes the pool only marginally
            print("pageviews %s skipped: %s" % (month, exc), file=sys.stderr, flush=True)
            continue
        for article in (data.get("items") or [{}])[0].get("articles", []):
            title = article.get("article", "").replace("_", " ")
            if not title or title in SKIP_TITLES or title.startswith(SKIP_PREFIXES):
                continue
            views = article.get("views", 0)
            if title not in ranked or views > ranked[title]:
                ranked[title] = views
    ordered = sorted(ranked.items(), key=lambda kv: -kv[1])[:limit]
    return ordered


def resolve_qids(wiki, cache_dir, titles, api_host="en.wikipedia.org"):
    """Wikipedia titles -> Wikidata QIDs (batched, 50 per request)."""
    result = {}
    for start in range(0, len(titles), 50):
        chunk = titles[start:start + 50]
        params = urlencode({"action": "query", "format": "json", "formatversion": "2",
                            "prop": "pageprops", "ppprop": "wikibase_item",
                            "redirects": "1", "titles": "|".join(chunk)})
        key = "qids_" + hashlib.sha256(("|".join(chunk)).encode()).hexdigest()[:24]
        try:
            data = wiki.get("https://%s/w/api.php?%s" % (api_host, params), cache_dir, key)
        except RuntimeError as exc:
            print("qid batch skipped (%s): %s" % (chunk[0], exc), file=sys.stderr, flush=True)
            continue
        for page in (data.get("query", {}).get("pages") or []):
            qid = (page.get("pageprops") or {}).get("wikibase_item")
            if qid and page.get("title"):
                result[page["title"]] = qid
    return result


def fetch_types(wiki, cache_dir, qids, languages="en|zh"):
    """QID -> (label, [P31 ids]) via the Wikidata API (batches of 50)."""
    result = {}
    for start in range(0, len(qids), 50):
        chunk = qids[start:start + 50]
        params = urlencode({"action": "wbgetentities", "format": "json",
                            "ids": "|".join(chunk), "props": "labels|claims",
                            "languages": languages})
        key = "types_%s_%s" % (languages.replace("|", "-"),
                               hashlib.sha256("|".join(chunk).encode()).hexdigest()[:24])
        try:
            data = wiki.get("https://www.wikidata.org/w/api.php?" + params, cache_dir, key)
        except RuntimeError as exc:
            print("type batch skipped (%s): %s" % (chunk[0], exc), file=sys.stderr, flush=True)
            continue
        for qid, entity in (data.get("entities") or {}).items():
            labels = entity.get("labels") or {}
            english = (labels.get("en") or {}).get("value", "")
            chinese = (labels.get("zh") or {}).get("value", "")
            label = english or chinese
            p31 = []
            for claim in (entity.get("claims") or {}).get("P31", []):
                value = ((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value")
                if isinstance(value, dict) and value.get("id"):
                    p31.append(value["id"])
            if label:
                result[qid] = (label, p31, chinese)
    return result


def classify(p31_ids):
    for category, wanted in CATEGORY_MAP.items():
        if any(qid in wanted for qid in p31_ids):
            return category
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--output", type=Path, default=Path("data/wiki_entities/entities.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/wiki"))
    parser.add_argument("--months", type=int, default=3, help="How many recent months of pageviews to union")
    parser.add_argument("--top", type=int, default=1000, help="Keep top-N viewed articles per month")
    parser.add_argument("--quota-person", type=int, default=150)
    parser.add_argument("--quota-country", type=int, default=50)
    parser.add_argument("--quota-place", type=int, default=80)
    parser.add_argument("--quota-organization", type=int, default=60)
    parser.add_argument("--quota-work", type=int, default=40)
    parser.add_argument("--quota-event", type=int, default=40)
    parser.add_argument("--wiki-lang", default="en", help="Wikipedia language edition (en default; zh also supported)")
    parser.add_argument("--clash-controller", default=None,
                        help="ClashX REST controller used to rotate exit IPs")
    parser.add_argument("--clash-group", default="🔰 选择节点")
    parser.add_argument("--clash-nodes", help="Comma-separated node subset; default: all usable nodes in the group")
    parser.add_argument("--rotate-every", type=int, default=50, help="Rotate exit IP every N successful requests")
    parser.add_argument("--no-rotate", action="store_true", help="Disable exit-IP rotation")
    args = parser.parse_args(argv)
    quotas = {"person": args.quota_person, "country": args.quota_country, "place": args.quota_place,
              "organization": args.quota_organization, "work": args.quota_work, "event": args.quota_event}
    rotator = None
    if not args.no_rotate:
        nodes = [n.strip() for n in args.clash_nodes.split(",")] if args.clash_nodes else None
        rotator = NodeRotator(controller=args.clash_controller, group=args.clash_group,
                              nodes=nodes, rotate_every=args.rotate_every)
        atexit.register(rotator.restore)
    wiki = Wiki(args.proxy, rotator=rotator)
    project = "%s.wikipedia" % args.wiki_lang
    api_host = "%s.wikipedia.org" % args.wiki_lang
    months = latest_months(args.months)
    print(json.dumps({"lang": args.wiki_lang, "months": months, "top": args.top,
                      "quotas": quotas}, ensure_ascii=False), flush=True)
    ranked = fetch_pageviews(wiki, args.cache_dir, months, args.top, project=project)
    print("候选文章: %d" % len(ranked), flush=True)
    titles = [title for title, _ in ranked]
    qid_by_title = resolve_qids(wiki, args.cache_dir, titles, api_host=api_host)
    print("解析到 QID: %d" % len(qid_by_title), flush=True)
    types = fetch_types(wiki, args.cache_dir, sorted(set(qid_by_title.values())))
    entities, counts = [], defaultdict(int)
    seen_qids = set()
    for title, views in ranked:
        qid = qid_by_title.get(title)
        if not qid or qid in seen_qids or qid not in types:
            continue
        label, p31, label_zh = types[qid]
        category = classify(p31)
        if not category or counts[category] >= quotas[category]:
            continue
        seen_qids.add(qid)
        counts[category] += 1
        entities.append({
            "qid": qid, "title": title, "label": label, "label_zh": label_zh or None,
            "category": category, "category_label": CATEGORY_LABELS[category],
            "types_p31": p31, "monthly_views_max": views,
            "wiki_url": "https://%s/wiki/%s" % (api_host, quote(title.replace(" ", "_"))),
            "wiki_lang": args.wiki_lang,
        })
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for entity in entities:
            handle.write(json.dumps(entity, ensure_ascii=False) + "\n")
    temporary.replace(out)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "months": months,
              "candidates": len(ranked), "resolved": len(qid_by_title),
              "selected": len(entities), "by_category": dict(counts),
              "quotas": quotas, "wiki_stats": wiki.stats, "output": str(out),
              "notes": ["Ranking uses zh.wikipedia monthly pageviews (max over the months), "
                        "so 'common' means actually viewed by readers.",
                        "Types come from Wikidata P31 (direct instance-of only)."]}
    Path(str(out) + ".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps({"status": "ok", "entities": len(entities), "by_category": dict(counts),
                      "output": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
