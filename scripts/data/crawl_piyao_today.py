#!/usr/bin/env python3
"""Crawl the "今日辟谣" column and other text columns of piyao.org.cn.

By default this crawls the 今日辟谣 (Today's Rumor Refutation) column; pass
--column-url one or more times (e.g. .../gz.htm .../xzbb.htm) to crawl other
columns.  All columns are merged by article URL and every record carries its
source column name.

The column's complete listing is delivered as a pseudo-data JSON referenced by
the index page (data="datasource:<id>" -> ./ds_<id>.json, loaded client-side by
the Xinhua page widget `Xhwpage`).  This script resolves that listing, filters
it by publication date, fetches every article (HTML cache, polite delay,
robots.txt honoured), parses the per-item sections, and writes:

  articles.jsonl   one raw-article record per daily post, shaped like
                   01_crawl.py output so it can feed mtimefact.data.graph_building
  claims.jsonl     one record per claim item: the original claim text, its
                   refuting evidence (真相 / 详情 / 事实 paragraph plus any
                   following paragraphs), the evidence source attribution
                   parsed from a trailing （来源：…）, every date mentioned in
                   the claim or evidence, and the post's publication date
  sections.jsonl   every non-boilerplate item of every post (label, heading
                   text, body paragraphs, images, is_claim_item flag), so
                   advisory/explainer items such as 提示/热点/科普 are kept
                   structured as well; claims.jsonl is the filtered view
  report.json      crawl statistics, per-year counts and honest coverage notes

Only the standard library is used.  Re-running reuses the on-disk HTML cache,
so an interrupted crawl resumes without new network traffic for cached pages.

Site facts verified before writing this script (2026-10):
  * robots.txt / sitemap.xml return 404; no crawl directives are published.
  * The 今日辟谣 column's listing currently holds 920 entries, published
    2023-02-01 .. today.  No 今日辟谣 entries exist on the site for 2021-2022.
  * Early daily posts embed the rumor text inside poster images; the text layer
    carries the 真相/事实/详情 corrections.  Image URLs are recorded so a later
    OCR/vision pass can recover those claims.
"""
from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
import re
import sys
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.robotparser import RobotFileParser

BASE = "https://www.piyao.org.cn/"
COLUMN_URL = BASE + "jrpy/index.htm"
DATASOURCE_RE = re.compile(r'data="datasource:([0-9a-f]{32})"')
LIST_DATASOURCE_RES = (
    re.compile(r'<[^>]*id="list"[^>]*data="datasource:([0-9a-f]{32})"'),
    re.compile(r'<[^>]*data="datasource:([0-9a-f]{32})"[^>]*id="list"'),
)
PLATFORM_HOSTS = {"www.piyao.org.cn", "piyao.org.cn"}

# Paragraph headings, in spaced form ("辟 谣") as used by recent articles.
# Canonical label -> regex allowing optional whitespace between characters.
LABEL_PATTERNS = [
    ("警方提示", r"警\s*方\s*提\s*示"),
    ("责任编辑", r"责\s*任\s*编\s*辑"),
    ("举报电话", r"举\s*报\s*电\s*话"),
    ("举报网址", r"举\s*报\s*网\s*址"),
    ("举报客户端", r"举\s*报\s*客\s*户\s*端"),
    ("谣言", r"谣\s*言"),
    ("辟谣", r"辟\s*谣"),
    ("真相", r"真\s*相"),
    ("误区", r"误\s*区(?:\s*(?:[一二三四五六七八九十]|\d+))?"),
    ("详情", r"详\s*情"),
    ("科普", r"科\s*普"),
    ("事实", r"事\s*实"),
    ("提示", r"提\s*示"),
    ("通报", r"通\s*报"),
    ("热点", r"热\s*点"),
    ("关注", r"关\s*注"),
    ("警惕", r"警\s*惕"),
    ("重要", r"重\s*要"),
    ("自行办理", r"自\s*行\s*办\s*理"),
    ("案例", r"案\s*例(?:\s*(?:[一二三四五六七八九十]|\d+))?"),
    ("监制", r"监\s*制"),
    ("统筹", r"统\s*筹"),
    ("来源", r"来\s*源"),
    ("文字", r"文\s*字"),
    ("设计", r"设\s*计"),
    ("执行", r"执\s*行"),
    ("编辑", r"编\s*辑"),
    ("策划", r"策\s*划"),
    ("审核", r"审\s*核"),
]
CLAIM_LABELS = {"谣言", "辟谣", "误区"}
CORRECTION_LABELS = {"真相", "详情", "事实", "科普"}
META_LABELS = {"监制", "统筹", "来源", "文字", "设计", "执行", "编辑", "责任编辑", "策划", "审核"}
# Remaining labels (提示/通报/热点/关注/警惕/重要/案例/举报*/自行办理/警方提示) become "section" items.


class FetchError(Exception):
    def __init__(self, reason, detail=""):
        super().__init__(detail or reason)
        self.reason = reason


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Client:
    """Cached, rate-limited fetcher. Never bypasses robots.txt or challenges."""

    def __init__(self, cache_dir, delay, timeout, retries, user_agent):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.delay, self.timeout, self.retries, self.user_agent = delay, timeout, retries, user_agent
        self.last_request, self.robots = {}, {}
        self.stats = Counter()
        self.opener = build_opener(NoRedirect())

    def _request(self, url, delay=None):
        origin = "{0.scheme}://{0.netloc}".format(urlsplit(url))
        interval = max(self.delay, delay or 0)
        wait = interval - (time.monotonic() - self.last_request.get(origin, 0))
        if wait > 0:
            time.sleep(wait)
        self.last_request[origin] = time.monotonic()
        self.stats["network_requests"] += 1
        request = Request(url, headers={"User-Agent": self.user_agent,
                                        "Accept": "text/html,application/json,application/xml,text/xml,*/*;q=0.1"})
        with self.opener.open(request, timeout=self.timeout) as response:
            payload = response.read(10_000_001)
            if len(payload) > 10_000_000:
                raise FetchError("too_large", "Response exceeds 10 MB limit")
            charset = response.headers.get_content_charset()
            if not charset:
                match = re.search(br"charset\s*=\s*[\"']?([\w-]+)", payload[:4096], re.I)
                charset = match.group(1).decode("ascii", errors="ignore") if match else "utf-8"
            try:
                return payload.decode(charset, errors="replace")
            except LookupError:
                return payload.decode("utf-8", errors="replace")

    def _robots(self, url):
        origin = "{0.scheme}://{0.netloc}".format(urlsplit(url))
        if origin in self.robots:
            return self.robots[origin]
        parser = RobotFileParser(origin + "/robots.txt")
        try:
            parser.parse(self._request(origin + "/robots.txt").splitlines())
        except HTTPError as exc:
            if exc.code in {404, 410}:
                parser.parse([])  # explicitly absent robots.txt: no directives published
            else:
                raise FetchError("robots_unavailable", f"robots.txt HTTP {exc.code}; source not crawled") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise FetchError("robots_unavailable", f"robots.txt unavailable: {exc}") from exc
        self.robots[origin] = parser
        return parser

    def fetch(self, url, redirects=0):
        parser = self._robots(url)
        if not parser.can_fetch(self.user_agent, url):
            raise FetchError("robots_denied", "robots.txt disallows this URL for the crawler user agent")
        suffix = Path(urlsplit(url).path).suffix.lower()
        if suffix not in {".html", ".htm", ".json", ".xml"}:
            suffix = ".bin"
        key = hashlib.sha256(url.encode()).hexdigest()
        cached, sidecar = self.cache / (key + suffix), self.cache / (key + ".meta.json")
        if cached.exists() and sidecar.exists():
            self.stats["cache_hits"] += 1
            return cached.read_text(encoding="utf-8"), json.loads(sidecar.read_text(encoding="utf-8"))
        crawl_delay = parser.crawl_delay(self.user_agent) or parser.crawl_delay("*") or 0
        rate = parser.request_rate(self.user_agent) or parser.request_rate("*")
        if rate and rate.requests:
            crawl_delay = max(crawl_delay, rate.seconds / rate.requests)
        for attempt in range(self.retries + 1):
            try:
                text = self._request(url, crawl_delay)
                lowered = text[:10000].lower()
                if any(marker in lowered for marker in ("<title>just a moment", "cf-chl-", "verify you are human")):
                    raise FetchError("access_blocked", "Publisher challenge page; no bypass attempted")
                info = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "html_sha256": hashlib.sha256(text.encode()).hexdigest()}
                cached.write_text(text, encoding="utf-8")
                sidecar.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
                return text, info
            except HTTPError as exc:
                if exc.code in {301, 302, 303, 307, 308}:
                    # Old CMS articles were migrated; follow only same-platform hops.
                    location = exc.headers.get("Location", "") if exc.headers else ""
                    target = urljoin(url, location)
                    if redirects < 5 and target != url and urlsplit(target).hostname in PLATFORM_HOSTS:
                        self.stats["redirects"] += 1
                        exc.close()
                        return self.fetch(target, redirects + 1)
                    exc.close()
                    raise FetchError("redirect_blocked", "redirect outside the platform or too deep: " + target) from None
                if exc.code in {401, 402, 403, 451}:
                    raise FetchError("access_blocked", f"HTTP {exc.code}; no bypass attempted") from exc
                if exc.code not in {429, 500, 502, 503, 504} or attempt == self.retries:
                    raise FetchError("http_error", f"HTTP {exc.code}") from exc
                if exc.code == 429:
                    raise FetchError("rate_limited", "HTTP 429; stop and retry the run later") from exc
            except (URLError, TimeoutError, OSError) as exc:
                if attempt == self.retries:
                    raise FetchError("network_error", str(exc)) from exc
            self.stats["retries"] += 1
            time.sleep(min(2 ** attempt, 8))
        raise FetchError("network_error", "Retry budget exhausted")


HEAD_RE = re.compile(
    r"^(?P<label>" + "|".join(pattern for _, pattern in LABEL_PATTERNS) + r")(?P<sep>[\s：:]+)(?P<rest>.*)$",
    re.S)


def normalize_label(raw_label):
    compact = re.sub(r"\s+", "", raw_label)
    for name, _ in LABEL_PATTERNS:
        if compact == name or (name in {"误区", "案例"} and re.fullmatch(name + r"[一二三四五六七八九十\d]+", compact)):
            return name
    return compact


def heading_of(paragraph):
    """Return (label, rest) when the paragraph opens a new item, else None.

    The site uses three heading styles: ``谣言：...`` (colon), the spaced label
    form ``辟 谣 网传...``, and label plus whitespace only such as
    ``热点 农业农村部辟谣“猪病炒作”``.  HEAD_RE requires a colon or
    whitespace after the label, so a body sentence that merely starts with the
    same characters ("谣言止于智者") is deliberately not a heading.
    """
    match = HEAD_RE.match(paragraph)
    if not match:
        return None
    rest = re.sub(r"^[|｜·、—\-\s]+", "", match.group("rest")).strip()
    return normalize_label(match.group("label")), rest


def strip_tags(value):
    return html_lib.unescape(re.sub(r"<[^>]+>", "", value))


SOURCE_RE = re.compile(r"[（(]\s*来源\s*[:：]\s*(?P<src>[^）)]*?)\s*[）)]")
CHINESE_DATE_RE = re.compile(r"(?P<y>20\d{2})\s*年\s*(?P<m>\d{1,2})\s*月(?:\s*(?P<d>\d{1,2})\s*[日号])?")
ISO_DATE_RE = re.compile(r"(?P<y>20\d{2})-(?P<m>\d{2})-(?P<d>\d{2})")
MONTH_DAY_RE = re.compile(r"(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*[日号]")


def split_source(text):
    """Detach a trailing （来源：…） attribution from evidence text.

    Only an attribution that closes the text is removed from the prose; the
    attribution itself is returned so it is kept as a structured field.
    """
    if not text:
        return text, None
    matches = list(SOURCE_RE.finditer(text))
    if not matches:
        return text, None
    last = matches[-1]
    if text[last.end():].strip("。 \t\r\n"):
        return text, None
    return text[:last.start()].strip(), last.group("src").strip() or None


def extract_dates(text):
    """Every explicit date in text: ISO-normalized when complete, raw kept.

    Chinese full dates (2024年3月30日) normalize to ISO; bare month-day
    references (3月30日) and year-month (2024年3月) keep precision=month_day /
    month with iso=null, because the year is not stated in the text.
    """
    found, seen = [], set()

    def add(raw, iso, precision):
        if raw not in seen:
            seen.add(raw)
            found.append({"raw": raw, "iso": iso, "precision": precision})

    if not text:
        return found

    def mask(pattern, source):
        raws = []

        def _mask(match):
            raws.append(match.group(0))
            return " " * len(match.group(0))

        return pattern.sub(_mask, source), raws

    masked, chinese = mask(CHINESE_DATE_RE, text)
    for raw in chinese:
        match = CHINESE_DATE_RE.fullmatch(raw)
        iso = None
        if match.group("d"):
            try:
                iso = date(int(match.group("y")), int(match.group("m")), int(match.group("d"))).isoformat()
            except ValueError:
                iso = None
        add(raw, iso, "day" if match.group("d") else "month")
    masked, iso_raws = mask(ISO_DATE_RE, masked)
    for raw in iso_raws:
        try:
            add(raw, date.fromisoformat(raw).isoformat(), "day")
        except ValueError:
            pass
    _, month_days = mask(MONTH_DAY_RE, masked)
    for raw in month_days:
        add(raw, None, "month_day")
    return found


def article_parts(page):
    """Return (paragraph stream, image urls) parsed from the article container.

    Newer pages use #detailContent; the pre-2023 CMS template uses .con_txt.
    The stream is a list of ("text", str) / ("img", index) entries in document
    order; footer boilerplate after 责任编辑 / con_right / share markers is cut.
    """
    container = re.search(r'<div[^>]*\bid="detailContent"[^>]*>', page)
    if not container:
        container = re.search(r'<div[^>]*\bclass="[^"]*\bcon_txt\b[^"]*"[^>]*>', page)
    segment = page[container.end():] if container else page
    for marker in ("责任编辑", 'class="con_right"', 'id="con_right"', 'class="fx"', 'class="share-'):
        cut = segment.find(marker)
        if cut > 0:
            segment = segment[:cut]
    segment = re.sub(r"<script\b.*?</script>", " ", segment, flags=re.S | re.I)
    segment = re.sub(r"<style\b.*?</style>", " ", segment, flags=re.S | re.I)
    images = []

    def _mark(match):
        src = re.search(r'src="([^"]+)"', match.group(0), re.I)
        images.append(src.group(1).strip() if src else "")
        return "\n@@IMG%d@@\n" % (len(images) - 1)

    segment = re.sub(r"<img\b[^>]*>", _mark, segment, flags=re.I)
    segment = re.sub(r"<br\b[^>]*>", "\n@@\n", segment, flags=re.I)
    segment = re.sub(r"</(?:p|div|h[1-6]|li|td|tr|section|article|blockquote)>", "\n@@\n", segment, flags=re.I)
    text = html_lib.unescape(re.sub(r"<[^>]+>", "", segment))
    stream = []
    for raw in text.split("@@"):
        clean = re.sub(r"[\s\u3000\u2002\u2003\u00a0]+", " ", raw).strip()
        marker = re.fullmatch(r"IMG(\d+)", clean)
        if marker:
            stream.append(("img", int(marker.group(1))))
        elif clean:
            stream.append(("text", clean))
    return stream, images


def parse_article(page, url):
    """Parse one article into title/sections/claims/body text."""
    title_match = re.search(r'<div class="share-title"[^>]*>(.*?)</div>', page, re.S)
    if title_match:
        title = strip_tags(title_match.group(1)).strip()
    else:
        h1 = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S)
        raw_title = re.search(r"<title>(.*?)</title>", page, re.S)
        title = strip_tags(h1.group(1)).strip() if h1 else strip_tags(raw_title.group(1)).strip()
    title = re.sub(r"\s*[-—_]中国互联网联合辟谣平台\s*$", "", title).strip()
    date_match = re.search(r"时间：\s*[^\d]*(20\d{2}-\d{2}-\d{2})", page) or \
                 re.search(r"publishTime\s*[:=]\s*\"?(20\d{2}-\d{2}-\d{2})", page)
    stream, raw_images = article_parts(page)
    images = [urljoin(url, src) for src in raw_images if src and not src.startswith("data:")]
    items, lead_images, lead_text = [], [], []
    for kind, value in stream:
        if kind == "img":
            if items:
                items[-1]["image_indices"].append(value)
            else:
                lead_images.append(value)
            continue
        heading = heading_of(value)
        if heading:
            items.append({"label": heading[0], "heading_text": heading[1], "body": [], "image_indices": []})
        elif items:
            items[-1]["body"].append(value)
        else:
            lead_text.append(value)
    if not items and lead_text:
        # Articles without 谣言/真相-style headings (news posts, old CMS pages):
        # keep the full text as one plain section instead of dropping it.
        items.append({"label": "正文", "heading_text": "", "body": lead_text,
                      "image_indices": lead_images[:]})
    for item in items:
        item["images"] = [images[i] for i in item["image_indices"] if i < len(images)]
    sections = [{"label": item["label"], "text": item["heading_text"],
                 "body": item["body"], "images": item["images"]}
                for item in items if item["label"] not in META_LABELS]
    claims = []
    for index, item in enumerate(items):
        if item["label"] not in CLAIM_LABELS or not item["heading_text"]:
            continue
        correction = items[index + 1] if index + 1 < len(items) and items[index + 1]["label"] in CORRECTION_LABELS else None
        evidence_from = None
        evidence = None
        if correction:
            evidence = "\n".join([correction["heading_text"]] + correction["body"]).strip() or None
            evidence_from = "following_correction"
        elif item["body"]:
            # Sub-items such as 误区一/二/三 carry their explanation in their own body.
            evidence = "\n".join(item["body"]).strip() or None
            evidence_from = "same_item_body"
        evidence, evidence_source = split_source(evidence)
        claims.append({
            "label": item["label"],
            "claim": item["heading_text"],
            "claim_dates": extract_dates(item["heading_text"]),
            "correction": correction["heading_text"] if correction else None,
            "correction_label": correction["label"] if correction else None,
            "correction_body": correction["body"] if correction else [],
            "evidence": evidence,
            "evidence_source": evidence_source,
            "evidence_from": evidence_from,
            "evidence_dates": extract_dates(evidence or ""),
        })
    text_lines = []
    for item in items:
        if item["label"] in META_LABELS:
            continue
        if item["heading_text"]:
            text_lines.append(item["label"] + "：" + item["heading_text"])
        text_lines.extend(item["body"])
    return {"title": title, "published_at": date_match.group(1) if date_match else None,
            "sections": sections, "claims": claims,
            "body_text": "\n".join(line for line in text_lines if line.strip()),
            "images": images, "lead_images": [images[i] for i in lead_images if i < len(images)]}


def load_listing(client, column_url, listing_file=None):
    """Resolve one column's listing JSON (page -> #list datasource -> ds_<id>.json)."""
    if listing_file:
        raw = json.loads(Path(listing_file).read_text(encoding="utf-8"))
        source = str(listing_file)
    else:
        page, _ = client.fetch(column_url)
        match = None
        for regex in LIST_DATASOURCE_RES:
            match = regex.search(page)
            if match:
                break
        if not match:
            match = DATASOURCE_RE.search(page)
        if not match:
            raise FetchError("listing_parse", "datasource id not found on " + column_url)
        ds_url = urljoin(column_url, "ds_%s.json" % match.group(1))
        raw = json.loads(client.fetch(ds_url)[0])
        source = ds_url
    rows = raw.get("datasource")
    if not isinstance(rows, list) or not rows:
        raise FetchError("listing_parse", "listing JSON contains no datasource array")
    items, skipped = [], []
    for row in rows:
        raw_url = (row.get("publishUrl") or "").strip()
        published = (row.get("publishTime") or "")[:10]
        target = urljoin(column_url, raw_url)
        if (not raw_url or not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", published)
                or urlsplit(target).hostname not in PLATFORM_HOSTS):
            # External links (xinhuanet interviews, local-platform microsites) stay
            # out of scope: this crawler's contract is piyao.org.cn itself.
            skipped.append({"title": re.sub(r"<[^>]+>", "", row.get("title") or "").strip()[:60],
                            "publishUrl": raw_url, "publishTime": row.get("publishTime"),
                            "contentType": row.get("contentType")})
            continue
        items.append({"url": target, "title": re.sub(r"<[^>]+>", "", row.get("title") or "").strip(),
                      "published_at": published, "publish_time": row.get("publishTime"),
                      "content_id": row.get("contentId")})
    if not items:
        raise FetchError("listing_parse", "listing has no same-site rows for " + column_url)
    return items, {"source": source, "category": raw.get("categoryDisplayName"),
                   "list_url": raw.get("listUrl"), "column_url": column_url, "total": len(items),
                   "skipped_rows": len(skipped), "skipped_samples": skipped[:5],
                   "range": [min(item["published_at"] for item in items),
                             max(item["published_at"] for item in items)]}


def write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("data/piyao_today"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/piyao"))
    parser.add_argument("--column-url", action="append",
                        help="Column page URL, repeatable (e.g. https://www.piyao.org.cn/rm.htm); default: the 今日辟谣 column")
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2021, 1, 1),
                        help="Earliest publication date (default 2021-01-01; the column itself starts 2023-02-01)")
    parser.add_argument("--end-date", type=date.fromisoformat, default=None, help="Latest publication date (default today)")
    parser.add_argument("--listing-file", type=Path, help="Optional local ds_<id>.json snapshot instead of fetching")
    parser.add_argument("--limit", type=int, help="Process only the first N selected articles (testing)")
    parser.add_argument("--delay", type=float, default=1.0, help="Minimum seconds per host (default 1.0, polite)")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--user-agent", default="MTimeFact/0.1 (research crawler)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs in --output-dir")
    args = parser.parse_args(argv)
    args.end_date = args.end_date or date.today()
    if args.start_date > args.end_date:
        parser.error("--start-date must not be after --end-date")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.delay < 0 or args.timeout <= 0 or not 0 <= args.retries <= 5:
        parser.error("delay >= 0, timeout > 0, and retries between 0 and 5 are required")
    return args


def main(argv=None):
    args = parse_args(argv)
    out = args.output_dir
    targets = [out / "articles.jsonl", out / "claims.jsonl", out / "sections.jsonl",
               out / "report.json", out / "listing.json"]
    if not args.overwrite and any(path.exists() for path in targets):
        print("Output exists; choose a new directory or use --overwrite", file=sys.stderr)
        return 1
    client = Client(args.cache_dir, args.delay, args.timeout, args.retries, args.user_agent)
    merged, columns_info = {}, []
    try:
        for column_url in args.column_url or [COLUMN_URL]:
            column_items, info = load_listing(client, column_url, args.listing_file)
            columns_info.append(info)
            for item in column_items:
                if item["url"] in merged:
                    merged[item["url"]]["columns"].append(info.get("category") or column_url)
                    continue
                item["column"] = info.get("category") or column_url
                item["columns"] = [item["column"]]
                merged[item["url"]] = item
    except (FetchError, ValueError, OSError) as exc:
        print("Listing unavailable: " + str(exc), file=sys.stderr)
        return 1
    items = list(merged.values())
    listing_info = {"columns": columns_info, "total_unique": len(items),
                    "total_rows": sum(info["total"] for info in columns_info)}
    listing_dates = sorted(item["published_at"] for item in items)
    selected = [item for item in items if args.start_date.isoformat() <= item["published_at"] <= args.end_date.isoformat()]
    selected.sort(key=lambda item: (item["published_at"], item["publish_time"], item["url"]))
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        print("No listing entries inside %s .. %s" % (args.start_date, args.end_date), file=sys.stderr)
        return 2
    articles, claims, sections, failures = [], [], [], []
    for number, item in enumerate(selected, 1):
        try:
            page, info = client.fetch(item["url"])
            parsed = parse_article(page, item["url"])
            article_id = "raw_" + hashlib.sha256(("piyao:" + item["url"]).encode()).hexdigest()[:20]
            record = {
                "id": article_id, "source": "piyao", "url": item["url"], "title": parsed["title"] or item["title"],
                "language": "zh", "published_at": item["published_at"], "retrieved_at": info.get("retrieved_at"),
                "text": parsed["body_text"],
                "metadata": {
                    "column": item["column"],
                    "columns": item["columns"],
                    "listing_title": item["title"],
                    "publish_time": item["publish_time"],
                    "content_id": item["content_id"],
                    "html_date": parsed["published_at"],
                    "labels": [section["label"] for section in parsed["sections"]],
                    "item_count": len(parsed["sections"]),
                    "claim_count": len(parsed["claims"]),
                    "image_urls": parsed["images"],
                    "html_sha256": info.get("html_sha256"),
                },
            }
            articles.append(record)
            for index, claim in enumerate(parsed["claims"], 1):
                claims.append({
                    "id": "claim_" + hashlib.sha256(
                        ("piyao:%s:%d:%s" % (item["url"], index, claim["claim"])).encode()).hexdigest()[:20],
                    "article_id": article_id, "source": "piyao", "column": item["column"],
                    "url": item["url"], "article_title": record["title"],
                    "published_at": item["published_at"], "published_time": item["publish_time"],
                    "item_index": index,
                    "item_label": claim["label"], "claim": claim["claim"],
                    "claim_dates": claim["claim_dates"],
                    "label": False,  # every extracted claim is a refuted rumor statement
                    "source_verdict": "false",  # the column publishes refuted rumors; basis recorded below
                    "verdict_basis": "column_labels_the_statement_as_refuted_rumor",
                    "evidence": claim["evidence"],
                    "evidence_source": claim["evidence_source"],
                    "evidence_from": claim["evidence_from"],
                    "evidence_dates": claim["evidence_dates"],
                    "correction": claim["correction"],
                    "correction_label": claim["correction_label"],
                    "correction_body": claim["correction_body"],
                    "images": next((section["images"] for section in parsed["sections"]
                                    if section["text"] == claim["claim"] and section["label"] == claim["label"]), []),
                })
            for index, section in enumerate(parsed["sections"], 1):
                sections.append({
                    "id": "section_" + hashlib.sha256(
                        ("piyao:%s:%d" % (item["url"], index)).encode()).hexdigest()[:20],
                    "article_id": article_id, "source": "piyao", "column": item["column"],
                    "url": item["url"], "article_title": record["title"],
                    "published_at": item["published_at"], "published_time": item["publish_time"],
                    "item_index": index,
                    "item_label": section["label"], "text": section["text"],
                    "body": section["body"], "images": section["images"],
                    "is_claim_item": section["label"] in CLAIM_LABELS,
                })
            print("article %d/%d %s ok claims=%d %s" % (number, len(selected), item["published_at"],
                                                        len(parsed["claims"]), item["url"]), flush=True)
        except (FetchError, ValueError, TypeError, OSError, KeyError) as exc:
            failures.append({"url": item["url"], "published_at": item["published_at"],
                             "reason": getattr(exc, "reason", type(exc).__name__), "detail": str(exc)})
            print("article %d/%d %s FAILED %s: %s %s" % (number, len(selected), item["published_at"],
                                                          getattr(exc, "reason", type(exc).__name__), exc, item["url"]), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "articles.jsonl", articles)
    write_jsonl(out / "claims.jsonl", claims)
    write_jsonl(out / "sections.jsonl", sections)
    write_json(out / "listing.json", {"retrieved_at": datetime.now(timezone.utc).isoformat(),
                                      "listing": listing_info, "items": items})
    claims_by_year = Counter(claim["published_at"][:4] for claim in claims)
    claims_by_label = Counter(claim["item_label"] for claim in claims)
    articles_by_column = Counter(article["metadata"]["column"] for article in articles)
    claims_by_column = Counter(claim["column"] for claim in claims)
    no_claim = [article["url"] for article in articles if article["metadata"]["claim_count"] == 0]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "piyao", "columns": [info.get("category") or info["column_url"] for info in columns_info],
        "listing": listing_info,
        "listing_date_range": [listing_dates[0], listing_dates[-1]] if listing_dates else None,
        "requested_range": [args.start_date.isoformat(), args.end_date.isoformat()],
        "selected_articles": len(selected), "articles_parsed": len(articles), "failures": failures,
        "articles_by_column": dict(articles_by_column), "claims_by_column": dict(claims_by_column),
        "claims": len(claims), "claims_by_year": dict(sorted(claims_by_year.items())),
        "claims_by_label": dict(claims_by_label),
        "claims_with_evidence": sum(1 for claim in claims if claim["evidence"]),
        "claims_with_source": sum(1 for claim in claims if claim["evidence_source"]),
        "claims_with_claim_dates": sum(1 for claim in claims if claim["claim_dates"]),
        "claims_with_evidence_dates": sum(1 for claim in claims if claim["evidence_dates"]),
        "sections": len(sections),
        "sections_by_label": dict(Counter(section["item_label"] for section in sections)),
        "claim_items_in_sections": sum(1 for section in sections if section["is_claim_item"]),
        "articles_without_claims": len(no_claim), "articles_without_claims_urls": no_claim[:50],
        "images_recorded": sum(len(article["metadata"]["image_urls"]) for article in articles),
        "network": dict(client.stats),
        "notes": [
            "Columns crawled: " + "; ".join(
                "%s %s..%s (%d rows)" % (info.get("category") or info["column_url"],
                                         info["range"][0], info["range"][1], info["total"])
                for info in columns_info),
            "Claims are paragraphs labelled 谣言/辟谣/误区 paired with the following 真相/详情/事实 correction.",
            "source_verdict=false records the column's editorial stance (it publishes refuted rumors), not an independent fact-check.",
            "Early posts (mostly 2023) embed rumor wording inside poster images; image_urls are recorded for a later OCR pass.",
            "Sections labelled 提示/通报/科普/热点/案例 are advisories or notices, not claims, and stay out of claims.jsonl.",
            "sections.jsonl keeps every item structured (is_claim_item marks the claim slots); 热点/关注 question-form items live there, not in claims.jsonl.",
        ],
    }
    write_json(out / "report.json", report)
    print(json.dumps({"status": "ok" if not failures else "partial", "articles": len(articles),
                      "claims": len(claims), "failures": len(failures),
                      "output": str(out.resolve())}, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
