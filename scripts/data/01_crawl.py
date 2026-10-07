
"""Crawl/import raw articles for MTimeFact (Python 3.9+, standard library only).

Run --help for bounded online crawling or local HTML import with fixed inputs.
No login, CAPTCHA handling, robots bypass, or automatic verdict inference.
"""
from __future__ import annotations


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
import sys
import time
from collections import Counter, deque
from datetime import date, datetime, timezone
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.robotparser import RobotFileParser

from mtimefact.data.crawl_support import (SOURCES, article_url, canonical_url, default_seeds, discovery_url,
                           discover_links, extract_record, source_for_url)


class FetchError(Exception):
    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Client:
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
        request = Request(url, headers={"User-Agent": self.user_agent, "Accept": "text/html,application/xhtml+xml,application/xml,text/xml;q=0.9,*/*;q=0.1"})
        with self.opener.open(request, timeout=self.timeout) as response:
            payload = response.read(10_000_001)
            if len(payload) > 10_000_000:
                raise FetchError("too_large", "Response exceeds 10 MB limit")
            charset = response.headers.get_content_charset()
            if not charset:
                
                import re
                match = re.search(br"charset\s*=\s*[\"']?([\w-]+)", payload[:4096], re.I)
                charset = match.group(1).decode("ascii", errors="ignore") if match else "utf-8"
            try:
                text = payload.decode(charset, errors="replace")
            except LookupError:
                text = payload.decode("utf-8", errors="replace")
            return text, response.headers.get("Content-Type", "")

    def _robots(self, url):
        origin = "{0.scheme}://{0.netloc}".format(urlsplit(url))
        if origin in self.robots:
            return self.robots[origin]
        parser = RobotFileParser(origin + "/robots.txt")
        try:
            text, _ = self._request(origin + "/robots.txt")
            parser.parse(text.splitlines())
        except HTTPError as exc:
            if exc.code in {404, 410}:
                parser.parse([])
            else:
                raise FetchError("robots_unavailable", f"robots.txt HTTP {exc.code}; source not crawled") from exc
        except (URLError, TimeoutError, OSError, HTTPException) as exc:
            raise FetchError("robots_unavailable", f"robots.txt unavailable: {exc}") from exc
        self.robots[origin] = parser
        return parser

    def sitemaps(self, url):
        return self._robots(url).site_maps() or []

    def fetch(self, url, redirects=0):
        parser = self._robots(url)
        if not parser.can_fetch(self.user_agent, url):
            raise FetchError("robots_denied", "robots.txt disallows this URL for crawler user agent")
        key = hashlib.sha256(url.encode()).hexdigest()
        cached = self.cache / (key + ".html")
        sidecar = self.cache / (key + ".json")
        if cached.exists() and sidecar.exists():
            self.stats["cache_hits"] += 1
            info = json.loads(sidecar.read_text(encoding="utf-8"))
            return cached.read_text(encoding="utf-8"), info
        crawl_delay = parser.crawl_delay(self.user_agent) or parser.crawl_delay("*") or 0
        rate = parser.request_rate(self.user_agent) or parser.request_rate("*")
        if rate and rate.requests:
            crawl_delay = max(crawl_delay, rate.seconds / rate.requests)
        for attempt in range(self.retries + 1):
            try:
                text, content_type = self._request(url, crawl_delay)
                lowered = text[:10000].lower()
                if any(marker in lowered for marker in ("<title>just a moment", "cf-chl-", "<title>access denied", "verify you are human")):
                    raise FetchError("access_blocked", "Publisher challenge/denial page; no bypass attempted")
                info = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(), "content_type": content_type,
                        "html_sha256": hashlib.sha256(text.encode()).hexdigest(), "cache_file": str(cached.resolve())}
                cached.write_text(text, encoding="utf-8")
                sidecar.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
                return text, info
            except HTTPError as exc:
                if exc.code in {301, 302, 303, 307, 308}:
                    target = canonical_url(urljoin(url, exc.headers.get("Location", "")))
                    if redirects < 5 and target != url and source_for_url(target) == source_for_url(url):
                        return self.fetch(target, redirects + 1)
                    raise FetchError("redirect_blocked", f"Redirect outside publisher or too many redirects: {target}") from exc
                if exc.code in {401, 402, 403, 451}:
                    raise FetchError("access_blocked", f"HTTP {exc.code}; no bypass attempted") from exc
                if exc.code not in {429, 500, 502, 503, 504} or attempt == self.retries:
                    raise FetchError("http_error", f"HTTP {exc.code}") from exc
                if exc.code == 429:
                    
                    raise FetchError("rate_limited", "HTTP 429; stop and retry the run later") from exc
            except (URLError, TimeoutError, OSError, HTTPException) as exc:
                if attempt == self.retries:
                    raise FetchError("network_error", str(exc)) from exc
            self.stats["retries"] += 1
            time.sleep(min(2 ** attempt, 8))
        raise FetchError("network_error", "Retry budget exhausted")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", choices=sorted(SOURCES), default=list(SOURCES))
    parser.add_argument("--seed-url", action="append", default=[], help="Article, listing, RSS or sitemap URL; repeatable")
    parser.add_argument("--urls-file", type=Path, help="UTF-8 file with one publisher URL per line (# comments allowed)")
    offline = parser.add_mutually_exclusive_group()
    offline.add_argument("--html-dir", type=Path, help="Import local .html/.htm files recursively without network")
    offline.add_argument("--fixture", type=Path, help="Import one local HTML fixture without network")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Default: OUTPUT.report.json")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/html"))
    parser.add_argument("--max-pages", type=int, default=100, help="Total page attempts (including listings); robots are additional")
    parser.add_argument("--max-articles", type=int, default=50)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2022, 11, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--keep-undated", action="store_true", help="Keep and explicitly flag dates unknown; default skips them")
    parser.add_argument("--delay", type=float, default=1.5, help="Minimum seconds per host; robots delay may increase this")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--min-text-chars", type=int, default=80)
    parser.add_argument("--user-agent", default="MTimeFact/0.1 (research crawler)")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing output/report")
    args = parser.parse_args(argv)
    if args.start_date > args.end_date:
        parser.error("--start-date must not be after --end-date")
    if args.max_pages <= 0 or args.max_articles <= 0 or args.min_text_chars < 1:
        parser.error("page/article limits and minimum text length must be positive")
    if args.delay < 0 or args.timeout <= 0 or not 0 <= args.retries <= 5:
        parser.error("delay >= 0, timeout > 0, and retries between 0 and 5 are required")
    if (args.fixture or args.html_dir) and (args.seed_url or args.urls_file):
        parser.error("offline HTML input cannot be combined with online seeds")
    args.report = args.report or Path(str(args.output) + ".report.json")
    if args.output.resolve() == args.report.resolve():
        parser.error("output and report must be different files")
    for path in (args.output, args.report):
        if path.exists() and not args.overwrite:
            parser.error(f"Refusing to overwrite {path}; choose a new path or pass --overwrite")
    return args


def run(args):
    counts, failures, records, record_ids = Counter(), [], [], set()
    client = None
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "sources": args.sources,
              "date_range": [args.start_date.isoformat(), args.end_date.isoformat()],
              "mode": "offline_html" if args.fixture or args.html_dir else "network",
              "limits": {"max_pages": args.max_pages, "max_articles": args.max_articles},
              "notes": ["The original collection scripts and exact URL manifest were not released.",
                        "A bounded run does not cover the paper's entire corpus.",
                        "Claim/verdict fields are copied only from ClaimReview JSON-LD when present; audit all extracted articles."]}

    def failure(url, reason, detail):
        counts[reason] += 1
        failures.append({"url": url, "reason": reason, "detail": str(detail)})

    def accept(html, source, url, info):
        record = extract_record(html, source, url, info.get("retrieved_at"))
        record["metadata"].update(info)
        if not record["title"] or len(record["text"]) < args.min_text_chars:
            failure(url, "parse_rejected", "Missing title or article text below --min-text-chars")
            return
        published = record["published_at"]
        if published is None:
            counts["undated"] += 1
            if not args.keep_undated:
                failure(url, "undated_skipped", "No explicit publication date; not inferred from body/URL")
                return
            record["metadata"]["date_range_verified"] = False
        elif not args.start_date <= date.fromisoformat(published) <= args.end_date:
            counts["out_of_range"] += 1
            return
        else:
            record["metadata"]["date_range_verified"] = True
        if record["id"] in record_ids:
            counts["duplicate_articles"] += 1
            return
        records.append(record)
        record_ids.add(record["id"])
        counts["articles_written"] += 1
        counts["articles_" + source] += 1

    if args.fixture or args.html_dir:
        paths = [args.fixture] if args.fixture else sorted(path for path in args.html_dir.rglob("*") if path.suffix.lower() in {".html", ".htm"})
        for path in paths[:args.max_pages]:
            if len(records) >= args.max_articles:
                break
            counts["page_attempts"] += 1
            try:
                sidecar = path.with_suffix(".json")
                info = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
                if not isinstance(info, dict):
                    raise ValueError("HTML sidecar must be an object")
                source = info.get("source", args.sources[0])
                if source not in args.sources:
                    raise ValueError("Sidecar source is not selected by --sources")
                url = info.get("url") or path.resolve().as_uri()
                accept(path.read_text(encoding="utf-8"), source, url,
                       {"fixture": True, "local_html": str(path.resolve()), "retrieved_at": info.get("retrieved_at")})
            except (OSError, ValueError, TypeError) as exc:
                failure(str(path), "local_input_error", exc)
    else:
        client = Client(args.cache_dir, args.delay, args.timeout, args.retries, args.user_agent)
        seeds = list(args.seed_url)
        if args.urls_file:
            seeds.extend(line.strip() for line in args.urls_file.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#"))
        explicit = bool(seeds)
        if not explicit:
            
            source_seeds = [deque(default_seeds(source, args.start_date, args.end_date)) for source in args.sources]
            while any(source_seeds):
                for group in source_seeds:
                    if group:
                        seeds.append(group.popleft())
        queue, seen = deque(seeds), set()
        blocked_sources = set()
        
        
        discovered_sitemaps = set()
        while queue and counts["page_attempts"] < args.max_pages and len(records) < args.max_articles:
            url = canonical_url(queue.popleft())
            if url in seen:
                continue
            seen.add(url)
            source = source_for_url(url)
            if urlsplit(url).scheme not in {"http", "https"} or source not in args.sources:
                failure(url, "unsupported_url", "URL is outside selected publisher root/www domains")
                continue
            if source in blocked_sources:
                counts["blocked_source_urls_skipped"] += 1
                continue
            counts["page_attempts"] += 1
            try:
                html, info = client.fetch(url)
                counts["pages_fetched"] += 1
                final_url = info.get("url", url)
                if article_url(source, final_url) or (explicit and url in seeds and not discovery_url(source, final_url) and "<html" in html.lower()):
                    accept(html, source, final_url, info)
                links = discover_links(html, final_url, source)
                articles = [link for link in links if article_url(source, link)]
                listings = [link for link in links if not article_url(source, link)]
                
                queue.extendleft(reversed(articles))
                queue.extend(listings)
                if not explicit and source not in discovered_sitemaps:
                    discovered_sitemaps.add(source)
                    queue.extend(link for link in client.sitemaps(url) if source_for_url(link) == source)
            except FetchError as exc:
                failure(url, exc.reason, exc)
                if exc.reason in {"access_blocked", "rate_limited", "robots_unavailable"}:
                    blocked_sources.add(source)
            except (ValueError, TypeError, OSError) as exc:
                failure(url, "parse_error", exc)
        report["urls_remaining"] = len(queue)
        report["blocked_sources"] = sorted(blocked_sources)
        report["network"] = dict(client.stats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    report.update({"finished_at": datetime.now(timezone.utc).isoformat(), "counts": dict(counts), "failures": failures,
                   "status": "success" if records else "no_usable_articles", "output": str(args.output.resolve())})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "articles": len(records), "report": str(args.report)}, ensure_ascii=False))
    return 0 if records else 2


def main():
    args = parse_args()
    try:
        return run(args)
    except (OSError, ValueError) as exc:
        print(f"Crawler input/output error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
