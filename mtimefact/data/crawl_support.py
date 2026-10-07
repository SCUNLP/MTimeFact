"""Small, auditable HTML/source adapters for the MTimeFact crawler.

These are engineering adapters, not the authors' unpublished original crawler.
Publication dates are never guessed from dates mentioned in the article body.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any, Iterator
from urllib.parse import urldefrag, urljoin, urlsplit
from xml.etree import ElementTree


SOURCES = {
    "factcheck": {"domain": "factcheck.org", "language": "en", "home": "https://www.factcheck.org/"},
    "snopes": {"domain": "snopes.com", "language": "en", "home": "https://www.snopes.com/fact-check/"},
    "piyao": {"domain": "piyao.org.cn", "language": "zh", "home": "https://www.piyao.org.cn/"},
}
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
SKIP = {"script", "style", "nav", "footer", "aside", "noscript", "form", "button", "svg"}
BLOCKS = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "li", "blockquote", "br"}


@dataclass
class Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[Any] = field(default_factory=list)

    def walk(self) -> Iterator["Node"]:
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk()

    def text(self, include_script: bool = False) -> str:
        if self.tag in SKIP and not include_script:
            return ""
        chunks = [child.text(include_script) if isinstance(child, Node) else child for child in self.children]
        result = "".join(chunks)
        return "\n" + result + "\n" if self.tag in BLOCKS else result


class Document(HTMLParser):
    def __init__(self, html: str):
        super().__init__(convert_charrefs=True)
        self.root = Node("document")
        self.stack = [self.root]
        self.feed(html)
        self.close()

    def handle_starttag(self, tag, attrs):
        node = Node(tag, dict((key, value or "") for key, value in attrs))
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def clean(text: str) -> str:
    return "\n".join(line for line in (re.sub(r"\s+", " ", line).strip() for line in text.splitlines()) if line)


def source_for_url(url: str) -> str | None:
    host = (urlsplit(url).hostname or "").lower()
    for name, config in SOURCES.items():
        
        if host in {config["domain"], "www." + config["domain"]}:
            return name
    return None


def canonical_url(url: str) -> str:
    return urldefrag(url)[0]


def article_url(source: str, url: str) -> bool:
    path = urlsplit(url).path
    if source == "factcheck":
        return bool(re.match(r"^/20\d{2}/\d{2}/[^/]+/?$", path))
    if source == "snopes":
        return bool(re.match(r"^/fact-check/[^/]+/?$", path))
    return bool(re.search(r"/(?:20\d{6}/[^/]+/c\.html|20\d{2}-\d{2}/\d{2}/c_\d+\.htm)$", path))


def discovery_url(source: str, url: str) -> bool:
    path = urlsplit(url).path.lower()
    if "sitemap" in path or path.endswith(".xml") or "feed" in path:
        return True
    if source == "factcheck":
        return bool(re.match(r"^/(?:20\d{2}/\d{2}/(?:page/\d+/)?|page/\d+/)?$", path))
    if source == "snopes":
        return bool(re.match(r"^/fact-check/(?:page/\d+/)?$", path))
    return path in {"/", "/index.htm", "/index.html", "/ld.htm", "/jj.htm", "/rm.htm", "/rm/ndbd.htm", "/jrpy/"} or bool(re.search(r"/(?:node|list)_\d+(?:_\d+)?\.html?$", path))


def default_seeds(source: str, start: date, end: date) -> list[str]:
    if source == "piyao":
        
        
        return ["https://www.piyao.org.cn/rm/ndbd.htm", SOURCES[source]["home"], "https://www.piyao.org.cn/jrpy/"]
    if source != "factcheck":
        return [SOURCES[source]["home"]]
    
    months = []
    current = date(start.year, start.month, 1)
    while current <= end:
        months.append(f"https://www.factcheck.org/{current.year}/{current.month:02d}/")
        current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)
    return list(reversed(months))


def normalize_date(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    
    match = re.match(r"^(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})", value)
    if match:
        try:
            return date(*map(int, match.groups())).isoformat()
        except ValueError:
            return None
    try:
        return parsedate_to_datetime(value).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        pass
    for pattern in ("%B %d, %Y", "%b %d, %Y", "%Y%m%d"):
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            pass
    return None


def _objects(value: Any) -> Iterator[dict]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _objects(child)


def _is_type(obj: dict, kind: str) -> bool:
    types = obj.get("@type", [])
    if isinstance(types, str):
        types = [types]
    if not isinstance(types, list):
        return False
    return any(isinstance(value, str) and value.rsplit("/", 1)[-1] == kind for value in types)


def extract_record(html: str, source: str, url: str, retrieved_at: str | None = None) -> dict:
    """Extract source fields; missing claim/verdict stay missing, never guessed."""
    document = Document(html)
    nodes = list(document.root.walk())
    metadata = {}
    meta = {node.attrs.get("property", node.attrs.get("name", "")).lower(): node.attrs.get("content", "")
            for node in nodes if node.tag == "meta"}
    objects = []
    invalid_jsonld = 0
    for node in nodes:
        if node.tag == "script" and node.attrs.get("type", "").lower() == "application/ld+json":
            try:
                objects.extend(_objects(json.loads(node.text(include_script=True))))
            except (ValueError, TypeError):
                invalid_jsonld += 1
    articles = [obj for obj in objects if any(_is_type(obj, kind) for kind in ("NewsArticle", "Article", "BlogPosting"))]
    reviews = [obj for obj in objects if _is_type(obj, "ClaimReview")]
    headline = next((obj.get("headline") for obj in articles if isinstance(obj.get("headline"), str)), "")
    h1 = next((clean(node.text()) for node in nodes if node.tag == "h1"), "")
    title = h1 or headline or meta.get("article:title") or meta.get("og:title") or next((clean(node.text()) for node in nodes if node.tag == "title"), "")
    body = next((obj["articleBody"] for obj in articles if isinstance(obj.get("articleBody"), str)), "")
    body_method = "jsonld.articleBody"
    if not body:
        class_markers = {"entry-content", "article-body", "article-content", "articleBody", "rich-text", "main-content", "main-article"}
        id_markers = {"detail", "detailContent", "content", "p-detail", "article", "article-content", "zoom"}
        candidates = [node for node in nodes if (class_markers.intersection(node.attrs.get("class", "").split())
                      or node.attrs.get("id") in id_markers or node.attrs.get("itemprop") == "articleBody")]
        if not candidates:
            candidates = [node for node in nodes if node.tag == "article"]
        if not candidates:
            candidates = [node for node in nodes if node.tag == "main"]
        if candidates:
            body = max((clean(node.text()) for node in candidates), key=len)
            body_method = "html.article_container"
        else:
            
            body = "\n".join(clean(node.text()) for node in nodes if node.tag == "p")
            body_method = "html.paragraph_fallback"
    published_at, date_method = None, None
    date_values = [(obj.get("datePublished"), "jsonld.datePublished") for obj in articles + reviews]
    date_values.extend((meta.get(key), "meta." + key) for key in
                       ("article:published_time", "article:publish_time", "datepublished", "publishdate", "pubdate", "date", "publish_date", "og:release_date"))
    date_values.extend((node.attrs.get("datetime") or clean(node.text()), "html.time") for node in nodes
                       if node.tag == "time" and "modified" not in node.attrs.get("itemprop", "").lower()
                       and "updated" not in node.attrs.get("class", "").lower())
    
    date_values.extend((clean(node.text()), "html.publication_element") for node in nodes
                       if set(node.attrs.get("class", "").lower().split()).intersection({"pubtime", "publish-time", "pub-time"})
                       or node.attrs.get("id", "").lower() in {"pubtime", "publish-time"})
    for value, method in date_values:
        parsed = normalize_date(value)
        if parsed:
            published_at, date_method = parsed, method
            break
    metadata.update({"extraction_method": body_method, "publication_date_method": date_method,
                     "date_status": "known" if published_at else "unknown", "invalid_jsonld_blocks": invalid_jsonld})
    if reviews:
        metadata["claim_reviews"] = [{key: obj[key] for key in ("claimReviewed", "reviewRating", "datePublished", "url", "reviewBody") if key in obj}
                                     for obj in reviews]
    record = {"id": "raw_" + hashlib.sha256((source + ":" + canonical_url(url)).encode()).hexdigest()[:20],
              "source": source, "url": canonical_url(url), "title": clean(title), "language": SOURCES[source]["language"],
              "published_at": published_at, "retrieved_at": retrieved_at or datetime.now(timezone.utc).isoformat(),
              "text": clean(body), "metadata": metadata}
    if reviews:
        review = reviews[0]
        if isinstance(review.get("claimReviewed"), str):
            record["claim"] = clean(review["claimReviewed"])
        rating = review.get("reviewRating")
        if isinstance(rating, dict):
            label = rating.get("alternateName") or rating.get("name")
            if isinstance(label, str):
                record["source_verdict"] = label
        if isinstance(review.get("reviewBody"), str):
            record["evidence"] = clean(review["reviewBody"])
    return record


def discover_links(text: str, base_url: str, source: str) -> list[str]:
    """Same-publisher article/listing URLs from HTML, XML sitemaps, RSS, Atom."""
    links = []
    if text.lstrip().startswith("<?xml") or re.match(r"\s*<(?:urlset|sitemapindex|rss|feed)\b", text):
        try:
            root = ElementTree.fromstring(text)
            for node in root.iter():
                tag = node.tag.rsplit("}", 1)[-1]
                if tag in {"loc", "link"}:
                    links.append(node.attrib.get("href") or node.text or "")
        except ElementTree.ParseError:
            return []
    else:
        document = Document(text)
        links = [node.attrs["href"] for node in document.root.walk() if node.tag in {"a", "link"} and "href" in node.attrs]
    result = []
    seen = set()
    for href in links:
        target = canonical_url(urljoin(base_url, href.strip()))
        if urlsplit(target).scheme not in {"http", "https"} or source_for_url(target) != source:
            continue
        if target not in seen and (article_url(source, target) or discovery_url(source, target)):
            result.append(target)
            seen.add(target)
    return result
