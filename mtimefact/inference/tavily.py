"""Tavily search with exact-request snapshots and explicit, vendor-supplied dates.

API reference: https://docs.tavily.com/documentation/api-reference/endpoint/search
The provider's published_date may describe a publication OR a later update. It
is never replaced by the retrieval time or a date guessed from document text.
"""
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from mtimefact.common import load_json, stable_id, write_json


_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?"
    r"(?:[Zz]|[+-]\d{2}:\d{2})\Z"
)
_RFC_DATE = re.compile(
    r"(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), )?\d{1,2} "
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4} "
    r"\d{2}:\d{2}(?::\d{2})? (?:GMT|UTC|UT|[+-]\d{4})\Z"
)


class _NoRedirect(HTTPRedirectHandler):
    """Never forward a bearer credential to a redirected endpoint."""
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def urlopen(request, *, timeout):
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


def _publication_timestamp(value):
    """Normalize only explicit ISO dates/aware times or unambiguous RFC dates."""
    if not isinstance(value, str):
        return None
    try:
        if _DATE.fullmatch(value):
            return date.fromisoformat(value).isoformat()
        if _TIMESTAMP.fullmatch(value):
            normalized = value.replace("t", "T")
            if normalized[-1:] in ("Z", "z"):
                normalized = normalized[:-1] + "+00:00"
            if int(normalized[-2:]) >= 60:
                return None
            parsed = datetime.fromisoformat(normalized)
            return parsed.isoformat() if parsed.utcoffset() is not None else None
        if _RFC_DATE.fullmatch(value) and not value.endswith("-0000"):
            
            
            if re.search(r"[+-]\d{4}\Z", value):
                if int(value[-2:]) >= 60 or int(value[-4:-2]) >= 24:
                    return None
            parsed = parsedate_to_datetime(value)
            return parsed.isoformat() if parsed.utcoffset() is not None else None
    except (ValueError, TypeError, OverflowError):
        pass
    return None


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _valid_url(value, *, endpoint=False):
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in (("https",) if endpoint else ("http", "https"))
                or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or any(c.isspace() for c in value)):
            return False
        parsed.port  
        return not endpoint or not (parsed.query or parsed.fragment)
    except ValueError:
        return False


class TavilyClient:
    """Search once per exact request/context and preserve the returned rank list.

    Config is the full project config. ``retrieval`` supports ``base_url``
    (the API root), ``api_key_env``, ``search_depth``, ``topic``,
    ``max_document_chars``, ``timeout`` and ``retries``. Credentials are read
    only from the named environment variable, never included in cache keys.
    """

    def __init__(self, config, cache_dir=".cache/tavily"):
        spec = config.get("retrieval", {})
        http = config.get("http", {})
        if not isinstance(spec, dict) or not isinstance(http, dict):
            raise ValueError("retrieval and http config must be objects")
        if spec.get("provider", "tavily") != "tavily":
            raise ValueError("retrieval.provider must be tavily")
        base_url = spec.get("base_url", "https://api.tavily.com")
        if not _valid_url(base_url, endpoint=True):
            raise ValueError("Tavily base_url must be an HTTPS API root without credentials or query")
        self.url = base_url.rstrip("/") + "/search"
        self.api_key_env = spec.get("api_key_env", "TAVILY_API_KEY")
        if not isinstance(self.api_key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env):
            raise ValueError("Tavily api_key_env must name an environment variable")
        self.search_depth = spec.get("search_depth", "advanced")
        self.topic = spec.get("topic", "general")
        if self.search_depth not in ("advanced", "basic", "fast", "ultra-fast"):
            raise ValueError("Invalid Tavily search_depth")
        if self.topic not in ("general", "news", "finance"):
            raise ValueError("Invalid Tavily topic")
        self.max_document_chars = _integer(spec.get("max_document_chars", 12000),
                                           "retrieval.max_document_chars", 1, 1000000)
        self.retries = _integer(spec.get("retries", http.get("retries", 3)),
                                "retrieval.retries", 0, 5)
        self.timeout = spec.get("timeout", http.get("timeout", 120))
        if (type(self.timeout) not in (int, float) or not math.isfinite(self.timeout)
                or not 0 < self.timeout <= 300):
            raise ValueError("retrieval.timeout must be greater than 0 and at most 300 seconds")
        self.cache_dir = Path(cache_dir)

    def search(self, query, *, max_results=5, start_date=None, end_date=None,
               request_context=None):
        """Return {documents, metadata}; do not deduplicate, rerank or backfill."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Tavily query must be a nonempty string")
        _integer(max_results, "max_results", 1, 20)
        body = {"query": query, "search_depth": self.search_depth,
                "max_results": max_results, "topic": self.topic,
                "include_answer": False, "include_raw_content": "text",
                "include_published_date": True, "auto_parameters": False}
        for field, value in (("start_date", start_date), ("end_date", end_date)):
            if value is not None:
                if not isinstance(value, str) or not _DATE.fullmatch(value):
                    raise ValueError(f"{field} must be an explicit YYYY-MM-DD date")
                try:
                    date.fromisoformat(value)
                except ValueError:
                    raise ValueError(f"{field} must be a valid YYYY-MM-DD date") from None
                body[field] = value
        if start_date is not None and end_date is not None and start_date > end_date:
            raise ValueError("Tavily date window is reversed")
        
        
        identity = {"schema": "tavily_search_v1", "endpoint": self.url,
                    "request": body, "request_context": request_context,
                    "max_document_chars": self.max_document_chars,
                    "timeout": self.timeout, "retries": self.retries}
        try:
            key = stable_id("", identity)
        except (TypeError, ValueError):
            raise ValueError("Tavily request_context must be JSON serializable") from None
        cached = self.cache_dir / (key + ".json")
        if cached.exists():
            try:
                snapshot = load_json(cached)
                if (not isinstance(snapshot, dict) or snapshot.get("request_hash") != key
                        or not isinstance(snapshot.get("fetched_at"), str)):
                    raise ValueError()
                result = self._normalize(snapshot.get("response"), snapshot["fetched_at"],
                                         max_results, key)
            except (ValueError, OSError):
                raise ValueError("Invalid Tavily cache snapshot") from None
            result["metadata"].update({"request_hash": key, "request": body, "cache_hit": True})
            return result
        secret = os.environ.get(self.api_key_env)
        if not secret:
            raise RuntimeError(f"Missing environment variable {self.api_key_env} for Tavily search")
        raw = self._post(body, secret)
        fetched_at = datetime.now(timezone.utc).isoformat()
        result = self._normalize(raw, fetched_at, max_results, key)
        result["metadata"].update({"request_hash": key, "request": body,
                                   "cache_hit": False})
        write_json(cached, {"schema": "tavily_search_v1", "request_hash": key,
                            "fetched_at": fetched_at, "response": raw})
        return result

    def _normalize(self, raw, fetched_at, max_results, request_hash):
        if not isinstance(raw, dict) or not isinstance(raw.get("results"), list):
            raise ValueError("Tavily response must contain a results array")
        if len(raw["results"]) > max_results:
            raise ValueError("Tavily returned more results than requested")
        documents, truncated_documents, truncated_characters = [], 0, 0
        for rank, row in enumerate(raw["results"], 1):
            if not isinstance(row, dict) or not _valid_url(row.get("url")):
                raise ValueError(f"Tavily result at rank {rank} has an invalid URL or shape")
            if not isinstance(row.get("title"), str):
                raise ValueError(f"Tavily result at rank {rank} has an invalid title")
            for name in ("content", "raw_content"):
                if row.get(name) is not None and not isinstance(row[name], str):
                    raise ValueError(f"Tavily result at rank {rank} has invalid {name}")
            raw_content, content = row.get("raw_content"), row.get("content")
            use_raw = isinstance(raw_content, str) and bool(raw_content.strip())
            text = raw_content if use_raw else content
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Tavily result at rank {rank} has no usable text")
            identity = row.get("id")
            if identity is not None and (not isinstance(identity, str) or not identity.strip()):
                raise ValueError(f"Tavily result at rank {rank} has an invalid id")
            timestamp = _publication_timestamp(row.get("published_date"))
            removed = max(0, len(text) - self.max_document_chars)
            document = {"id": stable_id("tavily_", [request_hash, row["url"], rank]),
                        "url": row["url"], "title": row["title"],
                        "text": text[:self.max_document_chars], "rank": rank,
                        "timestamp": timestamp,
                        "timestamp_kind": "provider_publication_or_update" if timestamp else "unknown",
                        "timestamp_source": "tavily.published_date" if timestamp else None,
                        "published_date_raw": row.get("published_date"),
                        "retrieved_at": fetched_at,
                        "content_kind": "raw_content" if use_raw else "content",
                        "text_original_length": len(text), "text_truncated": removed > 0,
                        "truncated_characters": removed}
            if identity is not None:
                document["provider_result_id"] = identity
            if row.get("score") is not None:
                score = row["score"]
                if type(score) not in (int, float) or not math.isfinite(score):
                    raise ValueError(f"Tavily result at rank {rank} has an invalid score")
                document["score"] = score
            documents.append(document)
            truncated_documents += int(removed > 0)
            truncated_characters += removed
        metadata = {"provider": "tavily", "fetched_at": fetched_at,
                    "returned_results": len(documents), "max_results": max_results,
                    "ranking": "provider_order_with_duplicates_no_backfill",
                    "max_document_chars": self.max_document_chars,
                    "truncated_documents": truncated_documents,
                    "truncated_characters": truncated_characters,
                    "date_semantics": "provider_estimated_publication_or_update",
                    "timestamp_policy": "explicit_published_date_only_no_inference"}
        if isinstance(raw.get("request_id"), str):
            metadata["provider_request_id"] = raw["request_id"]
        return {"documents": documents, "metadata": metadata}

    def _post(self, body, secret):
        headers = {"Authorization": "Bearer " + secret, "Content-Type": "application/json"}
        for attempt in range(self.retries + 1):
            try:
                request = Request(self.url, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                  headers=headers, method="POST")
                with urlopen(request, timeout=self.timeout) as response:
                    try:
                        return json.load(response)
                    except (ValueError, UnicodeError):
                        raise ValueError("Tavily response is not valid JSON") from None
            except HTTPError as exc:
                code = exc.code
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                exc.close()
                if (code not in (408, 429) and not 500 <= code <= 599) or attempt == self.retries:
                    raise RuntimeError(f"Tavily API HTTP {code}; check access or configuration") from None
                delay = min(2 ** attempt, 30)
                if retry_after:
                    try:
                        seconds = float(retry_after)
                        if math.isfinite(seconds):
                            delay = min(max(seconds, 0), 30)
                    except (ValueError, TypeError):
                        pass
            except (URLError, TimeoutError, OSError):
                if attempt == self.retries:
                    raise RuntimeError("Tavily API transport failed after retries") from None
                delay = min(2 ** attempt, 30)
            time.sleep(delay)
