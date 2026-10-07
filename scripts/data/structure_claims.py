#!/usr/bin/env python3
"""Structure every crawled piyao article into claim / evidence / date records.

Deterministic, rule-based extraction over the articles.jsonl / claims.jsonl
produced by crawl_piyao_today.py.  Three extraction layers:

  1. labeled_item   — the 谣言/辟谣/误区 headings paired with their 真相/详情/事实
                      corrections (already extracted by the crawler; reused here)
  2. quoted_rumor   — news-style refutation articles: the quoted rumor statement
                      ("有网民发帖称“…”") plus the correction text that follows
  3. title_question — question headlines ("…？") whose body carries refutation
                      cues ("其实都是误区", "系谣言", "不实")

Articles that are pure video/poster posts (no text) or administrative news
without any refutable statement are counted in the report but produce no claim
— they cannot be structured as claim/evidence without inventing content.

Every emitted record carries: claim text, evidence text (with source
attribution parsed from a trailing （来源：…）), the publication timestamp and
every date mentioned in either text (ISO-normalized when complete).

Usage:
  python3 scripts/data/structure_claims.py \
      --dataset piyao_today=data/piyao_today \
      --dataset piyao_columns=data/piyao_columns \
      --output data/claims_structured.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crawl_piyao_today as cpt  # noqa: E402  (reuse date/source helpers)

QUOTE_RE = re.compile(r"[“『「]([^”』」]{6,400})[”』」]")
# cue words that typically introduce a rumor statement, right before the quote
CUE_BEFORE = re.compile(
    r"(称|声称|发布|流传|传言|网传|写道|显示|提到|表示|配文|发帖|帖文|消息|援引|爆料|反映|说|询问|疑问"
    r"|关于|出现|传出|曝出|曝|涉及|题为|名为|消息称)\s*[:：，,]?\s*$")
REFUTE_CUES = ("辟谣", "不实", "谣言", "虚假", "假的", "误读", "不属实", "纯属虚构",
               "澄清", "求证", "真相", "误区", "捏造", "不准确", "并无", "未发生", "系伪")
# reminder/boilerplate sentences trimmed from the end of evidence
TAIL_BOILERPLATE = re.compile(r"(不信谣|不传谣|提醒广大|请广大|共同维护|清朗网络|擦亮双眼)")
# a following item heading ends the correction slice inside multi-topic articles
HEADING_BOUNDARY = re.compile(
    r"(?:^|\n)\s*(谣言|真相|事实|详情|科普|提示|通报|热点|关注|误区|警惕|重要|案例|监制|统筹|来源|文字|设计|执行|编辑|责任编辑|策划|审核)[：:]")
SENT_SPLIT = re.compile(r"(?<=[。！？!?])")


def norm_text(value):
    return re.sub(r"\s+", "", value or "")


def clean_title(title, column):
    """Drop a leading column banner such as '晓真播报｜' from a title."""
    text = (title or "").strip()
    if "｜" in text:
        head, tail = text.split("｜", 1)
        if len(head) <= 24:
            text = tail.strip()
    elif "|" in text:
        head, tail = text.split("|", 1)
        if len(head) <= 24:
            text = tail.strip()
    return text


def quote_candidates(text):
    """Cue-preceded quoted spans only, ranked by length. Max 3.

    Quotes that merely repeat a slogan or a title ("宣传主题是“…”") are not
    rumor statements, so a quote without an introducing cue word is skipped.
    """
    hits = []
    for match in QUOTE_RE.finditer(text):
        quote = match.group(1).strip()
        if len(quote) < 8:
            continue
        before = text[max(0, match.start() - 18):match.start()]
        if not CUE_BEFORE.search(before):
            continue
        hits.append((len(quote), match.start(), quote))
    filtered = []
    for length, pos, quote in hits:
        if any(quote != other and quote in other for _, _, other in hits):
            continue
        filtered.append((length, pos, quote))
    ranked = sorted(filtered, key=lambda item: -item[0])
    return [item[2] for item in ranked[:3]]


def correction_after(text, quote):
    """The correction slice: everything after the quote's own sentence."""
    idx = text.find(quote)
    if idx < 0:
        return None
    tail = text[idx + len(quote):]
    boundary = re.search(r"[。！？!?]", tail)
    if boundary:
        tail = tail[boundary.end():]
    heading = HEADING_BOUNDARY.search(tail)
    if heading:
        tail = tail[:heading.start()]
    tail = tail.strip().lstrip("。；;，, ")
    if len(tail) < 20:
        return None
    sentences = [s for s in SENT_SPLIT.split(tail) if s.strip()]
    kept = []
    for sentence in sentences:
        if TAIL_BOILERPLATE.search(sentence) and kept:
            break
        kept.append(sentence.strip())
    return "".join(kept).strip() or None


def mine_article(article):
    """Return (claims, skip_reason). claims: list of dicts without ids."""
    text = (article.get("text") or "").strip()
    if not text:
        return [], "no_text"
    title = article.get("title") or ""
    combined = title + " " + text
    title_quote = QUOTE_RE.search(title)
    quotes = quote_candidates(text)
    if title_quote and title_quote.group(1).strip() not in quotes:
        quotes = [title_quote.group(1).strip()] + quotes
    refute = any(cue in combined for cue in REFUTE_CUES)
    if not refute:
        return [], "no_refutation_cue"
    claims = []
    for quote in quotes:
        evidence = correction_after(text, quote)
        evidence_scope = "correction_after_quote"
        if not evidence:
            evidence, evidence_scope = text, "full_text"
        evidence, evidence_source = cpt.split_source(evidence)
        confidence = "high" if (evidence and len(evidence) >= 30
                                and any(cue in evidence for cue in REFUTE_CUES)) else "low"
        claims.append({"claim": quote, "evidence": evidence, "evidence_source": evidence_source,
                       "evidence_scope": evidence_scope, "extraction_method": "quoted_rumor",
                       "confidence": confidence, "source_verdict": "false"})
    if not claims:
        question = clean_title(title, article.get("column"))
        if question.endswith(("？", "?")) and 8 <= len(question) <= 90:
            evidence, evidence_scope = text, "full_text"
            evidence, evidence_source = cpt.split_source(evidence)
            claims.append({"claim": question, "evidence": evidence, "evidence_source": evidence_source,
                           "evidence_scope": evidence_scope, "extraction_method": "title_question",
                           "confidence": "high", "source_verdict": "false"})
    if not claims:
        return [], "no_claim_pattern"
    return claims, None


def labeled_records(dataset_dir, dataset_name):
    path = Path(dataset_dir) / "claims.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        claim = json.loads(line)
        claim["dataset"] = dataset_name
        claim["extraction_method"] = "labeled_item"
        claim["evidence_scope"] = claim.get("evidence_from") or "following_correction"
        rows.append(claim)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", required=True,
                        help="NAME=DIR (a crawl output directory containing articles.jsonl/claims.jsonl); repeatable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    datasets = []
    for spec in args.dataset:
        if "=" not in spec:
            parser.error("--dataset must look like NAME=DIR")
        name, directory = spec.split("=", 1)
        datasets.append((name, Path(directory)))
    records, struct_report, seen = [], {}, set()
    for name, directory in datasets:
        articles_path = directory / "articles.jsonl"
        if not articles_path.exists():
            parser.error("missing %s" % articles_path)
        articles = [json.loads(line) for line in articles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        labeled = labeled_records(directory, name)
        labeled_urls = {row["url"] for row in labeled}
        by_column = defaultdict(Counter)
        for article in articles:
            column = (article.get("metadata") or {}).get("column") or "unknown"
            by_column[column]["articles"] += 1
            if article.get("url") in labeled_urls:
                by_column[column]["labeled_articles"] += 1
                continue  # structured claims already exist for this article
            claims, reason = mine_article(article)
            if not claims:
                by_column[column][reason] += 1
                continue
            for claim in claims:
                key = (article["url"], norm_text(claim["claim"]))
                if key in seen:
                    continue
                seen.add(key)
                claim_text = claim["claim"]
                evidence_text = claim.get("evidence") or ""
                records.append({
                    "id": "cclaim_" + hashlib.sha256(
                        ("piyao:%s:%s" % (article["url"], claim_text)).encode()).hexdigest()[:20],
                    "dataset": name, "source": "piyao", "column": column,
                    "url": article["url"], "article_title": article.get("title"),
                    "published_at": article.get("published_at"),
                    "published_time": (article.get("metadata") or {}).get("publish_time"),
                    "claim": claim_text,
                    "claim_dates": cpt.extract_dates(claim_text),
                    "label": False,  # every mined claim is a refuted rumor statement
                    "item_label": None,
                    "evidence": evidence_text or None,
                    "evidence_source": claim.get("evidence_source"),
                    "evidence_dates": cpt.extract_dates(evidence_text),
                    "evidence_scope": claim.get("evidence_scope"),
                    "extraction_method": claim["extraction_method"],
                    "confidence": claim.get("confidence"),
                    "source_verdict": claim.get("source_verdict"),
                })
                by_column[column][claim["extraction_method"]] += 1
        # labeled rows are the authoritative structured claims; append them now
        for row in labeled:
            key = (row["url"], norm_text(row["claim"]))
            if key in seen:
                continue
            seen.add(key)
            records.append({
                "id": row["id"], "dataset": name, "source": "piyao", "column": row.get("column"),
                "url": row["url"], "article_title": row.get("article_title"),
                "published_at": row.get("published_at"), "published_time": row.get("published_time"),
                "claim": row["claim"], "claim_dates": row.get("claim_dates") or cpt.extract_dates(row["claim"]),
                "label": row.get("label") if type(row.get("label")) is bool else False,
                "item_label": row.get("item_label"),
                "evidence": row.get("evidence"), "evidence_source": row.get("evidence_source"),
                "evidence_dates": row.get("evidence_dates") or cpt.extract_dates(row.get("evidence") or ""),
                "evidence_scope": row.get("evidence_from"),
                "extraction_method": "labeled_item", "confidence": "high",
                "source_verdict": row.get("source_verdict"),
            })
            column = row.get("column") or "unknown"
            by_column[column]["labeled_item"] += 1
        struct_report[name] = {column: dict(counts) for column, counts in sorted(by_column.items())}
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(out)
    totals = {
        "records": len(records),
        "by_method": dict(Counter(record["extraction_method"] for record in records)),
        "by_confidence": dict(Counter(record["confidence"] for record in records)),
        "by_label": dict(Counter(record["label"] for record in records)),
        "by_dataset": dict(Counter(record["dataset"] for record in records)),
        "with_evidence": sum(1 for record in records if record["evidence"]),
        "with_evidence_source": sum(1 for record in records if record["evidence_source"]),
        "with_published_at": sum(1 for record in records if record["published_at"]),
        "with_claim_dates": sum(1 for record in records if record["claim_dates"]),
        "with_evidence_dates": sum(1 for record in records if record["evidence_dates"]),
        "by_year": dict(sorted(Counter((record["published_at"] or "")[:4] for record in records).items())),
    }
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "output": str(out),
              "totals": totals, "datasets": struct_report,
              "notes": [
                  "labeled_item reuses the crawler's 谣言/真相 pair extraction; mined methods are deterministic rules, not an LLM.",
                  "no_text articles are video/poster posts (content lives in media); no_refutation_cue articles are administrative news or explainers without a refutable statement.",
                  "claim_dates/evidence_dates list explicit dates in the text; published_at/published_time timestamp every record.",
              ]}
    report_path = Path(str(out) + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "records": totals["records"], "by_method": totals["by_method"],
                      "output": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
