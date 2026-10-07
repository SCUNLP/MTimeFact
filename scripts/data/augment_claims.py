#!/usr/bin/env python3
"""Multi-strategy deterministic augmentation for structured piyao claims.

Nine perturbation strategies, each editing one span of the claim text:

  numeric_shift      8人 → 18人            quantities (unit preserved, dates untouched)
  place_swap         绵阳市 → 德阳市       administrative place names (same suffix, corpus pool)
  date_shift         3月30日 → 3月23日      explicit dates (format preserved)
  quantifier_swap    全部 → 部分            scope quantifiers
  org_swap           广州铁路局 → …        organization names (same suffix, corpus pool)
  severity_swap      严重 → 轻微            severity/scale adjectives
  temporal_hedge_swap近日 → 近期            vague temporal hedges (changes the temporal scope)
  framing_paraphrase 网传 → 有消息称        attribution framing templates
  person_swap        孙某某 → 李某某        anonymous-name surnames

Variants change one to three spans; the refuting evidence and timestamps stay
inherited from the parent record (recorded explicitly).  Synthetic records are
separable via ``label_origin``/``augmentation``.  Replacement values are
deterministic given ``--seed``; ``--repeats`` draws additional alternative
replacements per span, multiplying diversity.

Usage:
  python3 scripts/data/augment_claims.py \
      --input data/claims_structured.jsonl --output data/claims_augmented.jsonl \
      --target 10000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import date, timedelta
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crawl_piyao_today as cpt  # noqa: E402

PLACE_RE = re.compile(r"([\u4e00-\u9fa5]{2,8}(?:省|市|县|区|州|盟|旗))")
ORG_RE = re.compile(
    r"([\u4e00-\u9fa5]{2,10}(?:公司|集团|大学|学院|医院|学校|银行|协会|研究院|研究所|"
    r"机场|车站|铁路局|法院|检察院|工作室|俱乐部|基金会|事务中心))")
PLACE_BAD_TAIL = ("地区", "小区", "小镇", "城镇", "乡村", "农村", "新村", "景区", "园区",
                  "城区", "郊区", "市区", "灾区", "山区", "辖区", "库区", "牧区", "矿区",
                  "校区", "街区", "社区", "县区", "村镇", "城区", "九区", "某区", "各区",
                  "上市", "都市", "城市", "集市", "夜市", "超市", "股市", "楼市", "车市",
                  "菜市", "早市", "全市", "全省", "降价", "涨价")
PLACE_BAD_HEAD = ("该", "某", "这", "那", "多", "全", "各", "部", "一", "些", "两", "几")
PLACE_BAD_SUB = ("部分", "一些", "某些", "广大", "各地", "全国", "多家", "多个", "少数",
                 "首个", "市场", "上市", "都市", "超市", "股市", "楼市", "集市", "车市",
                 "夜市", "菜市", "早市", "节约", "节省", "省份")
PLACE_BAD_CHARS = set("着待援救可以节能会要把被让给的了是在有定格")
PROVINCES = {"河北省", "山西省", "辽宁省", "吉林省", "黑龙江省", "江苏省", "浙江省", "安徽省",
             "福建省", "江西省", "山东省", "河南省", "湖北省", "湖南省", "广东省", "海南省",
             "四川省", "贵州省", "云南省", "陕西省", "甘肃省", "青海省", "台湾省"}
NUMBER_RE = re.compile(r"(?<![\d.])(\d{1,6}(?:\.\d+)?)(万|亿|千|百|人|岁|元|块|个|名|次|天|周|斤|公斤|米|公里|%|％)?")
QUANTIFIER_PAIRS = (
    ("全部", "部分"), ("所有", "个别"), ("大多数", "少数"), ("大部分", "小部分"),
    ("多数", "少数"), ("经常", "偶尔"), ("总是", "偶尔"), ("从来", "偶尔"),
    ("完全", "基本"), ("彻底", "部分"), ("永远", "暂时"),
)
QUANTIFIER_GUARDS = {"少数": ("民族", "名族"), "多数": ("民族",)}
SEVERITY_PAIRS = (
    ("严重", "轻微"), ("大量", "少量"), ("重大", "一般"), ("急剧", "缓慢"),
    ("持续", "短暂"), ("巨大", "微小"), ("紧急", "常规"), ("猛增", "略涨"),
    ("骤降", "微降"), ("成群", "零星"),
)
TEMPORAL_HEDGE_MAP = {"近日": "近期", "日前": "本周", "近期": "日前", "目前": "如今",
                      "当天": "次日", "昨日": "前日", "今天": "昨天", "最近": "日前"}
FRAMING_MAP = {"网传": "有消息称", "有网民称": "有网友称", "网友称": "有网民称", "据传": "网传"}
SYNONYM_PAIRS = (
    ("发生", "出现"), ("死亡", "身亡"), ("受伤", "负伤"), ("疑似", "疑为"),
    ("声称", "宣称"), ("系谣言", "是谣言"), ("不实", "虚假"), ("辟谣", "澄清"),
    ("导致", "引发"), ("查处", "处置"), ("流传", "传播"), ("涉及", "牵涉"),
)
UNIT_RESTYLE_PAIRS = (
    (r"(?<=\d)元", "块钱"), (r"(?<=\d)块(?!钱)", "元"),
    (r"公里", "千米"), (r"千米", "公里"), (r"公斤", "千克"), (r"千克", "公斤"),
    (r"平方米", "平米"), (r"平米", "平方米"),
)
SURNAMES = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦许何吕张孔曹严华金魏陶姜谢邹柏水窦章苏潘葛范彭鲁韦马苗方俞任袁柳唐罗薛伍余米贝姚孟顾尹江钟卢汪石崔龚程邢裴陆荣翁荀羊甄曲封芮储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭历戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴郁胥能苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍璩桑桂濮牛寿通边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公"
PERSON_MOU_RE = re.compile(r"([\u4e00-\u9fa5])某(?:某|先生|女士)?")


def place_ok(name):
    if len(name) < 3 or len(name) > 9:
        return False
    if name.endswith("省"):  # province names must be the real 23, not "…节省" fragments
        return name in PROVINCES
    if any(name.endswith(bad) for bad in PLACE_BAD_TAIL):
        return False
    if name[0] in PLACE_BAD_HEAD:
        return False
    if any(bad in name for bad in PLACE_BAD_SUB):
        return False
    if any(char in PLACE_BAD_CHARS for char in name):
        return False
    return True


def org_ok(name):
    return 3 <= len(name) <= 12


def gather_pools(records):
    """Corpus pools for places and organizations; only names seen twice are
    usable as swap sources (single occurrences are usually parse noise)."""
    counts = Counter()
    org_counts = Counter()
    for record in records:
        claim = record.get("claim") or ""
        for match in PLACE_RE.finditer(claim):
            name = match.group(1)
            if place_ok(name):
                counts[name] += 1
        for match in ORG_RE.finditer(claim):
            name = match.group(1)
            if org_ok(name):
                org_counts[name] += 1
    pools = {}
    valid = set()
    for name, count in counts.items():
        if count >= 2:
            pools.setdefault(name[-1], set()).add(name)
            valid.add(name)
    org_pools = {}
    org_valid = set()
    for name, count in org_counts.items():
        if count >= 2:
            org_pools.setdefault(name[-3:], set()).add(name)
            org_valid.add(name)
    return ({"by_suffix": {k: sorted(v) for k, v in pools.items() if len(v) >= 2}, "valid": valid},
            {"by_suffix": {k: sorted(v) for k, v in org_pools.items() if len(v) >= 2}, "valid": org_valid})


def shift_amount(value, rng):
    if value < 10:
        step = rng.choice((1, 2, 3))
    elif value < 100:
        step = rng.choice((1, 2, 5, 10))
    elif value < 1000:
        step = rng.choice((10, 20, 50, 100))
    else:
        step = rng.choice((100, 200, 500, 1000))
    return max(1, value + (step if rng.random() < 0.6 else -step))


def numeric_spans(text, pools, rng):
    spans = []
    for match in NUMBER_RE.finditer(text):
        end = match.end()
        if end < len(text) and text[end] in "月日年号":
            continue
        raw = match.group(1)
        if "." in raw:
            after = "%g" % round(max(0.1, float(raw) + rng.choice((-0.5, -0.2, 0.2, 0.5))), 1)
        else:
            after = str(shift_amount(int(raw), rng))
        if after == raw:
            continue
        unit = match.group(2) or ""
        spans.append((match.start(), end, match.group(0), raw + unit, after + unit))
    return spans


def place_spans(text, pools, rng):
    spans = []
    for match in PLACE_RE.finditer(text):
        name = match.group(1)
        if name not in pools["valid"]:
            continue
        pool = [p for p in pools["by_suffix"].get(name[-1], []) if p != name and p not in text]
        if not pool:
            continue
        spans.append((match.start(), match.end(), name, name, rng.choice(pool)))
    return spans


def org_spans(text, org_pools, rng):
    spans = []
    for match in ORG_RE.finditer(text):
        name = match.group(1)
        if name not in org_pools["valid"]:
            continue
        pool = [p for p in org_pools["by_suffix"].get(name[-3:], []) if p != name and p not in text]
        if not pool:
            continue
        spans.append((match.start(), match.end(), name, name, rng.choice(pool)))
    return spans


def date_spans(text, pools, rng):
    spans = []
    for regex in (cpt.CHINESE_DATE_RE, cpt.ISO_DATE_RE, cpt.MONTH_DAY_RE):
        for match in regex.finditer(text):
            original = match.group(0)
            deltas = [-30, -15, -7, -3, -1, 1, 3, 7, 15, 30]
            rng.shuffle(deltas)
            after = None
            for delta in deltas:
                if regex is cpt.CHINESE_DATE_RE:
                    year, month, day = int(match.group("y")), int(match.group("m")), match.group("d")
                    if not day:
                        new_month = month + (1 if delta > 0 else -1)
                        new_month = 12 if new_month < 1 else 1 if new_month > 12 else new_month
                        after = "%d年%d月" % (year, new_month) if new_month != month else None
                    else:
                        try:
                            shifted = date(year, month, int(day)) + timedelta(days=delta)
                            after = "%d年%d月%d日" % (shifted.year, shifted.month, shifted.day)
                        except ValueError:
                            after = None
                elif regex is cpt.ISO_DATE_RE:
                    try:
                        after = (date.fromisoformat(original) + timedelta(days=delta)).isoformat()
                    except ValueError:
                        after = None
                else:
                    try:
                        shifted = date(2024, int(match.group("m")), int(match.group("d"))) + timedelta(days=delta)
                        after = "%d月%d日" % (shifted.month, shifted.day)
                    except ValueError:
                        after = None
                if after and after != original:
                    break
            if after and after != original:
                spans.append((match.start(), match.end(), original, original, after))
    return spans


def _pair_spans(text, pairs, guards):
    spans = []
    for before, after in pairs:
        for match in re.finditer(re.escape(before), text):
            tail = text[match.end():match.end() + 2]
            if any(tail.startswith(g) for g in guards.get(before, ())):
                continue
            spans.append((match.start(), match.end(), before, before, after))
    return spans


def quantifier_spans(text, pools, rng):
    return _pair_spans(text, QUANTIFIER_PAIRS, QUANTIFIER_GUARDS)


def severity_spans(text, pools, rng):
    return _pair_spans(text, SEVERITY_PAIRS, {})


def temporal_spans(text, pools, rng):
    spans = []
    for before, after in TEMPORAL_HEDGE_MAP.items():
        for match in re.finditer(re.escape(before), text):
            spans.append((match.start(), match.end(), before, before, after))
    return spans


def framing_spans(text, pools, rng):
    spans = []
    for before, after in FRAMING_MAP.items():
        for match in re.finditer(re.escape(before), text):
            spans.append((match.start(), match.end(), before, before, after))
    return spans


def person_spans(text, pools, rng):
    spans = []
    for match in PERSON_MOU_RE.finditer(text):
        surname = match.group(1)
        if surname not in SURNAMES:
            continue
        pool = [s for s in SURNAMES if s != surname]
        spans.append((match.start(), match.start() + 1, surname, surname, rng.choice(pool)))
    return spans


def synonym_spans(text, pools, rng):
    return _pair_spans(text, SYNONYM_PAIRS, {})


def unit_spans(text, pools, rng):
    spans = []
    for pattern, after in UNIT_RESTYLE_PAIRS:
        for match in re.finditer(pattern, text):
            spans.append((match.start(), match.end(), match.group(0), match.group(0), after))
    return spans


STRATEGY_ORDER = ("numeric_shift", "place_swap", "date_shift", "quantifier_swap", "org_swap",
                  "severity_swap", "temporal_hedge_swap", "framing_paraphrase", "person_swap",
                  "synonym_swap", "unit_restyle")


def spans_by_strategy(text, pools, org_pools, rng):
    result = {}
    for name in STRATEGY_ORDER:
        if name == "numeric_shift":
            spans = numeric_spans(text, pools, rng)
        elif name == "place_swap":
            spans = place_spans(text, pools, rng)
        elif name == "date_shift":
            spans = date_spans(text, pools, rng)
        elif name == "quantifier_swap":
            spans = quantifier_spans(text, pools, rng)
        elif name == "org_swap":
            spans = org_spans(text, org_pools, rng)
        elif name == "severity_swap":
            spans = severity_spans(text, pools, rng)
        elif name == "temporal_hedge_swap":
            spans = temporal_spans(text, pools, rng)
        elif name == "framing_paraphrase":
            spans = framing_spans(text, pools, rng)
        elif name == "synonym_swap":
            spans = synonym_spans(text, pools, rng)
        elif name == "unit_restyle":
            spans = unit_spans(text, pools, rng)
        else:
            spans = person_spans(text, pools, rng)
        if spans:
            result[name] = spans
    return result


def apply_span(text, span, replacement):
    start, end = span[0], span[1]
    return text[:start] + replacement + text[end:]


def make_variant(record, edits, rng):
    text = record["claim"]
    augmentations = []
    # Apply edits back-to-front so span offsets computed on the original text
    # stay valid for every edit (a front-to-back order would shift them).
    for strategy, span, replacement in sorted(edits, key=lambda edit: edit[1][0], reverse=True):
        text = apply_span(text, span, replacement)
        augmentations.append({"strategy": strategy, "field": "claim",
                              "before": span[3], "after": replacement})
    augmentations.reverse()
    variant_id = "aug_" + hashlib.sha256(
        ("%s:%s" % (record["id"], text)).encode()).hexdigest()[:20]
    variant = dict(record)
    variant.update({
        "id": variant_id,
        "claim": text,
        "claim_dates": cpt.extract_dates(text),
        "label_origin": "perturbed",
        "derived_from": record["id"],
        "augmentation": {"strategies": [a["strategy"] for a in augmentations], "edits": augmentations,
                         "evidence": "inherited_unchanged_from_parent"},
    })
    return variant


def record_variants(record, pools, org_pools, rng, max_per_claim):
    """Single-edit variants first (all spans), then 2- and 3-edit combinations."""
    produced = []
    available = spans_by_strategy(record["claim"], pools, org_pools, rng)
    for name, spans in available.items():
        for span in spans:
            if len(produced) >= max_per_claim:
                return produced
            produced.append(make_variant(record, [(name, span, span[4])], rng))
    names = list(available)
    for k in (2, 3):
        if len(produced) >= max_per_claim:
            break
        for combo in permutations(names, k):
            if len(produced) >= max_per_claim:
                break
            edits = []
            for position, name in enumerate(combo):
                spans = available[name]
                span = spans[0] if position == 0 else spans[-1]
                edits.append((name, span, span[4]))
            produced.append(make_variant(record, edits, rng))
    return produced


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=10000, help="Desired total records (originals + variants)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-claim", type=int, default=10, help="Cap on variants derived from one record per pass")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Extra passes that draw alternative replacement values (multiplies diversity)")
    args = parser.parse_args(argv)
    if args.target < 1 or args.max_per_claim < 1 or args.repeats < 1:
        parser.error("--target/--max-per-claim/--repeats must be positive")
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        parser.error("input has no records")
    for record in records:
        record.setdefault("label_origin", "original")
        record.setdefault("derived_from", None)
        record.setdefault("augmentation", None)
        record.setdefault("label", False)  # variants inherit the parent label
    pools, org_pools = gather_pools(records)
    seen = {re.sub(r"\s+", "", record["claim"]) for record in records}
    variants, per_strategy = [], Counter()
    full = lambda: len(records) + len(variants) >= args.target
    for pass_index in range(args.repeats):
        for record in records:
            if full():
                break
            rng = random.Random("%s:%s:%s" % (args.seed, pass_index, record["id"]))
            for variant in record_variants(record, pools, org_pools, rng, args.max_per_claim):
                key = re.sub(r"\s+", "", variant["claim"])
                if key in seen:
                    continue
                seen.add(key)
                variants.append(variant)
                for edit in variant["augmentation"]["edits"]:
                    per_strategy[edit["strategy"]] += 1
                if full():
                    break
        if full():
            break
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    all_records = records + variants
    with temporary.open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(out)
    shortfall = max(0, args.target - len(all_records))
    report = {
        "input_records": len(records), "variants": len(variants), "total": len(all_records),
        "target": args.target, "shortfall": shortfall,
        "by_label": dict(Counter(record["label"] for record in all_records)),
        "augmented_parents": len({variant["derived_from"] for variant in variants}),
        "variants_by_strategy": dict(per_strategy),
        "strategy_reach": {name: sum(1 for variant in variants
                                     if name in variant["augmentation"]["strategies"]) for name in STRATEGY_ORDER},
        "place_pool_suffixes": {s: len(v) for s, v in pools["by_suffix"].items()},
        "org_pool_suffixes": {s: len(v) for s, v in org_pools["by_suffix"].items()},
        "output": str(out),
        "notes": [
            "Each variant changes one to three spans of the claim text by deterministic rules; the parent claim is kept unchanged.",
            "Evidence and timestamps are inherited unchanged from the parent record (see augmentation.evidence).",
            "Filter real vs synthetic with label_origin (original/perturbed); augmentation.edits shows every applied edit.",
            "Diversity knobs: --repeats (alternative replacement draws) and --max-per-claim (variants per record per pass).",
            "A shortfall means the corpus did not offer enough editable spans for the requested target.",
        ],
    }
    report_path = Path(str(out) + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "total": len(all_records), "variants": len(variants),
                      "augmented_parents": report["augmented_parents"],
                      "by_strategy": dict(per_strategy), "shortfall": shortfall}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
