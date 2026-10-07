#!/usr/bin/env python3
"""High-difficulty synthesis: same pipeline as synthesize_claims.py, but the
prompt forces fine-grained, hard-to-spot perturbations and the path pool is the
tight-window / deep-hop selection from select_hard_paths.py.

Usage:
  python3 scripts/data/synthesize_hard_claims.py --limit-paths 30   # pilot
  python3 scripts/data/synthesize_hard_claims.py                    # full run
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthesize_claims as sc  # noqa: E402

HARD_PROMPT = """把下面 {count} 条时序知识图谱路径转换为**高难度**事实核查数据。

对每条路径产出：
- 2 条正样本 claim：忠实反映路径上全部关系、方向与日期的多跳断言（必须串联多条边才能验证）；像真实流传的说法，句式自然；若路径含日期，把日期信息织入或附加。
- 4 条负样本 claim：从六种扰动中选四种各做一条。**必须是细粒度扰动：表面看完全可信，粗心读者会当成真的，只有逐条比对证据才能发现差异。**
  · entity_substitution：替换为同类型、知名度相近、极易混淆的实体（禁止换成风马牛不相及的实体）；
  · chronological_distortion：只做最小改动——有月份精度时改 1~2 个月，否则改 1 年（禁止改 5 年/10 年这类一眼可见的差异）；
  · relational_corruption：改为语义最接近的关系或轻微改动事实细节（如 spouse→partner、educated_at→graduated_from、执导→联合执导），禁止改成明显荒谬的关系；
  · data_perturbation：数值改动幅度不超过 5%，或做同类量级下最小的合理变动（禁止夸张改动）；
  · contextual_contradiction：引入一个与证据直接冲突但看似合理的前提；
  · polarity_inversion：反转结论，但句式保持自然可信。
  每条只改动一个细节，且必须能被同一路径证据反驳。
- 每条记录附 explanation（引用路径证据说明为何为真/为假）、date（路径中明确的 ISO 日期，无则 null）、time_window（路径证据覆盖的时间窗，**统一精确到月 YYYY-MM**：日期证据截断到月；仅有年份的用该年 01-12 月表示；完全无日期才填 null）。

claim 用中文书写（专有名词可保留英文或通用中文译名），简洁不超过 120 字。
输出 JSON：
{{"paths":[{{"path_index":1,"positives":[{{"claim":"...","explanation":"...","date":null,"time_window":{{"start":"YYYY-MM","end":"YYYY-MM"}}}}],"negatives":[{{"claim":"...","perturbation":"entity_substitution","edited":"改了什么","explanation":"...","date":null,"time_window":{{"start":"YYYY-MM","end":"YYYY-MM"}}}}]}}]}}

路径（含证据事实）：
{items}"""


def main():
    argv = sys.argv[1:]
    if "--paths-dir" not in argv:
        argv += ["--paths-dir", "data/graphs/paths_hard"]
    if "--output" not in argv:
        argv += ["--output", "data/synthesized_hard.jsonl"]
    if "--cache-dir" not in argv:
        argv += ["--cache-dir", ".cache/llm_synth_hard"]
    out_path = argv[argv.index("--output") + 1]
    sc.PROMPT = HARD_PROMPT
    code = sc.main(argv)
    if code != 0:
        return code
    # tag the finished records as the hard set
    out = Path(out_path)
    rows = []
    for line in out.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        r["method"] = "llm-think-high-hard"
        r["difficulty"] = "hard"
        rows.append(r)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out)
    print(json.dumps({"tagged_hard": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
