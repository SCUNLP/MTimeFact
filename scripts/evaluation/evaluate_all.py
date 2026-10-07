
"""Run all four paper metrics, with per-hop reports and explicit judge failure coverage."""


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import csv
import sys
from pathlib import Path
from mtimefact.common import write_json
from mtimefact.evaluation.runner import STAGES, add_arguments, context, evaluate_stage, save_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, all_stages=True)
    args = parser.parse_args()
    try:
        targets = [args.output_dir/(stage+suffix) for stage in STAGES for suffix in (".jsonl", ".jsonl.report.json")]
        targets += [args.output_dir/"summary.json", args.output_dir/"summary.csv"]
        if not args.overwrite and any(p.exists() for p in targets):
            raise ValueError("Evaluation output exists; choose a new directory or use --overwrite")
        config, gold, predictions, fixtures, options = context(args)
        summary = {"demo": args.demo, "n_gold": len(gold), "run_id": args.run_id, "stages": {}}
        csv_rows = []
        for stage in STAGES:
            rows, report = evaluate_stage(stage, gold, predictions, config, options, fixtures)
            save_stage(stage, rows, report, args.output_dir/(stage+".jsonl"), args, config)
            summary["stages"][stage] = {"metric": report["metric"], "overall": report["overall"], "by_hop": report["by_hop"]}
            for group, stats in [("all",report["overall"])] + list(report["by_hop"].items()):
                csv_rows.append({"stage":stage,"metric":report["metric"],"hop_count":group,
                                 **{k:stats[k] for k in ("score","n_gold","n_scored","evaluation_errors","scoring_coverage")}})
            print(stage, report["overall"], flush=True)
        summary["complete"] = all(v["overall"]["complete"] for v in summary["stages"].values())
        summary["note"] = "Four different metrics; do not average them into an unsupported overall score. Demo scores are hand-authored fixture checks only."
        write_json(args.output_dir/"summary.json", summary)
        with (args.output_dir/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
            writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]))
            writer.writeheader(); writer.writerows(csv_rows)
        return 0 if summary["complete"] else 2
    except (OSError, ValueError) as exc:
        print("Evaluation input error: " + str(exc),file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
