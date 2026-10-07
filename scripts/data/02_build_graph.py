
"""Build an audited temporal graph from raw fact-check JSONL."""


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import json
import sys
from pathlib import Path

from mtimefact.data.graph_building import ROOT, build_graph


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw JSONL records from 01_crawl.py")
    parser.add_argument("--output", type=Path, required=True, help="Temporal graph JSON")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "llm")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing graph and report outputs")
    parser.add_argument("--demo", action="store_true", help="Offline fixture-only extraction and audit")
    parser.add_argument("--enrich-wikipedia", action="store_true", help="Retrieve and audit official MediaWiki context (live only)")
    args = parser.parse_args(argv)
    if args.demo and args.enrich_wikipedia:
        parser.error("--demo is offline and cannot use --enrich-wikipedia")
    try:
        report_path = args.output.with_suffix(".report.json")
        if not args.overwrite and (args.output.exists() or report_path.exists()):
            raise ValueError("output graph or report already exists; pass --overwrite to replace it")
        config = json.loads(args.config.read_text(encoding="utf-8"))
        records = []
        for number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("input line %d is not a JSON object" % number)
                records.append(record)
        llm = None
        if not args.demo:
            from mtimefact.common import JsonLLM
            llm = JsonLLM(config, args.cache_dir)
        graph, report = build_graph(records, llm, demo=args.demo, config=config,
                                    cache_dir=args.cache_dir, enrich_wikipedia=args.enrich_wikipedia)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"graph": str(args.output), "report": str(report_path), "nodes": len(graph["nodes"]),
                          "edges": len(graph["edges"]), "accepted_records": report["accepted_records"],
                          "rejected_records": len(report["rejections"]), "demo": args.demo}, ensure_ascii=False))
        return 0 if graph["edges"] else 2
    except (OSError, ValueError, RuntimeError) as exc:
        print("Graph construction failed: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
