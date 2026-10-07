
"""Convert finalized synthesis samples into separate gold and prediction-template files.

Temporal retrieval windows are supplied explicitly; never guessed from the
claim's reference date, model output, or min/max dates in the entire graph.
"""


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path
from mtimefact.common import require_gpt5_claim_generator, load_json, read_jsonl, write_json, write_jsonl


def prepare(samples, graph, windows):
    if graph.get("metadata", {}).get("demo"):
        raise ValueError("Offline demo graphs/plans are not finalized benchmark data")
    nodes = {n["id"]:n for n in graph["nodes"]}
    edges = {e["id"]:e for e in graph["edges"]}
    sources = {s["id"]:s for s in graph["sources"]}
    gold, templates = [], []
    seen=set()
    for sample in samples:
        sid=sample["id"]
        if sid in seen:
            raise ValueError("Duplicate sample id: "+sid)
        seen.add(sid)
        if (sample.get("demo") or sample.get("status") != "final" or not sample.get("claim")
                or type(sample.get("label")) is not bool):
            raise ValueError("Expected finalized, audited GPT-5 claims; cannot evaluate draft plans")
        require_gpt5_claim_generator({"llm": {"generator": sample.get("generation", {})}})
        if sample.get("audit", {}).get("valid") is not True:
            raise ValueError("Sample lacks accepted audit provenance")
        window = windows.get(sid, sample.get("time_window"))
        if not isinstance(window,dict) or not all(k in window for k in ('start','end')):
            raise ValueError(f"{sid}: provide an explicit retrieval time_window in --windows; reference_date is not a validity interval")
        from datetime import date
        if date.fromisoformat(window['start']) > date.fromisoformat(window['end']):
            raise ValueError(sid+": reversed retrieval window")
        path_edges=[]
        for e in sample['path']['edges']:
            if e['id'] not in edges:
                raise ValueError(sid+": path edge missing from supplied graph")
            path_edges.append(edges[e['id']])
        source_ids=sorted({s for e in path_edges for s in e['source_ids']} | set(sample.get('source_ids',[])))
        evidence=[{'id':source_id,'text':sources[source_id]['text'],'url':sources[source_id].get('url'),
                   'timestamp':sources[source_id].get('published_at'),'timestamp_kind':'article_publication_date'} for source_id in source_ids]
        gold.append({'id':sid,'claim':sample['claim'],'label':sample['label'],'hop_count':sample['hop_count'],
                     'reference_date':sample['reference_date'],'time_window':window,
                     'evidence':evidence,'reasoning_graph':{'nodes':[nodes[n] for n in sample['entity_ids']], 'edges':path_edges},
                     'source_sample_id':sid})
        templates.append({'id':sid,'queries':[],'explanation':'','predicted_label':None})
    return gold,templates


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--windows',type=Path,required=True,help='JSON object mapping sample IDs to {start,end}; trusted protocol annotations')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    outputs=[args.output_dir/'gold.jsonl',args.output_dir/'predictions.template.jsonl',args.output_dir/'manifest.json']
    if not args.overwrite and any(f.exists() for f in outputs):
        p.error('Output exists; use --overwrite or a new directory')
    gold,predictions=prepare(list(read_jsonl(args.samples)),load_json(args.graph),load_json(args.windows))
    if not gold:
        p.error('No finalized samples')
    write_jsonl(outputs[0],gold);write_jsonl(outputs[1],predictions)
    write_json(outputs[2],{'samples':len(gold),'predictions_status':'empty_template_not_a_model_run',
                           'time_windows':'explicit_trusted_annotations','gold_file':str(outputs[0])})
    print({'samples':len(gold),'gold':str(outputs[0]),'prediction_template':str(outputs[1])})


if __name__=='__main__':
    main()
