"""Regenerate fictional, non-benchmark fixtures. Run from any directory."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
records = []
for c in range(1, 9):
    specs = [("a", f"Demo {c} Alice", "person"), ("b", f"Demo {c} Orion Club", "organization"),
             ("c", f"Demo {c} Port Vale", "location"), ("d", f"Demo {c} Summit", "event"),
             ("e", f"Demo {c} Proposal", "event"), ("f", str(210+c), "quantity"),
             ("g", f"Demo {c} Morgan", "person")]
    entities = [{"id":i,"label":label,"type":kind,"domain":f"fictional-component-{c}",
                 "disambiguation":f"fictional-demo-{c}-{i}"} for i,label,kind in specs]
    label = {i:l for i,l,_ in specs}
    triples = [("a","member_of","b"),("b","headquartered_in","c"),("c","hosted","d"),
               ("d","endorsed","e"),("e","has_value","f"),("g","is_barred_from","b")]
    wording = {"member_of":"was a member of", "headquartered_in":"was headquartered in", "hosted":"hosted",
               "endorsed":"endorsed", "has_value":"received an exact vote total of", "is_barred_from":"was barred from"}
    facts = []
    sentences = []
    for h,r,t in triples:
        quote = f'On 2024-06-15, {label[h]} {wording[r]} {label[t]}.'
        sentences.append(quote)
        facts.append({"head":h,"relation":r,"tail":t,"start":"2024-06-15","end":"2024-06-15",
                      "time_precision":"day","temporal_kind":"event","evidence_quote":quote,"polarity":"positive"})
    text = "Fictional software test fixture; not a real-world fact-check. " + " ".join(sentences)
    title = f"Fictional temporal fact-check fixture {c}"
    claim = sentences[0]
    raw = {"id":f"demo-source-{c}","source":"factcheck","url":f"https://www.factcheck.org/2024/06/fictional-demo-{c}/",
           "title":title,"language":"en","published_at":"2024-06-16","retrieved_at":"2026-09-29T00:00:00Z",
           "text":text,"claim":claim,"source_verdict":"true","fixture":True,
           "pre_extracted":{"time_sensitive":True,"claim":claim,"source_verdict":"true","entities":entities,"facts":facts},
           "pre_audit":{"accepted":True,"confidence":1.0,"entity_accuracy":True,"temporal_grounding":True,
                        "verdict_consistent":True,"evidence_support":True}}
    records.append(raw)
    structured = {"@context":"https://schema.org","@type":"ClaimReview","claimReviewed":claim,
                  "datePublished":"2024-06-16","url":raw["url"],"headline":title,"reviewBody":text,
                  "reviewRating":{"@type":"Rating","alternateName":"True"}}
    page = '<!doctype html><html><head><title>'+title+'</title><meta property="article:published_time" content="2024-06-16T10:00:00Z">'
    page += '<script type="application/ld+json">'+json.dumps(structured)+'</script></head><body><article><h1>'+title+'</h1><p>'+html.escape(text)+'</p></article></body></html>'
    (ROOT / "html" / f"demo{c}.html").write_text(page, encoding="utf-8")
    (ROOT / "html" / f"demo{c}.json").write_text(json.dumps({"source":"factcheck","url":raw["url"]}), encoding="utf-8")
(ROOT / "raw_demo.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),encoding="utf-8")
print("8 fictional components, 48 evidence edges")
