"""Create HAND-AUTHORED evaluator test cases, not GPT-5 generated benchmark claims."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent

def save(name,rows):
    (ROOT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')


def graph(hops):
    nodes=[{'id':f'n{i}','label':f'Fictional node {i}','type':'event'} for i in range(hops+1)]
    edges=[{'id':f'e{i}','head':f'n{i}','tail':f'n{i+1}','relation':'subsequent_to','start':'2024-06-15','end':'2024-06-15'} for i in range(hops)]
    return {'nodes':nodes,'edges':edges}


def doc(i,date='2024-06-15',text='Handwritten fictional evidence for evaluator tests.'):
    return {'id':i,'url':f'https://example.invalid/{i}','text':text,'timestamp':date}


gold=[]
for hops in range(2,6):
    gold.append({'id':f'eval-{hops}','fixture':True,'demo':True,'hop_count':hops,'label':hops!=3,
                 'claim':f'Handwritten fictional test claim requiring the supplied {hops}-edge event chain.',
                 'reference_date':'2024-06-30','time_window':{'start':'2024-06-01','end':'2024-06-30'},
                 'reasoning_graph':graph(hops),'evidence':[doc('gold-evidence')],
                 'fixture_notice':'Manually specified test input; not a synthesized benchmark claim or measured model output.'})

p2={'id':'eval-2','fixture':True,'demo':True,'queries':[
    {'id':'q1','text':'What happened on 2024-06-15 in the first part of this fictional chain?','depends_on':[],
     'retrieved_documents':[doc('relevant'),doc('relevant'),doc('old','2023-01-01'),doc('unrelated',text='Unrelated fixture text.'),doc('undated',None)]},
    {'id':'q2','text':'On 2024-06-15, what event followed the answer to q1?','depends_on':['q1'],
     'retrieved_documents':[doc('next'),doc('next-boundary','2024-06-30')]}],
    'explanation':'The supplied event evidence supports the fictional chain and hence this claim.','predicted_label':True}
p3={'id':'eval-3','fixture':True,'demo':True,'queries':[
    {'id':'q1','text':'What was the first event in 2024-06-15?','depends_on':[],'retrieved_documents':[doc('first')]},
    {'id':'q2','text':'What happened one day later?','depends_on':['q1'],'retrieved_documents':[]},
    {'id':'q3','text':'What was the next event?','depends_on':['q2'],'retrieved_documents':[]}],
    'explanation':'The fictional claim is refuted by the evidence.','predicted_label':True}
p4={'id':'eval-4','fixture':True,'demo':True,'queries':[],'explanation':'','predicted_label':None}

def rating(q,kind,correct=True):
    return {'query_id':q,'kind':kind,'correct':correct,'reason':'Hand-authored scoring expectation for this fixture.'}

def query_grade(sr,ratings):
    return {'gold_sufficient':True,'s_dc':1,'s_sr':sr,'temporal_ratings':ratings,'reason':'Hand-authored test judgement; no judge API was called.'}

judgements={
 'eval-2':{
  'queries':[query_grade(1,[rating('q1','absolute'),rating('q2','absolute')])],
  'retrieval':[{'relevance':[{'rank_id':f'q0:r{i}','relevant':rel} for i,rel in [(1,True),(3,True),(4,False),(5,True)]]},
               {'relevance':[{'rank_id':'q1:r1','relevant':True},{'rank_id':'q1:r2','relevant':True}]}],
  'explanation':[{'i_align':1,'reason':'Agreement.'},{'gold_sufficient':True,'sufficiency':1,'factual_correctness':.9,'temporal_plausibility':.8,'reason':'Fixture quality.'}]},
 'eval-3':{
  'queries':[query_grade(.5,[rating('q1','absolute'),rating('q2','relative'),rating('q3','missing',False)])],
  'retrieval':[{'relevance':[{'rank_id':'q0:r1','relevant':True}]}],
  'explanation':[{'i_align':0,'reason':'Explanation refutes but predicted verdict supports.'}]}
}
save('evaluation_gold.jsonl',gold)
save('evaluation_predictions.jsonl',[p2,p3,p4])
(ROOT/'evaluation_judgements.json').write_text(json.dumps(judgements,indent=2),encoding='utf-8')
(ROOT/'evaluation_expected.json').write_text(json.dumps({'queries':(1+.65)/4,'retrieval':(.3+(.2/3))/4,'explanation':.9/4,'verdict':.25},indent=2),encoding='utf-8')
print('Four evaluator test cases: full, partial, empty, and absent model output.')
