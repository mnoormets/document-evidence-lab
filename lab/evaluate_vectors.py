"""Measured persistent retrieval/reranking comparison; keep failures and timings."""
import json,hashlib,time,statistics
from pathlib import Path
from .semantic import load_index as memory_index
from .vector_search import load_index,close
from .grounded import GroundedQA
ROOT=Path(__file__).resolve().parents[1]
def main():
    fixture=ROOT/'fixtures/qa-review.json';data=json.loads(fixture.read_text(encoding='utf-8'));docs=data['documents']
    memory=memory_index(docs);stored=load_index(docs,rerank=True)
    report={'scope':data['scope']+' Comparison added after fixtures were already reviewed. No independent accuracy claim.',
      'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),'storage':'Qdrant embedded disk persistence; not a running distributed Qdrant server',
      'models':{name:json.loads((ROOT/'data'/file).read_text(encoding='utf-8'))['model']+'@'+json.loads((ROOT/'data'/file).read_text(encoding='utf-8'))['revision'] for name,file in [('encoder','model-manifest.json'),('reranker','reranker-manifest.json')]},
      'policy':{'dense_cosine_threshold':.30,'candidate_limit':12,'result_limit':3,'sparse':'BM25','fusion':'RRF k=60','reranker_threshold':'none; raw logit is not calibrated confidence'},'methods':{}}
    try:
      for mode in ['neural','qdrant','qdrant_rerank']:
        details=[];times=[];correct=0;false_answers=0;hits_at_3=0;answerable=0
        for case in data['cases']:
          start=time.perf_counter();hits=memory.search(case['question'],'neural',3) if mode=='neural' else stored.search(case['question'],mode,3)
          elapsed=(time.perf_counter()-start)*1000;times.append(elapsed)
          result=GroundedQA(docs,lambda q:hits).answer(case['question']);cite=result['citation'];expected=case['expected_document']
          if expected is not None:
            answerable+=1;hits_at_3+=any(h['document_id']==expected for h in hits)
          success=result['status']=='abstained' if expected is None else bool(cite) and cite['document_id']==expected and case['expected_fragment'] in result['answer']
          correct+=bool(success);false_answers+=bool(cite) and not success
          details.append({'question':case['question'],'expected_document':expected,'retrieved':[h['passage_id'] for h in hits],'retrieval_ms':round(elapsed,3),'correct':bool(success),'answer':result})
        report['methods'][mode]={'cases':len(details),'correct_decisions':correct,'false_answers':false_answers,'answerable_cases':answerable,'document_hit_at_3':hits_at_3/answerable,'mean_retrieval_ms':statistics.mean(times),'p95_retrieval_ms':sorted(times)[int(.95*(len(times)-1))],'details':details}
      (ROOT/'vector-evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
      print(json.dumps({k:{x:y for x,y in v.items() if x!='details'} for k,v in report['methods'].items()},indent=2))
    finally:close()
if __name__=='__main__':main()
