"""Fixed held-out query set, reproducible metrics; no inferred achievements."""
import json
import platform
import time
from pathlib import Path
from .search import SearchIndex
from .extract import extract
ROOT=Path(__file__).resolve().parents[1]

def evaluate():
    corpus=json.loads((ROOT/'fixtures/documents.json').read_text(encoding='utf-8'))
    cases=json.loads((ROOT/'fixtures/queries.json').read_text(encoding='utf-8'))
    index=SearchIndex(corpus);summary={}
    for mode in ['bm25','hybrid']:
        hits=0;mrr=0;elapsed=[];details=[]
        for case in cases:
            start=time.perf_counter();result=index.search(case['query'],mode,3);elapsed.append((time.perf_counter()-start)*1000)
            ids=[r['document_id'] for r in result]
            rank=next((i for i,x in enumerate(ids,1) if x in case['relevant_ids']),None)
            hits+=bool(rank);mrr+=1/rank if rank else 0
            details.append({'query':case['query'],'expected':case['relevant_ids'],'retrieved':ids,'first_relevant_rank':rank})
        summary[mode]={'query_count':len(cases),'hit_rate_at_3':hits/len(cases),'mrr_at_3':mrr/len(cases),
            'mean_search_ms':sum(elapsed)/len(elapsed),'cases':details}
    field_correct=0;field_total=0
    for doc in corpus:
        actual=extract(doc['text'])['fields']
        for field,expected in doc['expected_fields'].items():
            field_total+=1;field_correct+=actual[field]==expected
    return {'scope':'Synthetic label-based documents only; no neural embeddings or LLM inference.',
        'python':platform.python_version(),'documents':len(corpus),'retrieval':summary,
        'extraction':{'correct_fields':field_correct,'total_fields':field_total,'exact_field_accuracy':field_correct/field_total}}

def main():
    report=evaluate();target=ROOT/'evaluation.json';target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
