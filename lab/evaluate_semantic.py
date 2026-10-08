import json
import time
from importlib.metadata import version
from pathlib import Path
from .search import SearchIndex
from .semantic import load_index
ROOT=Path(__file__).resolve().parents[1]

def main():
    docs=json.loads((ROOT/'fixtures/documents.json').read_text(encoding='utf-8'))
    cases=json.loads((ROOT/'fixtures/semantic-queries.json').read_text(encoding='utf-8'))
    start=time.perf_counter();neural=load_index(docs);setup=time.perf_counter()-start
    lexical=SearchIndex(docs);results={}
    for mode in ['bm25','hybrid','neural','neural_hybrid']:
        hits=0;rr=0;positive=0;unknown_returns=0;unknown_count=0;elapsed=[];details=[]
        for case in cases:
            start=time.perf_counter()
            found=(neural if mode.startswith('neural') else lexical).search(case['query'],mode,3)
            elapsed.append((time.perf_counter()-start)*1000)
            ids=[f['document_id'] for f in found]
            rank=next((i for i,x in enumerate(ids,1) if x in case['relevant_ids']),None)
            if case['relevant_ids']:
                positive+=1;hits+=bool(rank);rr+=1/rank if rank else 0
            else:
                unknown_count+=1;unknown_returns+=bool(ids)
            details.append({'query':case['query'],'expected':case['relevant_ids'],'retrieved':ids,'first_relevant_rank':rank})
        results[mode]={'answerable_queries':positive,'hit_rate_at_3':hits/positive,'mrr_at_3':rr/positive,
            'unanswerable_queries':unknown_count,'unanswerable_return_rate':unknown_returns/unknown_count,
            'mean_query_ms':sum(elapsed)/len(elapsed),'cases':details}
    manifest=json.loads((ROOT/'data/model-manifest.json').read_text(encoding='utf-8'))
    report={'scope':'Synthetic retrieval comparison only. No generated answers, fine-tuning or real-document accuracy claim.',
        'model':manifest['model'],'model_revision':manifest['revision'],'device':'cpu','embedding_dimensions':int(neural.embeddings.shape[1]),
        'sentence_transformers':version('sentence-transformers'),'torch':version('torch'),'cosine_threshold':0.30,
        'setup_seconds':setup,'methods':results,
        'limitations':'Small authored case set, not an independent general benchmark. Threshold is an initial policy, not calibrated confidence.'}
    (ROOT/'semantic-evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='methods'},indent=2))
    print(json.dumps({k:{field:value for field,value in r.items() if field!='cases'} for k,r in results.items()},indent=2))
if __name__=='__main__':main()
