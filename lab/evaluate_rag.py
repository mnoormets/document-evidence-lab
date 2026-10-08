"""Real local model experiment; per-case failures retained, no fabricated accuracy."""
import json
import time
from pathlib import Path
from .semantic import load_index
from .generator import LocalGenerator
from .rag import EvidenceRAG
ROOT=Path(__file__).resolve().parents[1]

def main():
    docs=json.loads((ROOT/'fixtures/documents.json').read_text(encoding='utf-8'))
    cases=json.loads((ROOT/'fixtures/rag-cases.json').read_text(encoding='utf-8'))
    neural=load_index(docs);generator=LocalGenerator();summary={}
    for backend in ['retrieval','local_llm']:
        service=EvidenceRAG(docs,lambda q:neural.search(q,'neural',3),generator)
        correct=0;answered=0;citation_valid=0;latencies=[];details=[]
        for case in cases:
            result=service.answer(case['question'],backend);latencies.append(result['timing_ms']['total'])
            citation=result['citation']
            if citation:
                answered+=1;source=next(d['text'] for d in docs if d['id']==citation['document_id'])
                citation_valid+=source[citation['start']:citation['end']]==result['answer']
            expected=case['expected_document']
            success=(result['status']=='abstained') if expected is None else (
                bool(citation) and citation['document_id']==expected and case['expected_fragment'] in result['answer'])
            correct+=bool(success)
            details.append({'question':case['question'],'expected_document':expected,'expected_fragment':case['expected_fragment'],
                'correct':bool(success),'result':result})
        summary[backend]={'cases':len(cases),'correct_cases':correct,'answered_cases':answered,'valid_citations':citation_valid,
            'mean_total_ms':sum(latencies)/len(latencies),'p95_total_ms':sorted(latencies)[min(len(latencies)-1,int(len(latencies)*.95))],
            'details':details}
    report={'scope':'Extractive RAG on six synthetic documents. Not a general benchmark or production accuracy claim.',
        'generator':generator.identity,'retrieval_model':json.loads((ROOT/'data/model-manifest.json').read_text())['model'],
        'policy':'extractive-rag-v1','constrained_decoding':'finite JSON choice grammar','device':'cpu','methods':summary,
        'limitations':'Citations prove source membership, not relevance or truth. Injection filter is heuristic, not a security guarantee.'}
    (ROOT/'rag-evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:{field:value for field,value in r.items() if field!='details'} for k,r in summary.items()},indent=2))
if __name__=='__main__':main()
