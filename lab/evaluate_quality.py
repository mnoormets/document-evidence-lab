"""Document-disjoint synthetic review, with frozen references and retained failures."""
import hashlib,json
from pathlib import Path
from .semantic import load_index
from .grounded import GroundedQA
from .rag import EvidenceRAG
ROOT=Path(__file__).resolve().parents[1]

def main():
    fixture=ROOT/'fixtures/qa-review.json';data=json.loads(fixture.read_text(encoding='utf-8'))
    docs=data['documents'];neural=load_index(docs)
    retrieve=lambda q:neural.search(q,'neural',3)
    methods={'retrieval':EvidenceRAG(docs,retrieve),'grounded':GroundedQA(docs,retrieve)}
    report={'scope':data['scope'],'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),
        'retrieval_model':json.loads((ROOT/'data/model-manifest.json').read_text(encoding='utf-8'))['model'],
        'threshold':0.30,'methods':{}}
    for name,service in methods.items():
        details=[];correct=0;valid=0;answered=0;false_answers=0
        for case in data['cases']:
            result=service.answer(case['question'],'retrieval') if name=='retrieval' else service.answer(case['question'])
            cite=result['citation'];expected=case['expected_document']
            if cite:
                answered+=1;source=next(d['text'] for d in docs if d['id']==cite['document_id'])
                valid+=source[cite['start']:cite['end']]==result['answer']
            success=result['status']=='abstained' if expected is None else bool(cite) and cite['document_id']==expected and case['expected_fragment'] in result['answer']
            correct+=bool(success);false_answers+=bool(cite) and not success
            details.append({'case':case,'correct':bool(success),'result':result})
        report['methods'][name]={'cases':len(details),'correct':correct,'answered':answered,'false_answers':false_answers,'valid_citations':valid,'details':details}
    report['release_gate']={'required_correct_fraction':0.9,'required_false_answers':0,
        'passed':report['methods']['grounded']['correct']/len(data['cases'])>=0.9 and report['methods']['grounded']['false_answers']==0,
        'meaning':'Synthetic regression gate. Cases inspected during improvement; not independent held-out accuracy or production certification.'}
    (ROOT/'quality-evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({name:{k:v for k,v in metrics.items() if k!='details'} for name,metrics in report['methods'].items()},indent=2))
    print(json.dumps(report['release_gate'],indent=2))
if __name__=='__main__':main()
