"""Offline release check: typed routing, exact citations and fixture integrity.

Does not measure semantic model quality; model experiments are separate.
"""
import hashlib,json,sys
from pathlib import Path
from .grounded import GroundedQA
ROOT=Path(__file__).resolve().parents[1]

def check():
    fixture=ROOT/'fixtures/qa-review.json';data=json.loads(fixture.read_text(encoding='utf-8'))
    # Explicit names exercise document-scoped deterministic policy without weights.
    service=GroundedQA(data['documents'],lambda question:[])
    results=[]
    for case in data['cases']:
        response=service.answer(case['question']);cite=response['citation'];expected=case['expected_document']
        correct=response['status']=='abstained' if expected is None else bool(cite) and cite['document_id']==expected and case['expected_fragment'] in response['answer']
        exact=True
        if cite:
            text=next(d['text'] for d in data['documents'] if d['id']==cite['document_id'])
            exact=text[cite['start']:cite['end']]==response['answer']
        results.append({'question':case['question'],'correct':bool(correct),'exact_source':exact,'reason':response['reason']})
    return {'scope':'Offline typed document routing and extraction regression; not neural or independent accuracy.',
        'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),'cases':len(results),
        'passed':all(r['correct'] and r['exact_source'] for r in results),'results':results}

def main():
    report=check();(ROOT/'release-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
    if not report['passed']:sys.exit(1)
if __name__=='__main__':main()
