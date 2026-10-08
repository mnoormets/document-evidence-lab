import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from lab.api import app
from lab.search import SearchIndex
from lab.extract import extract
ROOT=Path(__file__).resolve().parents[1]

def test_every_synthetic_field_and_evidence_span():
    docs=json.loads((ROOT/'fixtures/documents.json').read_text(encoding='utf-8'))
    for doc in docs:
        result=extract(doc['text'])
        assert result['fields']==doc['expected_fields']
        for item in result['evidence'].values():
            assert doc['text'][item['start']:item['end']]==item['quote']

@pytest.mark.parametrize('text',['Amount: 1.00 EUR\nAmount: 2.00 EUR','Amount: 1.00 EUR\nAmount: 1.00 EUR'])
def test_duplicate_amount_abstains(text):
    result=extract(text)
    assert result['fields']['amount'] is None
    assert 'amount: ambiguous' in result['warnings']

def test_invalid_date_and_missing_amount_abstain():
    result=extract('Due date: 2026-02-30')
    assert result['fields']['due_date'] is None and 'due_date: invalid date' in result['warnings']
    assert result['fields']['amount'] is None

def test_empty_query_and_empty_corpus():
    assert SearchIndex([]).search('payment')==[]
    assert SearchIndex([{'id':'a','text':'Payment webhook'}]).search(' !!! ')==[]

def test_character_search_can_recover_inflected_word():
    index=SearchIndex([{'id':'a','text':'Smartpost'}])
    assert index.search('Smartposti','bm25')==[]
    assert index.search('Smartposti','hybrid')[0]['document_id']=='a'

def test_unknown_query_abstains_and_duplicate_ids_rejected():
    assert SearchIndex([{'id':'a','text':'payment'}]).search('zzzzzzzz')==[]
    with pytest.raises(ValueError):SearchIndex([{'id':'a','text':'x'},{'id':'a','text':'y'}])

def test_api_validation_and_document_lookup():
    with TestClient(app) as c:
        assert c.get('/api/search?q=Smartpost').status_code==200
        assert c.get('/api/search?q=x&mode=neural').status_code==422
        assert c.get('/api/documents/missing').status_code==404
        assert c.post('/api/extract',json={'text':'Amount: 4.50 EUR','customer_email':'x'}).status_code==422
        result=c.post('/api/extract',json={'text':'Amount: 4.50 EUR'}).json()
        assert result['fields']['amount']['value']=='4.50'
