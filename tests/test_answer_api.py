from fastapi.testclient import TestClient
from lab import api

class Service:
    def __init__(self,reason='verified_source_selection'):self.reason=reason
    def answer(self,question,backend):return {'reason':self.reason,'backend':backend,'answer':question}
    def metrics(self):return {'counters':{'requests':1}}
    def _finish(self,result,started):return result

def test_answer_validation_default_and_metrics(monkeypatch):
    monkeypatch.setattr(api,'RAG_SERVICE',Service())
    with TestClient(api.app) as client:
        assert client.post('/api/answer',json={'question':''}).status_code==422
        assert client.post('/api/answer',json={'question':'x'*301}).status_code==422
        assert client.post('/api/answer',json={'question':'x','backend':'cloud'}).status_code==422
        assert client.post('/api/answer',json={'question':'x','api_key':'private'}).status_code==422
        assert client.post('/api/answer',json={'question':'x','backend':'retrieval'}).json()['backend']=='retrieval'
        assert client.get('/api/rag-metrics').json()['counters']['requests']==1

def test_busy_returns_retry_status(monkeypatch):
    monkeypatch.setattr(api,'RAG_SERVICE',Service('model_busy'))
    with TestClient(api.app) as client:
        assert client.post('/api/answer',json={'question':'x','backend':'retrieval'}).status_code==429

def test_unavailable_does_not_fabricate_answer(monkeypatch):
    def fail():raise RuntimeError('missing local model')
    monkeypatch.setattr(api,'get_rag',fail)
    with TestClient(api.app) as client:
        assert client.post('/api/answer',json={'question':'x'}).status_code==503


def test_default_grounded_api_returns_real_field_from_source(monkeypatch):
    from lab.search import SearchIndex
    docs=[{'id':'invoice','text':'Reference: MAPLE-101\nAmount: 15 EUR\n\nMaple subscription.'}]
    fake=Service();fake.retriever=lambda q:SearchIndex(docs).search('Maple')
    monkeypatch.setattr(api,'DOCS',docs)
    monkeypatch.setattr(api,'RAG_SERVICE',fake)
    with TestClient(api.app) as client:
        result=client.post('/api/answer',json={'question':'How much does Maple cost?'})
        assert result.status_code==200 and result.json()['answer']=='Amount: 15 EUR'
        assert result.json()['backend']=='grounded'
