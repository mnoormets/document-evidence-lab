from fastapi.testclient import TestClient
from lab import api

class Service:
    def __init__(self,reason='verified_source_selection'):self.reason=reason
    def answer(self,question,backend):return {'reason':self.reason,'backend':backend,'answer':question}
    def metrics(self):return {'counters':{'requests':1}}

def test_answer_validation_default_and_metrics(monkeypatch):
    monkeypatch.setattr(api,'RAG_SERVICE',Service())
    with TestClient(api.app) as client:
        assert client.post('/api/answer',json={'question':''}).status_code==422
        assert client.post('/api/answer',json={'question':'x'*301}).status_code==422
        assert client.post('/api/answer',json={'question':'x','backend':'cloud'}).status_code==422
        assert client.post('/api/answer',json={'question':'x','api_key':'private'}).status_code==422
        assert client.post('/api/answer',json={'question':'x'}).json()['backend']=='retrieval'
        assert client.get('/api/rag-metrics').json()['counters']['requests']==1

def test_busy_returns_retry_status(monkeypatch):
    monkeypatch.setattr(api,'RAG_SERVICE',Service('model_busy'))
    with TestClient(api.app) as client:
        assert client.post('/api/answer',json={'question':'x'}).status_code==429

def test_unavailable_does_not_fabricate_answer(monkeypatch):
    def fail():raise RuntimeError('missing local model')
    monkeypatch.setattr(api,'get_rag',fail)
    with TestClient(api.app) as client:
        assert client.post('/api/answer',json={'question':'x'}).status_code==503
