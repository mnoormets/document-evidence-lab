import pytest
np=pytest.importorskip("numpy")
from fastapi.testclient import TestClient
from lab.api import app
from lab.semantic import NeuralIndex

class Encoder:
    def encode(self,texts,**options):
        lookup={'Payment confirmation': [1.,0.], 'Parcel tracking':[0.,1.],
            'cash received':[1.,0.], 'completely unrelated':[-1.,0.]}
        return np.array([lookup[t] for t in texts])

def test_neural_ranking_threshold_and_source_citation():
    index=NeuralIndex([{'id':'payment','text':'Payment confirmation'},{'id':'parcel','text':'Parcel tracking'}],Encoder())
    hits=index.search('cash received')
    assert [h['document_id'] for h in hits]==['payment']
    assert hits[0]['passage_id']=='payment:0' and hits[0]['cosine_similarity']==1
    assert index.search('completely unrelated')==[]
    assert index.search(' !!! ')==[]

def test_neural_hybrid_keeps_evidence_and_rejects_invalid_limit():
    index=NeuralIndex([{'id':'payment','text':'Payment confirmation'},{'id':'parcel','text':'Parcel tracking'}],Encoder())
    assert index.search('cash received','neural_hybrid')[0]['text']=='Payment confirmation'
    with pytest.raises(ValueError):index.search('cash received',k=0)
    assert NeuralIndex([],Encoder()).search('cash received')==[]

def test_model_unavailable_is_explicit_not_a_fake_lexical_result(monkeypatch):
    def unavailable(documents):raise RuntimeError('model missing')
    monkeypatch.setattr('lab.semantic.load_index',unavailable)
    with TestClient(app) as c:
        assert c.get('/api/search?q=payment&mode=neural').status_code==503
        assert c.get('/api/search?q=payment&mode=bm25').status_code==200


def test_changed_corpus_rebuilds_index_instead_of_leaking_previous_documents(monkeypatch):
    from lab import semantic
    first=[{'id':'old','text':'Payment confirmation'}]
    second=[{'id':'new','text':'Parcel tracking'}]
    monkeypatch.setattr(semantic,'_INDEX',NeuralIndex(first,Encoder()))
    monkeypatch.setattr(semantic,'_CORPUS_KEY',None)
    rebuilt=semantic.load_index(second)
    assert [p.document_id for p in rebuilt.passages]==['new']
    assert semantic.load_index(second) is rebuilt
