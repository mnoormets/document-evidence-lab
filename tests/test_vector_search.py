import numpy as np
import pytest
from qdrant_client import QdrantClient,models
from lab.vector_search import PersistentIndex

DOCS=[{'id':'payments','text':'Payment confirmation'},{'id':'parcels','text':'Parcel tracking'}]
class Encoder:
    def __init__(self):self.document_calls=0
    def encode(self,texts,**options):
        if len(texts)>1:self.document_calls+=1
        vectors={'Payment confirmation':[1.,0.],'Parcel tracking':[0.8,0.6],'cash received':[1.,0.],'unrelated':[-1.,0.]}
        return np.array([vectors[t] for t in texts])
class Reranker:
    def score(self,q,passages):return [10. if p=='Parcel tracking' else -5. for p in passages]

def test_persistent_restart_reuses_vectors_and_preserves_evidence(tmp_path):
    encoder=Encoder();client=QdrantClient(path=str(tmp_path/'db'))
    index=PersistentIndex(DOCS,encoder,client,'test-encoder-v1');hits=index.search('cash received')
    assert hits[0]['document_id']=='payments' and encoder.document_calls==1
    collection=index.collection;client.close()
    client=QdrantClient(path=str(tmp_path/'db'))
    rebuilt=PersistentIndex(DOCS,encoder,client,'test-encoder-v1')
    assert rebuilt.collection==collection and encoder.document_calls==1
    restored=rebuilt.search('cash received')
    assert [h['passage_id'] for h in restored]==[h['passage_id'] for h in hits]
    assert [h['cosine_similarity'] for h in restored]==pytest.approx([h['cosine_similarity'] for h in hits])
    assert [h['text'] for h in restored]==[h['text'] for h in hits];client.close()

def test_reranker_contract_changes_ranking_without_changing_sources():
    client=QdrantClient(':memory:')
    index=PersistentIndex(DOCS,Encoder(),client,'test',Reranker())
    assert index.search('cash received','qdrant_rerank')[0]['document_id']=='parcels'
    assert index.search('cash received','qdrant_rerank')[0]['text']=='Parcel tracking'
    assert 'fusion_score' in index.search('cash received','qdrant_rerank')[0];client.close()

def test_changed_corpus_or_model_does_not_reuse_old_collection():
    client=QdrantClient(':memory:');encoder=Encoder()
    old=PersistentIndex(DOCS,encoder,client,'v1')
    changed=PersistentIndex([{'id':'new','text':'Parcel tracking'}],encoder,client,'v1')
    model=PersistentIndex(DOCS,encoder,client,'v2')
    assert len({old.collection,changed.collection,model.collection})==3
    assert {h['document_id'] for h in changed.search('cash received')}=={'new'};client.close()

def test_partial_build_is_rebuilt_before_use():
    client=QdrantClient(':memory:');encoder=Encoder();index=PersistentIndex(DOCS,encoder,client,'v1')
    import uuid
    marker=str(uuid.uuid5(uuid.NAMESPACE_URL,index.collection+':ready'))
    client.delete(index.collection,models.PointIdsList(points=[marker]))
    PersistentIndex(DOCS,encoder,client,'v1');assert encoder.document_calls==2;client.close()

def test_no_fake_reranking_invalid_scores_and_limits():
    client=QdrantClient(':memory:');index=PersistentIndex(DOCS,Encoder(),client,'test')
    with pytest.raises(RuntimeError):index.search('cash received','qdrant_rerank')
    assert index.search('unrelated')==[] and index.search('!!!')==[]
    with pytest.raises(ValueError):index.search('cash received',k=0)
    class Bad:
        def score(self,q,p):return [float('nan')]*len(p)
    index.reranker=Bad()
    with pytest.raises(ValueError):index.search('cash received','qdrant_rerank');client.close()

def test_empty_corpus_returns_no_results():
    client=QdrantClient(':memory:');index=PersistentIndex([],Encoder(),client,'v1')
    assert index.search('cash received')==[];client.close()


def test_api_selects_persistent_backend_and_missing_reranker_is_explicit(monkeypatch):
    from fastapi.testclient import TestClient
    from lab import api,vector_search
    class Index:
        def search(self,query,mode):return [{'document_id':'DOC-001','passage_id':'DOC-001:0','text':'canonical fixture','backend':mode}]
    monkeypatch.setattr(vector_search,'load_index',lambda docs,rerank=False:Index())
    with TestClient(api.app) as c:
        response=c.get('/api/search',params={'q':'payment','mode':'qdrant_rerank'})
        assert response.status_code==200 and response.json()['results'][0]['backend']=='qdrant_rerank'
        assert c.get('/api/search',params={'q':'payment','mode':'made_up'}).status_code==422
    def unavailable(docs,rerank=False):raise RuntimeError('missing')
    monkeypatch.setattr(vector_search,'load_index',unavailable)
    with TestClient(api.app) as c:
        assert c.get('/api/search',params={'q':'payment','mode':'qdrant_rerank'}).status_code==503
