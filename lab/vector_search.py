"""Persistent Qdrant retrieval with optional local cross-encoder reranking."""
import hashlib,json,threading,uuid
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient,models
from .search import SearchIndex,tokens
ROOT=Path(__file__).resolve().parents[1]

class LocalReranker:
    def __init__(self,manifest):
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        torch.set_num_threads(2)
        self.tokenizer=AutoTokenizer.from_pretrained(manifest['snapshot_path'],local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForSequenceClassification.from_pretrained(manifest['snapshot_path'],local_files_only=True,trust_remote_code=False,use_safetensors=True).eval()
        self.lock=threading.Lock();self.identity=manifest['model']+'@'+manifest['revision']
    def score(self,query,passages):
        import torch
        values=[]
        with self.lock,torch.inference_mode():
            for start in range(0,len(passages),4):
                batch=passages[start:start+4]
                inputs=self.tokenizer([query]*len(batch),batch,padding=True,truncation=True,max_length=512,return_tensors='pt')
                values.extend(self.model(**inputs).logits.reshape(-1).float().tolist())
        return values

class PersistentIndex:
    def __init__(self,documents,encoder,client,model_identity,reranker=None):
        self.lexical=SearchIndex(documents);self.passages=self.lexical.passages
        self.encoder=encoder;self.client=client;self.reranker=reranker
        signature=json.dumps({'documents':documents,'encoder':model_identity,'chunking':'paragraph-v1'},sort_keys=True,ensure_ascii=False)
        self.corpus_hash=hashlib.sha256(signature.encode()).hexdigest()
        self.collection='evidence_'+self.corpus_hash[:32]
        self.lookup={p.passage_id:p for p in self.passages}
        # A build-complete marker ensures an interrupted partial upsert is not reused.
        if not client.collection_exists(self.collection):
            self._build()
        else:
            complete=client.retrieve(self.collection,[str(uuid.uuid5(uuid.NAMESPACE_URL,self.collection+':ready'))],with_payload=True)
            if not complete or complete[0].payload.get('count')!=len(self.passages):
                client.delete_collection(self.collection);self._build()
    def _build(self):
        if not self.passages:
            # Empty corpora have a schema but contain no search results.
            dim=1;vectors=[]
        else:
            vectors=np.asarray(self.encoder.encode([p.text for p in self.passages],normalize_embeddings=True,convert_to_numpy=True))
            if vectors.ndim!=2 or vectors.shape[0]!=len(self.passages) or not np.isfinite(vectors).all():raise ValueError('Invalid document embeddings')
            dim=vectors.shape[1]
        self.client.create_collection(self.collection,vectors_config=models.VectorParams(size=dim,distance=models.Distance.COSINE))
        points=[models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL,self.collection+':'+p.passage_id)),vector=v.tolist(),payload={'document_id':p.document_id,'passage_id':p.passage_id,'text':p.text,'kind':'passage'}) for p,v in zip(self.passages,vectors)]
        for start in range(0,len(points),64):self.client.upsert(self.collection,points[start:start+64],wait=True)
        self.client.upsert(self.collection,[models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL,self.collection+':ready')),vector=[0.]*dim,payload={'kind':'ready','count':len(self.passages)})],wait=True)
    def search(self,query,mode='qdrant',k=3,candidates=12):
        if mode not in {'qdrant','qdrant_rerank'}:raise ValueError('Unknown persistent mode')
        if not 1<=k<=100 or not k<=candidates<=100:raise ValueError('Invalid candidate limits')
        if not tokens(query) or not self.passages:return []
        if mode=='qdrant_rerank' and self.reranker is None:raise RuntimeError('Reranker not prepared')
        vector=np.asarray(self.encoder.encode([query],normalize_embeddings=True,convert_to_numpy=True))[0]
        if not np.isfinite(vector).all():raise ValueError('Invalid query vector')
        result=self.client.query_points(self.collection,query=vector.tolist(),limit=candidates,score_threshold=.30,query_filter=models.Filter(must=[models.FieldCondition(key='kind',match=models.MatchValue(value='passage'))])).points
        ranked={p.payload['passage_id']:{'document_id':p.payload['document_id'],'passage_id':p.payload['passage_id'],'text':p.payload['text'],'cosine_similarity':float(p.score),'score':1/(60+rank)} for rank,p in enumerate(result,1)}
        # Candidate union: vector retrieval plus sparse BM25, followed by RRF.
        for rank,hit in enumerate(self.lexical.search(query,'bm25',k=candidates),1):
            if hit['passage_id'] in ranked:ranked[hit['passage_id']]['score']+=1/(60+rank)
            else:ranked[hit['passage_id']]={**hit,'cosine_similarity':None,'score':1/(60+rank)}
        hits=sorted(ranked.values(),key=lambda x:(-x['score'],x['passage_id']))[:candidates]
        if mode=='qdrant_rerank' and hits:
            values=np.asarray(self.reranker.score(query,[h['text'] for h in hits])).reshape(-1)
            if len(values)!=len(hits) or not np.isfinite(values).all():raise ValueError('Invalid reranker scores')
            for hit,value in zip(hits,values):hit['fusion_score']=hit['score'];hit['score']=float(value)
            hits.sort(key=lambda h:(-h['score'],h['passage_id']))
        return [{**h,'backend':mode} for h in hits[:k]]

_LOCK=threading.Lock();_CLIENT=None;_INDEXES={};_RERANKER=None;_ENCODER=None

def load_index(documents,rerank=False):
    global _CLIENT,_RERANKER,_ENCODER
    with _LOCK:
        manifest_path=ROOT/'data/model-manifest.json'
        if not manifest_path.exists():raise RuntimeError('Run python -m lab.prepare_model first')
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if _ENCODER is None:
            from . import semantic
            if semantic._INDEX is not None:_ENCODER=semantic._INDEX.encoder
            else:
                from sentence_transformers import SentenceTransformer
                import torch
                torch.set_num_threads(2)
                _ENCODER=SentenceTransformer(manifest['snapshot_path'],device='cpu',local_files_only=True,trust_remote_code=False,model_kwargs={'use_safetensors':True})
        if _CLIENT is None:_CLIENT=QdrantClient(path=str(ROOT/'data/qdrant'))
        if rerank and _RERANKER is None:
            path=ROOT/'data/reranker-manifest.json'
            if not path.exists():raise RuntimeError('Run python -m lab.prepare_reranker first')
            _RERANKER=LocalReranker(json.loads(path.read_text(encoding='utf-8')))
        key=hashlib.sha256(json.dumps(documents,sort_keys=True).encode()).hexdigest()
        if key not in _INDEXES:_INDEXES[key]=PersistentIndex(documents,_ENCODER,_CLIENT,manifest['model']+'@'+manifest['revision'],_RERANKER)
        _INDEXES[key].reranker=_RERANKER
        return _INDEXES[key]

def close():
    global _CLIENT
    with _LOCK:
        if _CLIENT is not None:_CLIENT.close();_CLIENT=None
        _INDEXES.clear()
