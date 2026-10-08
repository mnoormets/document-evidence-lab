"""Optional CPU neural retrieval, loaded only from an explicitly prepared local model."""
import json
import threading
from pathlib import Path
from .search import SearchIndex,tokens
ROOT=Path(__file__).resolve().parents[1]
_LOCK=threading.Lock()
_INDEX=None

class NeuralIndex:
    def __init__(self,documents,encoder):
        self.lexical=SearchIndex(documents);self.passages=self.lexical.passages;self.encoder=encoder
        self.embeddings=encoder.encode([p.text for p in self.passages],normalize_embeddings=True,convert_to_numpy=True) if self.passages else None
    def search(self,query,mode='neural',k=3):
        if mode not in {'neural','neural_hybrid'}:raise ValueError('Unknown neural mode')
        if not 1<=k<=1000:raise ValueError('Invalid result limit')
        if not tokens(query) or not self.passages:return []
        import numpy as np
        vector=self.encoder.encode([query],normalize_embeddings=True,convert_to_numpy=True)[0]
        similarities=self.embeddings@vector
        if not np.isfinite(similarities).all():raise ValueError('Invalid embedding scores')
        ranked=sorted((i for i,s in enumerate(similarities) if s>=0.30),key=lambda i:(-float(similarities[i]),self.passages[i].passage_id))
        scores={self.passages[i].passage_id:1/(60+rank) for rank,i in enumerate(ranked,1)}
        if mode=='neural_hybrid':
            for rank,hit in enumerate(self.lexical.search(query,'bm25',k=len(self.passages)),1):
                scores[hit['passage_id']]=scores.get(hit['passage_id'],0)+1/(60+rank)
        chosen=sorted(scores,key=lambda pid:(-scores[pid],pid))[:k]
        lookup={p.passage_id:(i,p) for i,p in enumerate(self.passages)}
        return [{'document_id':lookup[pid][1].document_id,'passage_id':pid,'text':lookup[pid][1].text,
            'score':round(scores[pid],6),'cosine_similarity':round(float(similarities[lookup[pid][0]]),6)} for pid in chosen]


def load_index(documents):
    global _INDEX
    with _LOCK:
        if _INDEX is not None:return _INDEX
        manifest_path=ROOT/'data/model-manifest.json'
        if not manifest_path.exists():raise RuntimeError('Neural model not prepared; run python -m lab.prepare_model')
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(2)
        encoder=SentenceTransformer(manifest['snapshot_path'],device='cpu',local_files_only=True,
            trust_remote_code=False,model_kwargs={'use_safetensors':True})
        _INDEX=NeuralIndex(documents,encoder)
        return _INDEX
