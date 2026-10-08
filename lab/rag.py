"""Citation-grounded extractive RAG with bounded cache and aggregate observability."""
from collections import Counter,OrderedDict,deque
import hashlib
import json
import re
import threading
import time
from uuid import uuid4
from pydantic import BaseModel,ConfigDict,Field
from .search import SearchIndex
from .generator import ModelBusy,InputBudgetExceeded
POLICY='extractive-rag-v1'
INJECTION=re.compile(r'ignore.{0,40}(previous|instructions)|system prompt|reveal.{0,30}(secret|password)|send.{0,30}(credentials|password)|eira.{0,30}(juhis|eelnev)',re.I|re.S)
class Selection(BaseModel):
    model_config=ConfigDict(extra='forbid')
    choice:int=Field(ge=0,le=12,strict=True)

class EvidenceRAG:
    def __init__(self,documents,retriever,generator=None,cache_ttl=300,cache_size=64):
        self.documents={d['id']:d['text'] for d in documents};self.retriever=retriever;self.generator=generator
        self.passages={p.passage_id:p for p in SearchIndex(documents).passages}
        self.corpus_hash=hashlib.sha256(json.dumps(self.documents,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        self.cache_ttl=cache_ttl;self.cache_size=cache_size;self.cache=OrderedDict();self.lock=threading.Lock()
        self.counts=Counter();self.latencies=deque(maxlen=256)

    def candidates(self,hits):
        result=[];quarantined=0
        for hit in hits:
            canonical=self.passages.get(hit.get('passage_id'))
            if not canonical or canonical.document_id!=hit.get('document_id') or canonical.text!=hit.get('text'):continue
            normalized=re.sub(r'[\u200b-\u200f\ufeff]','',canonical.text)
            if INJECTION.search(normalized):quarantined+=1;continue
            document=self.documents[canonical.document_id];base=document.find(canonical.text)
            for match in re.finditer(r'[^\n]+',canonical.text):
                quote=match.group().strip()
                if not 8<=len(quote)<=400:continue
                start=base+match.start()+len(match.group())-len(match.group().lstrip())
                end=start+len(quote)
                if document[start:end]!=quote:continue
                candidate={'document_id':canonical.document_id,'passage_id':canonical.passage_id,'quote':quote,'start':start,'end':end}
                if candidate not in result:result.append(candidate)
                if len(result)==12:return result,quarantined
        return result,quarantined

    def _finish(self,response,start):
        response={**response,'request_id':str(uuid4()),'policy':response.get('policy',POLICY),'corpus_sha256':self.corpus_hash}
        response['timing_ms']={**response.get('timing_ms',{}),'total':round((time.perf_counter()-start)*1000,3)}
        with self.lock:
            self.counts['requests']+=1;self.counts[response['status']]+=1
            if response.get('cache_hit'):self.counts['cache_hits']+=1
            self.counts['reason:'+response['reason']]+=1;self.latencies.append(response['timing_ms']['total'])
        return response

    def answer(self,question,backend='local_llm'):
        if not isinstance(question,str) or not 1<=len(question.strip())<=300:raise ValueError('Question length must be 1-300')
        if backend not in {'local_llm','retrieval'}:raise ValueError('Unknown answer backend')
        question=question.strip();start=time.perf_counter()
        identity=self.generator.identity if self.generator else 'none'
        key=hashlib.sha256(json.dumps([question,backend,POLICY,self.corpus_hash,identity]).encode()).hexdigest()
        with self.lock:
            cached=self.cache.get(key)
            if cached and time.monotonic()-cached[0]<self.cache_ttl:
                self.cache.move_to_end(key)
                result={**cached[1],'cache_hit':True,'timing_ms':{'retrieval':0,'generation':0}}
            else:
                if cached:self.cache.pop(key)
                result=None
        if result is not None:return self._finish(result,start)
        retrieved=self.retriever(question);retrieval_ms=(time.perf_counter()-start)*1000
        candidates,quarantined=self.candidates(retrieved)
        result={'status':'abstained','reason':'no_evidence','answer':None,'citation':None,'cache_hit':False,
            'quarantined_passages':quarantined,'backend':backend,'model':identity,
            'timing_ms':{'retrieval':round(retrieval_ms,3),'generation':0}}
        if candidates:
            choice=1
            if backend=='local_llm':
                if self.generator is None:result['reason']='model_unavailable';return self._finish(result,start)
                generation_start=time.perf_counter()
                try:
                    raw=self.generator.generate(question,candidates)
                    choice=Selection.model_validate_json(raw).choice
                    if choice>len(candidates):raise ValueError('Nonexistent source selection')
                except ModelBusy:
                    result['reason']='model_busy';choice=None
                except InputBudgetExceeded:
                    result['reason']='context_budget_exceeded';choice=None
                except (ValueError,RuntimeError,OSError,TypeError):
                    result['reason']='invalid_or_failed_model_output';choice=None
                finally:result['timing_ms']['generation']=round((time.perf_counter()-generation_start)*1000,3)
                if choice is None:return self._finish(result,start)
            if choice:
                source=candidates[choice-1]
                result.update(status='answered',reason='verified_source_selection',answer=source['quote'],citation=source)
            else:result['reason']='model_abstained'
        with self.lock:
            self.cache[key]=(time.monotonic(),result.copy());self.cache.move_to_end(key)
            while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return self._finish(result,start)

    def metrics(self):
        with self.lock:
            latencies=sorted(self.latencies)
            return {'counters':dict(self.counts),'cache_entries':len(self.cache),'retained_latency_samples':len(latencies),
                'p95_total_ms':latencies[min(len(latencies)-1,int(len(latencies)*.95))] if latencies else None,
                'policy':POLICY,'corpus_sha256':self.corpus_hash,'notice':'Aggregate process metrics; no questions or documents logged.'}
