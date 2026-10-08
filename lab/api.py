import json
from pathlib import Path
from typing import Literal
from fastapi import FastAPI,HTTPException,Query
from fastapi.responses import FileResponse
from pydantic import BaseModel,ConfigDict,Field
from .search import SearchIndex
from .extract import extract
ROOT=Path(__file__).resolve().parents[1]
DOCS=json.loads((ROOT/'fixtures/documents.json').read_text(encoding='utf-8'))
INDEX=SearchIndex(DOCS)
app=FastAPI(title='Document Evidence Lab',version='0.3.0')
class TextInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    text:str=Field(min_length=1,max_length=20000)
@app.get('/')
def page():return FileResponse(ROOT/'web/index.html')
@app.get('/api/search')
def search(q:str=Query(min_length=1,max_length=300),mode:Literal['bm25','hybrid','neural','neural_hybrid']='hybrid'):
    try:
        if mode.startswith('neural'):
            from .semantic import load_index
            hits=load_index(DOCS).search(q,mode)
        else:hits=INDEX.search(q,mode)
    except (RuntimeError,ImportError,OSError,ValueError):raise HTTPException(503,'Neural model unavailable; choose a lexical mode')
    return {'mode':mode,'results':hits,'notice':'Synthetic documents; retrieved passages are evidence, not generated answers.'}
@app.get('/api/documents/{document_id}')
def document(document_id:str):
    doc=next((x for x in DOCS if x['id']==document_id),None)
    if doc is None:raise HTTPException(404,'Document not found')
    return {'id':doc['id'],'text':doc['text'],**extract(doc['text'])}
@app.post('/api/extract')
def extraction(body:TextInput):return extract(body.text)
@app.get('/api/evaluation')
def evaluation():
    target=ROOT/'evaluation.json'
    if not target.exists():raise HTTPException(404,'Run python -m lab.evaluate first')
    return json.loads(target.read_text(encoding='utf-8'))

@app.get('/api/status')
def status():return {'neural_prepared':(ROOT/'data/model-manifest.json').exists(),'data':'synthetic','mode':'local_cpu'}
@app.get('/api/semantic-evaluation')
def semantic_evaluation():
    target=ROOT/'semantic-evaluation.json'
    if not target.exists():raise HTTPException(404,'Neural evaluation has not run')
    return json.loads(target.read_text(encoding='utf-8'))

# Initialize the optional AI pipeline lazily; never download a model in a request.
import threading
import time
RAG_LOCK=threading.Lock()
RAG_SERVICE=None
class AnswerInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    question:str=Field(min_length=1,max_length=300)
    backend:Literal['grounded','local_llm','retrieval']='grounded'

def get_rag():
    global RAG_SERVICE
    with RAG_LOCK:
        if RAG_SERVICE is None:
            from .semantic import load_index
            from .rag import EvidenceRAG
            from .generator import LocalGenerator
            neural=load_index(DOCS)
            generator=LocalGenerator() if (ROOT/'data/generator-manifest.json').exists() else None
            RAG_SERVICE=EvidenceRAG(DOCS,lambda q:neural.search(q,'neural',3),generator)
        return RAG_SERVICE

@app.post('/api/answer')
def answer(body:AnswerInput):
    try:
        if body.backend=='grounded':
            from .grounded import GroundedQA
            service=get_rag()
            started=time.perf_counter()
            result=GroundedQA(DOCS,service.retriever).answer(body.question)
            result=service._finish(result,started)
        else:result=get_rag().answer(body.question,body.backend)
    except (RuntimeError,ImportError,OSError,ValueError):raise HTTPException(503,'Local AI pipeline unavailable')
    if result['reason']=='model_busy':raise HTTPException(429,'Local model busy; retry later')
    return result

@app.get('/api/rag-metrics')
def rag_metrics():
    return RAG_SERVICE.metrics() if RAG_SERVICE else {'counters':{},'state':'not_initialized'}

@app.get('/api/rag-evaluation')
def rag_evaluation():
    target=ROOT/'rag-evaluation.json'
    if not target.exists():raise HTTPException(404,'RAG evaluation has not run')
    return json.loads(target.read_text(encoding='utf-8'))

@app.get('/api/quality-evaluation')
def quality_evaluation():
    target=ROOT/'quality-evaluation.json'
    if not target.exists():raise HTTPException(404,'Run the quality evaluation first')
    return json.loads(target.read_text(encoding='utf-8'))
