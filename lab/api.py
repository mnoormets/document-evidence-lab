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
app=FastAPI(title='Document Evidence Lab',version='0.1.0')
class TextInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    text:str=Field(min_length=1,max_length=20000)
@app.get('/')
def page():return FileResponse(ROOT/'web/index.html')
@app.get('/api/search')
def search(q:str=Query(min_length=1,max_length=300),mode:Literal['bm25','hybrid']='hybrid'):
    return {'mode':mode,'results':INDEX.search(q,mode),'notice':'Synthetic documents; retrieved passages are evidence, not generated answers.'}
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
