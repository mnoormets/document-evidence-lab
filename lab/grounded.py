"""Typed source-grounded document QA; no answer is composed from model prose."""
import re
from .extract import extract
from .rag import EvidenceRAG

INTENTS={
    'amount':r'\b(amount|cost|price|summa|maksab|hind)\b|how much|kui palju',
    'due_date':r'\bdue\b|deadline|tähtaeg|maksetäht|millal.{0,60}\bmakse\b',
    'reference':r'\b(reference|viide|viitenumber)\b',
}

def intent(question):
    matches=[field for field,pattern in INTENTS.items() if re.search(pattern,question,re.I)]
    return matches[0] if len(matches)==1 else None

class GroundedQA:
    """Search selects a document; extraction selects an unambiguous exact field.

    General clause questions retain the passage baseline. Ambiguous fields abstain.
    Invoice-only structured answers are not a general-purpose reasoning engine.
    """
    def __init__(self,documents,retriever):
        self.documents={d['id']:d['text'] for d in documents}
        self.base=EvidenceRAG(documents,retriever)
        self.retriever=retriever
        self.anchors={}
        for document_id,text in self.documents.items():
            reference=extract(text)['fields']['reference']
            anchors=set()
            if reference:
                prefix=re.split(r'[-\d]',reference)[0].casefold()
                if len(prefix)>=4:anchors.add(prefix)
            for paragraph in text.split('\n\n')[1:]:
                first=re.match(r'([A-ZÄÖÜÕ][a-zäöüõ]{3,})\b',paragraph)
                if first:anchors.add(first.group(1).casefold())
            self.anchors[document_id]=anchors

    def retrieve_scoped(self,question):
        query_words=set(re.findall(r'[^\W_]+',question.casefold()))
        matched={doc for doc,anchors in self.anchors.items() if anchors & query_words}
        if len(matched)>1:return []
        if matched:
            # Explicit named entities scope the corpus before semantic retrieval.
            # Canonical document passages retain real text and exact provenance.
            doc=next(iter(matched))
            return [{'document_id':p.document_id,'passage_id':p.passage_id,'text':p.text}
                    for p in self.base.passages.values() if p.document_id==doc]
        # Numeric header similarity is not evidence for an unrelated prose query.
        return [h for h in self.retriever(question)
                if h.get('cosine_similarity',1)>=0.40 and '\n' not in h['text']
                and not re.match(r'^(Reference|Viide|Amount|Summa|Due date|Tähtaeg):',h['text'],re.I)]

    def answer(self,question):
        if not isinstance(question,str) or not 1<=len(question.strip())<=300:raise ValueError('Question length must be 1-300')
        field=intent(question)
        scoped=self.retrieve_scoped(question)
        if not field:
            service=EvidenceRAG([{'id':key,'text':value} for key,value in self.documents.items()],lambda q:scoped)
            # For prose questions, labelled headers cannot answer the question.
            clean=[h for h in scoped if not re.match(r'^(Reference|Viide|Amount|Summa|Due date|Tähtaeg):',h['text'],re.I)]
            service.retriever=lambda q:clean
            return service.answer(question,'retrieval')
        if not isinstance(question,str) or not 1<=len(question.strip())<=300:raise ValueError('Question length must be 1-300')
        import time
        from uuid import uuid4
        start=time.perf_counter();hits=scoped;reason='no_evidence';citation=None
        # Never trust a retriever's document ID or text without corpus verification.
        candidates,quarantined=self.base.candidates(hits)
        if candidates:
            document_id=candidates[0]['document_id'];source=self.documents[document_id]
            if self._contains_instruction(source):reason='quarantined_document'
            else:
                data=extract(source);span=data['evidence'].get(field)
                if span:
                    citation={'document_id':document_id,'passage_id':None,**span};reason='verified_typed_field'
                else:reason='field_missing_or_ambiguous'
        return {'status':'answered' if citation else 'abstained','reason':reason,
            'answer':citation['quote'] if citation else None,'citation':citation,
            'backend':'grounded','model':'typed-extraction-with-neural-retrieval',
            'cache_hit':False,'request_id':str(uuid4()),'policy':'typed-grounding-v1',
            'corpus_sha256':self.base.corpus_hash,'quarantined_passages':quarantined,
            'timing_ms':{'total':round((time.perf_counter()-start)*1000,3)}}

    @staticmethod
    def _contains_instruction(text):
        from .rag import INJECTION
        return bool(INJECTION.search(re.sub(r'[\u200b-\u200f\ufeff]','',text)))
