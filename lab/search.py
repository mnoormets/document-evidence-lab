"""BM25 + character n-gram search with reciprocal-rank fusion. No LLM."""
import math
import re
from collections import Counter
from dataclasses import dataclass


def tokens(text):
    return re.findall(r"[^\W_]+",text.casefold(),flags=re.UNICODE)


def grams(text):
    text=" ".join(tokens(text))
    return Counter(text[i:i+3] for i in range(max(0,len(text)-2)))


def cosine(a,b):
    denom=math.sqrt(sum(v*v for v in a.values())*sum(v*v for v in b.values()))
    return sum(v*b.get(k,0) for k,v in a.items())/denom if denom else 0.0

@dataclass(frozen=True)
class Passage:
    document_id: str
    passage_id: str
    text: str

class SearchIndex:
    def __init__(self,documents):
        ids=[d['id'] for d in documents]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate document IDs')
        self.passages=[Passage(d['id'],f"{d['id']}:{i}",part.strip()) for d in documents
            for i,part in enumerate(d['text'].split('\n\n')) if part.strip()]
        self.counts=[Counter(tokens(p.text)) for p in self.passages]
        self.lengths=[sum(c.values()) for c in self.counts]
        self.average=sum(self.lengths)/len(self.lengths) if self.lengths else 1
        self.df=Counter(t for c in self.counts for t in c)
        self.chargrams=[grams(p.text) for p in self.passages]

    def search(self,query,mode='hybrid',k=3):
        if mode not in {'bm25','hybrid'}:raise ValueError('Unknown mode')
        q=tokens(query)
        if not q or not self.passages:return []
        scores=[]
        for counts,length in zip(self.counts,self.lengths):
            score=0
            for term in set(q):
                tf=counts[term]
                if not tf:continue
                df=self.df[term];idf=math.log(1+(len(self.passages)-df+0.5)/(df+0.5))
                score+=idf*(tf*2.2)/(tf+1.2*(0.25+0.75*length/self.average))
            scores.append(score)
        sparse=sorted((i for i,s in enumerate(scores) if s>0),key=lambda i:(-scores[i],self.passages[i].passage_id))
        ranks={i:1/(60+rank) for rank,i in enumerate(sparse,1)}
        if mode=='hybrid':
            similarity=[cosine(grams(query),g) for g in self.chargrams]
            chars=sorted((i for i,s in enumerate(similarity) if s>=0.08),key=lambda i:(-similarity[i],self.passages[i].passage_id))
            for rank,i in enumerate(chars,1):ranks[i]=ranks.get(i,0)+1/(60+rank)
        order=sorted(ranks,key=lambda i:(-ranks[i],self.passages[i].passage_id))[:k]
        return [{'document_id':self.passages[i].document_id,'passage_id':self.passages[i].passage_id,
            'text':self.passages[i].text,'score':round(ranks[i],6)} for i in order]
