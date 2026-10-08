"""Explicit label extraction with source spans; ambiguous fields abstain."""
import re
from decimal import Decimal

PATTERNS={
    'reference':r'^(?:Viide|Reference):[ \t]*(.+)$',
    'amount':r'^(?:Summa|Amount):[ \t]*(\d+(?:[.,]\d{2})?)[ \t]+(EUR|USD|GBP)[ \t]*$',
    'due_date':r'^(?:Tähtaeg|Due date):[ \t]*(\d{4}-\d{2}-\d{2})[ \t]*$',
}

def extract(text):
    result={};evidence={};warnings=[]
    for field,pattern in PATTERNS.items():
        found=list(re.finditer(pattern,text,flags=re.MULTILINE|re.IGNORECASE))
        if len(found)!=1:
            result[field]=None
            warnings.append(f'{field}: '+('ambiguous' if found else 'missing'))
            continue
        match=found[0]
        if field=='amount':
            result[field]={'value':format(Decimal(match.group(1).replace(',','.')),'.2f'),'currency':match.group(2).upper()}
        elif field=='due_date':
            from datetime import date
            try:result[field]=date.fromisoformat(match.group(1)).isoformat()
            except ValueError:
                result[field]=None;warnings.append('due_date: invalid date');continue
        else:result[field]=match.group(1).strip()
        evidence[field]={'start':match.start(),'end':match.end(),'quote':match.group(0)}
    return {'fields':result,'evidence':evidence,'warnings':warnings}
