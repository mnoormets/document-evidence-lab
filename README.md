# Document Evidence Lab

AI-assisted personal learning project. Search synthetic Estonian/English documents,
extract explicitly labelled fields into JSON, and inspect exact evidence spans.
No customer records, confidential documents, hosted model calls or training claims.

## Run
Install requirements in a virtual environment. Run `python -m lab.evaluate`, then
`python -m uvicorn lab.api:app --host 127.0.0.1 --port 8771`.
Open http://127.0.0.1:8771/. Run `python -m pytest -q` for verification.

## Methods
- Baseline: BM25 token retrieval, k1=1.2 and b=0.75.
- Comparison: BM25 + cosine character-trigram ranking, reciprocal-rank fusion k=60.
  This is lexical hybrid retrieval, **not dense semantic retrieval or a neural RAG model**.
- Extraction: labelled reference, amount/currency, date. Exact spans accompany results.
  Missing, duplicated or invalid labels abstain instead of making up values.
- Fixed query evaluation: hit rate and reciprocal rank at 3, plus local search time.
  Multiple passages from one document can occupy multiple slots; passage ranking is
  deliberate. Expected document relevance is hand-labelled on a tiny synthetic set.

## Limitations
No PDF/OCR parser, model fine-tuning, semantic embeddings, authentication or persistence.
Local elapsed times are a small-data measurement, not a production throughput claim.
Documents are UTF-8 plain text. Extraction does not understand arbitrary clauses.
For a neural RAG extension, keep this baseline and evaluate new answers/citations on
separate examples. Document text must remain untrusted data, never instructions.

## Learning check
Explain why character grams can tolerate spelling/inflection and why they can also
return irrelevant matches. Explain why two Amount lines should trigger abstention.
Change one test query yourself, predict the ranking, then inspect the result.
