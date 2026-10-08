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
No PDF/OCR parser, model fine-tuning, authentication or document upload/persistence.
Optional semantic embeddings are prepared separately as described below.
Local elapsed times are a small-data measurement, not a production throughput claim.
Documents are UTF-8 plain text. Extraction does not understand arbitrary clauses.
Document text remains untrusted data, never instructions.

## Learning check
Explain why character grams can tolerate spelling/inflection and why they can also
return irrelevant matches. Explain why two Amount lines should trigger abstention.
Change one test query yourself, predict the ranking, then inspect the result.

## Local neural retrieval extension
Install requirements-neural.txt after installing CPU PyTorch for your platform.
Run `python -m lab.prepare_model` once to explicitly download public safetensors
weights. Model revision and local cache are saved under ignored data/. No API keys,
paid inference endpoints or remote custom model code are used. Then run
`python -m lab.evaluate_semantic`. The API modes neural and neural_hybrid use the
local model; if it is unavailable, the API returns 503 rather than pretending that
lexical retrieval is neural output.

The tested model is sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(Apache-2.0), revision recorded in semantic-evaluation.json, 384-dimensional vectors.
Neural search uses cosine >= 0.30; the hybrid method combines BM25 and neural ranks
with reciprocal-rank fusion. This threshold is an initial review policy, not a
calibrated confidence score. No model fine-tuning is claimed. The extractive RAG extension is described below.

The 16-case set contains 14 answerable and 2 unrelated queries, including cross-language
questions. Keep failures and unknown-query returns; do not report hit rate as general
accuracy. First model load is slow; measured warm query time excludes initialization.

Sources: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
and https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html.

## Evidence-constrained local RAG
Run `python -m lab.prepare_generator` once, then `python -m lab.evaluate_rag`.
The optional generator is HuggingFaceTB/SmolLM2-135M-Instruct, pinned to
12fd25f77366fa6b3b4b768ec3050bf629380bac. It runs on CPU, without hosted inference.
POST /api/answer accepts question and backend (retrieval or local_llm).
The model selects a numbered quote through a finite JSON token grammar. The server
validates the selection and exact document offsets; final answers are extractive
quotes, never unchecked model prose. Unknown evidence and invalid outputs abstain.
GET /api/rag-metrics exposes aggregate counts and timings without query text.

Quality matters more than adding a model: on eight synthetic questions, the first
retrieved quote got 5/8 cases correct and the small generator got 3/8. Citation
membership passed for all returned answers, but relevance did not. The default is
the retrieval baseline; local_llm is experimental. Full failures remain in
rag-evaluation.json. These results do not establish real-document accuracy.

The request is limited to 300 characters, model input to 1024 tokens, candidates
to 12, and cache to 64 entries with a five-minute TTL. Model calls are serialized;
busy calls return 429. Recognized instruction-injection phrases are quarantined,
but this heuristic is not a comprehensive injection defense. See ARCHITECTURE.md.

## Typed grounding and regression release
The default answer path is now `grounded`: explicit document names scope the corpus,
price/deadline/reference questions use unambiguous validated fields, and prose
questions exclude numeric headers. This fixes price questions that previously
returned cancellation text. Refund questions are not interpreted as due dates.
Run `python -m lab.evaluate_quality` for a 12-document, 26-question review. The
initial typed method got 22/26 right, retained in quality-before-v2.json. After
reviewed fixes it got 26/26: 20 correct answers and six correct abstentions. These
cases were inspected during improvement and are now regression cases, not a blind
held-out accuracy estimate. The first-passage baseline got 7/26 right.

45 behavior/API tests pass. API metrics record grounded requests; the bounded cache
applies to retrieval/local_llm paths, not typed field responses. Explicit names are
matched to document references/leading names; ambiguous multiple-document requests
abstain. Renamed entities, multilingual synonyms and unsupported complex clauses
need separate validation. The tiny generative model is still experimental.

## Reproducible source release
Run `python -m lab.check_release` after tests. It exits nonzero for a broken typed
routing/citation regression and writes release-check.json. This offline command
requires no model download. GitHub Actions installs the core requirements, runs
all behavior tests and the offline gate, then exports reviewable JSON artifacts.
Optional pretrained-model experiments must be run separately; CI does not claim
GPU training, Linux model timings or neural quality that it has not measured.
