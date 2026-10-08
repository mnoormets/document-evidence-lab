# Validation — 8 October 2026

45 automated tests cover retrieval/extraction, strict answer API inputs, busy and
unavailable responses, canonical source membership, exact citation offsets,
malformed selections, recognized injection text, decimal preservation, bounded
cache/TTL and finite JSON token constraints. Mock generators test invariants;
separate recorded experiments execute real pretrained models locally on CPU.

- semantic-evaluation.json: 16 synthetic queries, four retrieval methods. Neural
  retrieval found the expected document first on 14/14 answerable cases and
  abstained on 2/2 unrelated cases. Neural hybrid returned an unrelated result once.
- rag-evaluation.json: eight questions, six synthetic documents. Retrieval baseline
  correct 5/8; local SmolLM2 selector correct 3/8, answering only two questions.
  All returned citations were exact document spans. This does not establish
  relevance, factual truth or sufficient quality for unattended deployment.
- Local CPU timings include per-request retrieval/generation, excluding model setup.
  Model revisions, failure responses and reference fragments are retained.
- No fine-tuning, GPU cluster, real customer data, uptime SLA or production load
  test has been performed. Small overlapping fixture sets are exploratory evidence.

Run python -m pytest -q and python -m lab.evaluate_rag to reproduce. Install a
CPU-compatible PyTorch and requirements-neural.txt for the optional model paths.

## Quality regression
12 additional synthetic documents / 26 cases: first-passage baseline 7 correct,
initial typed path 22 correct, revised typed path 26 correct with six abstentions.
All 20 returned citations match the source exactly; no false answers in this set.
Cases were inspected while improving the policy. This is regression evidence,
not independent generalization accuracy. Before/after case-level outputs retained.
Corpus-change isolation, named-document routing, refund/date intent separation and
default grounded API behavior have specific regression tests.

An offline 26-case typed-routing release command is included and run locally.
Its report explicitly excludes semantic model accuracy; failures exit nonzero.
GitHub Actions exports reports, but hosted CI execution remains unverified until
public repository publication. One regression verifies typed policy metadata is
preserved when aggregate request instrumentation is applied.


## Persistent retrieval checkpoint — 8 October 2026
52 tests pass. Qdrant disk restart reuses existing vectors with the same source
payloads; small cosine floating-point differences after persistence are checked
with numerical tolerance. Corpus/model signatures isolate collections, and a
build-complete marker prevents use of incomplete ingestion.
Actual compact cross-encoder and encoder ran locally against 26 reviewed cases.
The new reranker changed rankings but did not improve decision accuracy on this
set; it increased latency. See vector-evaluation.json. BGE support is prepared
but BGE inference itself was not executed.

## Browser verification
An isolated headless browser selected Qdrant plus reranker, waited for the actual search response and asked for the subscription price through the same backend. The displayed answer contained 29.00 USD with its source. See vector-ui-report.json and tests/ui_vectors.cjs. Retrieval benchmark timings exclude initial model loading.
