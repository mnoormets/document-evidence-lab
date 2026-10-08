# Validation — 8 October 2026

44 automated tests cover retrieval/extraction, strict answer API inputs, busy and
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
