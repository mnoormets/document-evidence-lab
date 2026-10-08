# Validation — 8 October 2026

28 automated tests cover retrieval/extraction, strict answer API inputs, busy and
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
