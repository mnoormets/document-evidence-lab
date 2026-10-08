# Validation - 8 October 2026
Eleven automated tests passed: extraction/source spans, ambiguous amounts,
invalid dates, search edge cases, inflection recovery and API validation.
See evaluation.json for actual retrieval metrics on ten fixed synthetic queries.
No real-document evaluation, embedding model, neural RAG or model training here.
Runtime uses the neighboring Shopify environment on this computer; requirements
allow a separate virtual environment for a fresh checkout.

- Real local neural model download and CPU inference completed at pinned revision.
- semantic-evaluation.json records 16 cases, 4 methods, cosine policy, library versions,
  setup time, warm query timing and every expected/retrieved document ID.
- Neural retrieval found the relevant document first on 14/14 answerable synthetic
  cases, and returned no passage for 2/2 unrelated cases at the initial threshold.
- Neural/BM25 hybrid also returned a passage for one unrelated case: fusion can
  reintroduce lexical false positives. This is recorded, not hidden.
- No fine-tuning, real-document evaluation or generated-answer grounding is tested.
