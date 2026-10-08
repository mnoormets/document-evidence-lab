# Architecture and engineering decisions

Question -> bounded API -> neural passage retrieval -> canonical corpus check ->
quote candidates -> optional constrained LLM selection -> strict JSON validation ->
exact offset validation -> extractive answer or abstention.

## Why extractive output?
A citation attached to invented prose does not prove grounding. Returning an exact
quote makes source membership directly testable. The trade-off is reduced answer
fluency and incomplete relevance: our measured failures illustrate both.

## Why a small local generator?
This machine has limited available RAM and CPU inference. The 135M English model
keeps experiments inexpensive and reproducible, but its abstention/selection quality
is insufficient. Multilingual retrieval does not make the English generator equally
capable in every language. Bigger models require a fresh measured comparison.

## Failure boundaries
A retrieved passage must match its canonical document ID, passage ID and text.
Recognized malicious instructions are quarantined, not executed. Model selection
is restricted to finite JSON choice paths; malformed or out-of-range choices fail
closed. A false but genuine document can still yield a false answer. Injection
heuristics can miss attacks and reject benign text. No complete guardrail claim.

## Resource and privacy boundaries
Only an explicit preparation command downloads pinned public safetensors files;
requests never download models or run remote custom code. Candidate/input limits,
single-model lock and bounded TTL cache constrain resource usage. Cache keys include
corpus hash, model identity and policy. Metrics omit queries and document text.
The current server is local only, without tenant auth, TLS or durable audit storage.

## Next quality gate
Use independently authored held-out documents, ambiguous and contradicted claims,
more languages and adversarial fixtures. Compare stronger selectors, sentence
reranking and no-model structured extraction; retain failures and cost/latency.
Only promote a method after a frozen evaluation demonstrates improvement. Real
PDF parsing, provenance, access controls and pilot users are separate milestones.

## Revised default policy
Explicit named-document scope precedes optional neural retrieval. Typed field
questions validate the entire named source and extract a single labelled value.
Missing, duplicated or invalid fields abstain. General questions use canonical
prose evidence, excluding numeric header matches. No model prose is trusted.
Neural-index caching is keyed by corpus hash so a different corpus cannot reuse the
previous document index. The default is grounded; retrieval and local_llm remain
comparison methods. The reviewed 26-case release gate is a regression gate only.
