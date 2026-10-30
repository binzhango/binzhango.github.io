# Editorial and executable review: ai-108–ai-220

Reviewed on 2026-09-20. This batch contains **113 questions**, all with an individualized brief answer and detailed explanation. Source attribution was retained, and primary references used in the prose were also added to frontmatter. The root implementation agent owns ai-221–ai-270 after workload redistribution.

## Content decisions

- Coding prompts have scoped implementations and explicit assumptions. The absent banking evaluator, UI mockup, messy-code input, infection model and “closest key” definition are not invented: their answers identify the missing specification and provide a clearly described illustrative contract.
- Behavioral answers provide response frameworks without fictional personal accomplishments. Employer motivation in ai-154 is generalized while preserving the disagreement follow-up. Ai-128 now states a standalone concurrency question.
- Open-ended systems, diagnostics and judgment questions use the design/behavioral answer kinds where appropriate.
- The exact bounded-memory top-k premise in ai-188 is corrected: the example returns approximate heavy-hitter candidates with an error bound, not falsely labeled exact top-k results.
- Loss bounds in ai-165 state the loss, class count and argmax assumptions. Ai-197 derives the six-flip expectation. Ai-201 separates an illustrative optimizer-state budget from actual implementation-dependent storage.
- Ai-176 preserves both alternatives from the original ambiguous “in-memory database / Slack” prompt and explains the distinct contracts rather than silently deleting one.

## Verification

Run:

```sh
rtk proxy /tmp/interview-python/bin/python nuclear-nova/scripts/interview/examples-108-220.py
```

The temporary runtime used Python 3.12 with NumPy, PyTorch and pandas. Go and Node executed their respective examples. All **45 published code examples** passed: 40 Python, three SQL, one Go and one HTML/JavaScript example. Five additional arithmetic/gradient checks passed. The machine-readable result is in `example-results.json`.

Meaningful checks include:

- Causal attention invariance, incremental KV-cache/full-prefix equivalence, GQA agreement with PyTorch scaled-dot-product attention and finite gradients.
- TTL boundaries, LRU recency, concurrent token-bucket spending, snapshot compaction and transactional rollback.
- SSE parsing at every byte split, multibyte UTF-8, CR/LF/CRLF, comments, BOM, multiline data and incomplete final events.
- Bounded async concurrency, permanent-failure isolation, transient retry and propagation of durable-storage failure.
- SQL co-purchase deduplication, nested and adjacent date ranges, open-ended intervals, future salaries, pay cuts and deterministic ties.
- Exhaustive small-input comparisons for product-except-self, minimum-window substring and regex matching; tree round trips and bidirectional circular linkage.
- HTML success, empty and error states plus literal rendering of injection-shaped text. This is a DOM-adapter test, not screenshot or accessibility verification.
- Cross-entropy gradients against finite differences and the stated numerical memory calculations.

No browser UI, application server or shared application tests were operated by this batch. The root agent owns full-site build and visual acceptance.

## Limits and follow-up relationships

The snippets are educational cores with production limits explained beside them. They do not claim distributed failure guarantees, live-provider integration or fidelity to absent interview artifacts. No external API requests or real customer data were used in tests.

Potential overlaps were reported to the root for reconciliation, without deleting files: ai-082/139 (Constitutional AI with limitations), ai-054/180 (consequential actions and broader safeguards), ai-098/172 (scale-specific versus general enterprise RAG), ai-016/134 (forward pass versus tensor/sharding manipulation), ai-115/127 (generic worker pool versus 50,000-document durable processing), and ai-157/236 (single versus nested transactions). Implementation questions remain distinct from conceptual questions about the same mechanisms.
