# Interview bank verification

Release verification: **2026-09-23**. The initial September 20 answers were subsequently expanded following the author's request for comprehensive, self-contained explanations without per-question references. The comprehensive review is recorded separately below. No deployment was performed.

## Reconciliation and content

| Measure | Result |
| --- | ---: |
| Existing source questions | 167 |
| Imported source entries | 598 |
| Total reconciled entries | 765 |
| Equivalent entries merged | 3 |
| Published canonical questions | 762 |
| Reviewed canonical questions | 762 |
| Topics | 13 |

Every input entry maps to a retained question, documented duplicate, or preserved follow-up in [reconciliation.json](reconciliation.json). The three merged IDs are `llm-164` → `llm-163`, `ai-036` → `llm-143`, and `ai-400` → `ai-327`; retired IDs remain permalink aliases. All original 167 answer pairs remain in the unchanged source snapshot. Canonical prose was rewritten with authorization; AST-based checks retain every original code block and equation. Snapshot hashes are validated. [Similarity review](similarity-candidates.json) records decisions on related questions instead of merging merely shared terminology.

All canonical records have complete Question, Brief answer, and Detailed answer sections, valid permanent IDs and topics, unique editorial order, sources, and reviewed metadata. Open-ended answers are labeled Suggested approach or Response framework. Content, font licenses, source attribution, and migration evidence are stored in the repository; the finished page does not read from Downloads.

## Comprehensive answer revision

The September 23 revision replaces shallow summaries and reference-dependent explanations with authored teaching content. Brief answers remain concise for practice cards; detailed answers explain the mechanism or proposed design, assumptions, examples or calculations, edge cases, and tradeoffs appropriate to the question. Research remains in internal metadata. Per-question reference disclosures and external research links in answer bodies are removed, with regression checks covering both source Markdown and rendered HTML.

| Editorial scope | Records | Review evidence |
| --- | ---: | --- |
| Original llm-001–167, excluding merged 164 | 166 | [Comprehensive original review](comprehensive-review-original.md); six existing examples and 48 additional checks |
| ai-001–220, excluding merged 036 | 219 | [Comprehensive imported review](comprehensive-review-ai-001-220.md); 45 executable examples and 33 new arithmetic checks |
| ai-221–434, excluding merged 400; ai-575–598 | 237 | [Comprehensive imported review](comprehensive-review-ai-271-434.md), including the additional assigned ranges |
| ai-435–574 | 140 | All 140 authored and reviewed by the integrating agent; [independent spot review](comprehensive-review-ai-435-574.md) covered every code-bearing answer and representative prose in each batch, with 56 additional checks |

The independent review corrected fractional consensus-count validation, Merkle-diff sorting complexity, accelerator-second/hour cost conversion, and the explanation of contrasting multi-agent coordination approaches. Source snapshots and stable IDs remain unchanged. These checks establish the stated examples and documented contracts; they do not claim production-scale benchmarks or domain deployment validation.

The final detailed-answer corpus contains approximately 212,619 whitespace-delimited words, including code and notation (median 267.5 per answer). Length is reported for auditability; it is not used as a substitute for the substantive review described above.

## Build and automated checks

| Check | Actual result |
| --- | --- |
| `npm test` | 63 tests passed across seven files |
| `npx astro check` | 0 errors, 0 warnings; two existing unused-variable hints in favicon tooling |
| `npm run build` | Passed; 179 pages generated; existing card integration postbuild check passed |
| Production Playwright suite | 20 tests passed in 2.0 minutes against the complete revised bank |
| Responsive/theme coverage | All eight viewport/theme combinations passed with the expanded answers |

Browser checks cover AND-based question search combined with topic filters, empty results and clearing, native disclosures and expand/collapse, question/legacy permalinks, Read/Practice switching, guarded reveal, focus retention and navigation, ratings without automatic navigation, final summaries, Review Again, restart, reload/resume, reset, corrupt/unavailable storage, reduced motion, code copy/wrap/expand, idempotent helper initialization, unique rendered IDs, local reference targets, math, print, and no-JavaScript reading/printing. The measurement check reports no page JavaScript exceptions.

Visual acceptance used **375, 768, 1024, and 1440px**, each in light and dark themes. Inspected the article header, topic rail and controls, short and long prompts, code, comparison tables, equations, behavioral frameworks, and design scenarios. No page-level horizontal overflow was found. Study controls and code buttons meet the 44px minimum height. The unrelated epoch article and blog listing retain their normal layout and styling. Screenshots are in the ignored `test-results/interview/` directory; card-only captures hide the sticky site header solely to avoid screenshot occlusion.

## Answer execution and technical review

All of these checks passed, including a final execution on 2026-09-23:

| Scope | Evidence |
| --- | --- |
| Original article | Six Python examples executed, including attention, a public BERT checkpoint, loss calculations, and graph retrieval |
| Imported 001–107 | 15 independent checks of memory, bandwidth, latency, fleet capacity, and speculative-sampling relationships; [review](review-001-107.md) |
| Imported 108–220 | 45 code examples: 40 Python, three SQL, one Go, one HTML/JavaScript; five additional numerical/gradient checks; [review](review-108-220/README.md) |
| Imported 221–270 | Five Python examples with independent boundary/reference checks and five calculations; [review](review-221-270.md) |
| Imported 271–434 | 16 Python snippets plus randomized/oracle checks of routing, integer division, NMS, caches, streaming sketches, logging, runtime and SSE behavior; [review](review-271-434/README.md) |
| Imported 435–598 | 20 Python examples, SQL fixture, randomized/scenario checks, and Node streaming integration covering success, upstream errors, invalid media and cancellation; [review](review-435-598/README.md) |

The Python verification runtime used Python 3.12 with the documented scientific libraries; Go and Node executed their language-specific examples. These are scoped interview reference implementations. Injected providers, device backends, and small fixtures test the stated contracts; they are not deployed integrations, large-model training runs, or production performance benchmarks. Primary research URLs remain in internal frontmatter metadata. Published answers retain version and assumption qualifications but contain no external research links or per-question reference disclosures. Behavioral examples do not assert personal accomplishments by the author.

## Payload and interaction measurements

The complete generated article is **5,514,186 bytes**, **880,429 bytes gzipped**. The interview controller is **7,876 bytes**, **2,932 bytes gzipped**. Compression figures are local gzip estimates, not observed HTTP transfer sizes. The DOM contains exactly 762 question elements and 762 brief-answer elements: Read and Practice share the same rendered corpus. Both Geist font families load locally. Reading time includes the bank and displays 1,124 minutes; listing excerpts use the introduction.

On local desktop Chromium, five filter inputs took **3.5–4.5ms** in their synchronous handlers. Waiting through two animation frames took **15.3–29.6ms** for matching filters, **13ms** for an empty result, and **101.1ms** when restoring the complete reading list. This is a local observation without network/CPU throttling, not a mobile performance guarantee. Raw samples are in `test-results/interview/interaction-measurements.json`.

The article path, publication metadata, global navigation and adjacent-post links are retained. Progress is local self-assessment, not mastery measurement or spaced repetition. No account, backend, React runtime, or browser-side Markdown compiler was introduced.
