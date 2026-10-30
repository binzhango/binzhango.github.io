# Independent spot review: ai-435–574

Review date: 2026-09-23.

This is an independent spot review of the parent author's 140 revised answers, not a claim that a second reviewer exhaustively reviewed every technical statement in all 140 records. The parent author owns the complete editorial review. This pass examined substantive examples, calculations, executable invariants, and leftover reference-only prose.

## Coverage

Read all 16 answer bodies containing executable code in this range: `438`, `439`, `450`, `451`, `452`, `453`, `465`, `466`, `467`, `468`, `469`, `503`, `514`, `524`, `536`, and `567`. Examined their explanations, input assumptions, invariants, complexity statements, and limitations alongside the implementations.

Representative detailed-answer reading covered every 20-record batch, including `435`, `436`, `444`, `455`, `474`, `483`, `485`, `494`, `497`, `505`, `512`, `515`, `526`, `529`, `537`, `539`, `545`, `553`, `554`, `558`, `565`, `569`, `571`, and `574`. Additional numerical-example inspection covered annotation capacity, workflow reliability, regression denominators, vector-index sizing, collection yield, MinHash banding, latency budgets, web-index scale, RRF, inference capacity, retry success, contract-review costs, review-grid error counts, unsupported-claim rates, context/KV budgets, speech capacity, robot timing, and uncertainty bounds.

The inspected answers give topic-specific reasoning and examples rather than requiring readers to follow references to obtain the answer. The hypothetical prices, budgets, rates, and dataset assumptions are identified as illustrative. Operational designs generally separate successful transport/schema validation from correct application outcomes and acknowledge where their examples are only reference cores.

## Corrections applied

- **ai-438:** `AnnotationQueue.add` previously accepted fractional consensus counts such as `k=1.5`, allowing a round that could never satisfy `len(votes) == k`. Require a positive integer, rejecting booleans and noninteger values. Existing positive-integer behavior is preserved.
- **ai-466:** Corrected the diff complexity discussion. The actual implementation sorts child-name unions, so an N-child flat directory can require O(N log N), and path/list construction introduces depth-dependent work. The previous blanket worst-case linear claim omitted that work.
- **ai-529:** Corrected self-hosted cost units: accelerator-seconds must be divided by 3,600 before multiplication by cost per accelerator-hour.
- **ai-483:** Added the follow-up's concrete generator/reviewer and stronger-consultant patterns, explaining why clean-context independent review can coexist with coherent implementation ownership. This directly reconciles the apparent tension with the earlier context-sharing principle rather than offering only generic parallelism advice.
- **ai-450, ai-536, ai-574:** Removed reference-only sentences that did not teach the mechanism. Research metadata was retained.

No other question records, reconciliation decisions, source snapshots, or shared test files were changed by this spot review.

## Verification performed

The existing Python suite passed after the code correction:

`python scripts/interview/test_answers_435_598.py`

It executed **20 Python examples**, their inline assertions, the existing semantic regression cases, and the SQL fixture. The script's execution scope includes 575–598; that successful execution is not an independent prose-review claim for those additional records.

The existing Node streaming suite passed:

`node scripts/interview/test_answers_435_598.mjs`

It verifies normal TTS streaming, upstream failure, media validation, and cancellation using local HTTP fixtures. The first attempt was blocked by the filesystem/network sandbox's loopback-listen restriction; rerunning with the authorized temporary loopback servers passed on Node 26.9.0.

The independent [check script](./comprehensive-review-ai-435-574-checks.py) passed **56 checks**, recorded in [results](./comprehensive-review-ai-435-574-results.json): 54 worked numerical calculations, the new consensus-input regression cases, and a structural scan of all 140 answer bodies. Consensus checks include zero, negative, fractional, boolean, string, and missing values, plus successful one-vote completion. The structural scan found no external Markdown links or `See`/`References`-only directions in published bodies.

Source rechecks used Figure's original Helix and Helix 02 descriptions for the version-specific rates and hierarchy, Cognition's two posts for the contrasting coordination claims, and HHS cloud guidance for the ePHI/business-associate statement. The stored metadata already contained those primary URLs; no reader-facing reference section was added.

## Limits

The arithmetic fixtures validate stated hypothetical calculations, not real provider prices, production throughput, clinical performance, robot safety, or legal compliance. Code suites exercise educational implementations and controlled fixtures, not production-scale durability, multitenant authorization, or arbitrary adversarial inputs. Qualified domain review remains necessary for actual clinical, legal, and physical deployments. Shared site build, browser behavior, and release verification belong to the parent integration task and are not claimed by this spot-review report.
