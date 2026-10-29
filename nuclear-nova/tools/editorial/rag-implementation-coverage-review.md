# RAG implementation coverage review

Reviewed 2026-09-08. Scope: the ten RAG Engineering posts and the author’s implementation checkout. This is a code and test audit, not a production certification. No implementation or blog content was changed during this review.

## Conclusion

The repository has executable coverage for all ten major subject areas. It is a substantial local teaching workbench, but it does **not** implement every guarantee, extension, or operational practice discussed in the blogs. Several capabilities exist only in isolated lab endpoints rather than the answering path, and additional probes exposed correctness defects despite a passing deterministic suite.

Use “implemented core,” “partial,” and “design guidance” explicitly. Do not describe the whole series as an already completed end-to-end production implementation.

## Verification performed

- Deterministic pytest selection: 104 passed initially; the PostgreSQL test was blocked by sandbox shared-memory permissions. Its isolated rerun passed after escalation: **105 deterministic tests passed overall**.
- Two live-provider tests were deliberately excluded; no real model calls or model downloads were performed.
- Ruff passed for `src` and `tests`.
- Strict mypy passed for all 49 source files.
- Additional temporary SQLite/service probes checked publication, profile compatibility, and partial model migration. Pure-function probes checked ranking metrics and query rewriting.
- Existing PDF normalization tests mock the PDF partitioner. They do not establish real OCR or complex-layout accuracy.
- Existing ranking tests predominantly use deterministic fake embeddings and chat. They establish contracts and fixtures, not measured BGE, CLIP, or cross-encoder quality on a representative corpus.
- Implementation checkout remained unchanged; the pre-existing untracked `.idea/` directory remained.

## Coverage by article

| Part | Working implementation | Incomplete or outside the implemented scope | Assessment |
| --- | --- | --- | --- |
| 1 — RAG baseline | Document loading, splitting, in-memory similarity search, generation handoff, source registry, traces, validated responses. | Baseline sufficiency is nonempty retrieval, not evidence support. Live model quality was not exercised. | Implemented teaching baseline. |
| 2 — Parsing and chunking | Markdown hierarchy and source lines, text paragraphs, PDF adapter and metadata normalization, recursive splitting, stable versioned IDs, TF-IDF adjacency heuristic. | Markdown and recursive chunk strategies use the same splitter over normalized elements. “Semantic” splitting is an experimental lexical heuristic. No representative OCR/table extraction evaluation or tokenizer-based chunk contract. | Partial coverage of the broader parsing/chunking discussion. |
| 3 — Versioned ingestion | Content/configuration hashes, immutable versions, stored jobs, duplicate avoidance, explicit retries, abandoned-job recovery, resumable vector batches. | Ingestion and vector publication are separate. No complete collection-build cutover, worker leases, parser process isolation, parser deadline enforcement, or distributed recovery guarantee. | Implemented local lifecycle; publication guarantee incomplete. |
| 4 — Embedding contracts | Registered model/revision/prefix metadata, dimensions and batch-count checks, pgvector storage, HNSW migration, profile selection and index state. | Equal-dimension provider mismatch is accepted. Profile migration activates per document before collection completion. No application-level finite/zero/norm checks, full compatibility matching, or ANN-vs-exact benchmark suite. | Partial, with reproduced correctness gaps. |
| 5 — Context and multimodal | Sentence/window/parent/multi-granularity retrieval lab, provenance, deduplication, budget accounting, validated images, CLIP adapter and image search. | Normal answering requests do not carry an expansion option; dense candidates are mapped back to original chunks before generation. Image search does not feed a multimodal answer model. Budgets are word/regex estimates rather than exact prompt tokens. | Lab capabilities exist; full answer-path integration incomplete. |
| 6 — Hybrid/corrective retrieval | PostgreSQL lexical search, dense search, chunk-ID RRF, optional cross-encoder, context manifests, sufficiency heuristic, one alternate retrieval attempt. | Persisted dense metadata filtering occurs after candidate limiting. HyDE/step-back are string-prefix probes rather than generated hypothetical documents or abstractions. Reranker gain subtracts unlike score scales; no calibrated semantic sufficiency. | Working hybrid core; several simplifications and a recall risk. |
| 7 — Planning and structured queries | Typed metadata plans, CSV inference, allowlisted SQLAlchemy compilation, column/type checks, row limits, PostgreSQL read-only transaction and statement timeout. | No unified automatic passage/SQL dispatcher or combined answer; caller selects structured endpoint/table. Decomposition uses punctuation and phrase heuristics and misses ordinary “X and Y” questions. | Structured endpoint implemented; general routing incomplete. |
| 8 — Grounded generation | Pydantic output modes, registry citation checks, bounded repair, extractive degradation, evidence-only mode, abstention, prompt separation, safe UI rendering. | No semantic entailment/contradiction verifier. Upstream active/retrieved flags are trusted. Fixture deltas are sliced after validation; collection streaming emits completed output, not live validated claim streaming. | Output integrity implemented; factual support and advanced streaming remain open. |
| 9 — Evaluation | Immutable question sets, persisted case runs, phrase/status/citation metrics, threshold gates, baseline comparisons and provider fault injection. | Runtime metric bug; retrieval and packed-context measurement conflated; no pinned corpus snapshot/full configuration or stored complete trace; no semantic judge, structured-table/image evaluation dataset path, or representative load/quality benchmark. | Useful evaluation framework; correctness and coverage work required. |
| 10 — Security/operations | Opt-in bearer authentication, collection roles, nested-resource checks, admin evaluation boundary, audit records, secure-mode UI disablement, upload and SQL controls. | No document-level ACL/RLS, shared evidence/answer cache and invalidation system, in-flight revocation guarantee, retention/purge orchestration, distributed limits, or enterprise identity integration. | Governed local API implemented; broader operational chapter remains guidance. |

## Findings to fix first

### 1. Runtime nDCG can exceed 1 and fail an evaluation run

**Reproduced.** The blog quotes `_ranking_metrics`, but `_run_case` calls `_retrieval_metrics`. For source relevance `A`, three returned chunks from source `A` produce:

| Function | Input interpretation | MRR | Recall | nDCG |
| --- | --- | --- | --- | --- |
| Blog helper `_ranking_metrics` | Deduplicated source ranking `[A, A, A]` | 1 | 1 | 1 |
| Runtime `_retrieval_metrics` | Sources `[A, A, A]`, distinct chunks `[a1, a2, a3]` | 1 | 1 | 2.130930 |

The collection path preserves repeated source IDs in `_ranked_retrieval`. Runtime DCG rewards each repeated source hit while ideal DCG counts the single relevant source. `EvaluationCaseResult.ndcg_at_k` requires a value at most 1, so this result fails model validation and can mark the run failed. The helper named in Part 9 does not establish runtime behavior.

**Repair:** define a single relevance unit per metric, deduplicate source-level rankings, preserve paired source/chunk associations, and test the actual `_run_case` path with multiple chunks from one relevant source. Separate candidate recall from packed-context coverage; currently the collection path measures included/truncated manifest entries while the fixture path uses retrieved ranks.

Source: [evaluation service](https://github.com/binzhango/local_lke/blob/main/src/local_lke/evaluation/service.py), [evaluation result constraints](https://github.com/binzhango/local_lke/blob/main/src/local_lke/evaluation/models.py).

### 2. Index publication and model migration can temporarily lose searchable documents

**Reproduced using two documents and isolated SQLite storage.** Before updating a document, vector search returned two distinct versions. After ingesting the new version but before indexing it, vector search returned only one: the old version is already inactive and the new one has no searchable vectors.

A second probe rebuilt only one document under a new embedding profile. That operation switched the collection’s active profile immediately; the other document disappeared from default-profile search. `index_collection` is a loop over `index_version`, so it also has this intermediate state. A failure during migration can leave it in place.

**Repair:** stage an entire publication, verify every required document/index, and activate a collection build atomically. Keep the previous searchable publication available until cutover. Recheck active-version state when a long-running build commits; test updates and failures between stages.

Source: [ingestion persistence](https://github.com/binzhango/local_lke/blob/main/src/local_lke/storage/repository.py), [index service](https://github.com/binzhango/local_lke/blob/main/src/local_lke/indexing/service.py), [activation and search eligibility](https://github.com/binzhango/local_lke/blob/main/src/local_lke/indexing/repository.py).

### 3. Query compatibility accepts a different model when dimensions match

**Reproduced.** An index built with one model identity was searched successfully through a provider with a different model ID but the same vector length. The test isolates the missing guard; it does not claim that this synthetic provider changed actual semantic coordinates.

The profile records the necessary metadata, but the query path validates only dimension. The same weakness applies to the active multimodal profile’s query validation.

**Repair:** match the running encoder to the selected profile, including model identity/revision, normalization, and task prefixes. Test incompatible equal-dimension encoders, not only dimension mismatches. Treat fallback behavior explicitly so it cannot hide a broken deployment contract.

Source: [text index search](https://github.com/binzhango/local_lke/blob/main/src/local_lke/indexing/service.py), [image search](https://github.com/binzhango/local_lke/blob/main/src/local_lke/indexing/images.py).

### 4. Persisted dense retrieval applies metadata constraints after a limited search

**Code-confirmed; not separately reproduced under corpus load.** `_dense_candidates` asks `indexing.search` for collection-wide candidates, then removes chunk IDs outside `list_active_chunks(..., filters)`. A selective filename/page constraint can therefore yield no dense evidence even when an eligible match exists below the unfiltered candidate cutoff. Collection authorization is a separate boundary; this finding concerns metadata-filtered recall, not a demonstrated cross-collection leak.

**Repair:** push eligibility predicates into the vector query before ordering/limit. Add a fixture with many high-ranking ineligible chunks and one eligible chunk beyond the original candidate pool.

Source: [dense candidate path](https://github.com/binzhango/local_lke/blob/main/src/local_lke/retrieval/service.py), [vector query](https://github.com/binzhango/local_lke/blob/main/src/local_lke/indexing/repository.py).

### 5. Expansion exists in the lab but is not carried into ordinary answer generation

**Code-confirmed.** `VectorSearchRequest` supports expansion. The answering path constructs it without an expansion selection, then maps candidate IDs back to `ActiveChunk` and packs `chunk.text`. Expanded `context_text` from the index service is not handed to the generation service on this path.

**Repair:** make the selected expansion strategy part of the answer request contract, carry expanded text and original hit provenance through ranking and packing, and test an answer requiring a qualifier outside the triggering chunk. Add image-to-answer integration only if that is actually part of the product objective; image retrieval alone is a valid scoped feature.

Source: [retrieval service](https://github.com/binzhango/local_lke/blob/main/src/local_lke/retrieval/service.py), [expansion implementation](https://github.com/binzhango/local_lke/blob/main/src/local_lke/indexing/service.py), [request models](https://github.com/binzhango/local_lke/blob/main/src/local_lke/models.py).

## Other important boundaries

- **Query rewriting:** a probe using “What is the P1 target and who owns escalation?” returned `simple_lookup`. HyDE prepended “A relevant source passage would directly explain …”; step-back prepended “Background principles and context needed to answer: …”. Neither called a model to generate the content described by those research techniques. [Planning source](https://github.com/binzhango/local_lke/blob/main/src/local_lke/retrieval/planning.py).
- **Token budgets:** truncation slices whitespace-separated words; chunk token counts use a regex. Neither establishes a model-token ceiling including policy, schema, question, and output reserve. [Context packing](https://github.com/binzhango/local_lke/blob/main/src/local_lke/retrieval/service.py).
- **Citation checks:** valid IDs and schema do not prove claims are entailed. The code does not implement the third semantic-support check illustrated in Part 8. The article already explains this distinction; the diagram must continue to be read as a teaching model. [Generation source](https://github.com/binzhango/local_lke/blob/main/src/local_lke/generation/service.py).
- **Evaluation reproducibility:** equal immutable dataset IDs freeze questions and labels, not the active document corpus. Configuration snapshots omit some relevant settings and index/source identities; case results retain a trace ID without the complete generation/evidence trace. [Evaluation service](https://github.com/binzhango/local_lke/blob/main/src/local_lke/evaluation/service.py).
- **Secure scope:** the tested role boundary is collection-level and opt-in. Disabled authentication is intentional for the local workbench. Cache/revocation/retention designs in Part 10 are not implemented merely because collection grants can be revoked. [Security service](https://github.com/binzhango/local_lke/blob/main/src/local_lke/security/service.py), [secure application construction](https://github.com/binzhango/local_lke/blob/main/src/local_lke/web/app.py).

## Suggested completion sequence

1. Fix runtime evaluation semantics and tests so subsequent comparisons are trustworthy.
2. Fix full encoder compatibility and collection-wide publication/migration behavior.
3. Push metadata filtering into vector retrieval; connect context expansion to answers and enforce actual prompt-token budgets.
4. Decide which advanced topics are implementation goals: real query rewriting, unified structured routing, semantic claim checking, and multimodal generation. Keep the rest labelled as extensions.
5. Add real-provider and representative corpus evaluation, then implement the required security/operations guarantees for the intended deployment.

The blogs can teach beyond the repository. The necessary correction is to make the boundary explicit, rather than claiming every topic is already implemented or forcing every advanced technique into a local teaching application.
