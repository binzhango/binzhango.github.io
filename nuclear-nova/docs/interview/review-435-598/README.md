# Editorial review: ai-435–ai-598

Reviewed on 2026-09-20. All 164 assigned questions have an authored brief answer and detailed explanation. No questions were deleted. Source attribution remains in frontmatter; technical references used in the answers were also added to source metadata.

## Review scope

- Read each assigned prompt and authored a scenario-specific answer, with explicit assumptions and concrete failure modes, metrics, and tradeoffs.
- Kept open-ended engineering answers labeled as designs through `kind: design`; behavioral prompts use response frameworks without invented biography.
- Removed interview-session phrasing and generalized employer-motivation/ethics prompts while preserving substantive constraints and named technologies.
- Corrected clear topic mismatches, including robotics, clinical coding, int8 training, and operational debugging.
- Escaped currency symbols so Markdown math parsing cannot reinterpret financial examples as equations.
- Kept Merkle construction (465) and incremental comparison (466) as separate learning objectives. Text near-duplicates (451) and exact file duplicates (468) also remain distinct.
- Preserved absolute deployment-safety evaluation (575), which differs from measuring a relative regression in a prior build.

## Executable validation

`rtk proxy /tmp/interview-python/bin/python nuclear-nova/scripts/interview/test_answers_435_598.py`

Passed 20 published Python examples and their inline assertions, plus scenario-specific checks: annotation leases/rounds/consensus/cooldown, 100 randomized sweep-line comparisons, provider failover/timeouts/cancellation, batch limits, beam search boundaries, Merkle collision/path checks, exact-file grouping, 300 randomized rope edits and line lookups, legal-clause context limits, permission-filtered heap merge, prompt budget invariants, masked NumPy trajectory metrics, shape validation, pagination metadata, event deduplication, infection timestep ordering, and entity identity rules. The SQL example ran against an in-memory SQLite fixture demonstrating that multiple supplier issues do not multiply order totals.

`rtk proxy node nuclear-nova/scripts/interview/test_answers_435_598.mjs`

Passed isolated loopback integration tests for the published TTS proxy: successful audio streaming, upstream HTTP error, invalid media type, and browser cancellation propagating to upstream synthesis. Tested on Node.js 26.9.0. The sandbox initially disallowed binding loopback ports; the authorized escalated test run passed and closed both temporary servers.

## Primary evidence checked

- Hugging Face Hub Xet storage documentation: pointer files, object storage, chunk-level deduplication.
- DPO and ColBERT original papers for their training/ranking mechanisms.
- Cognition's June 2025 and April 2026 multi-agent posts, with their dated distinction between parallel reasoning and serialized writes.
- PagedAttention paper for allocation/sharing claims; no universal 10× claim is made.
- DAgger and Action Chunking with Transformers papers for imitation-learning distribution shift and action sequences.
- Figure's original Helix and Helix 02 descriptions; original two-system rates are explicitly versioned rather than presented as the latest architecture.
- FHIR R4 MedicationRequest and Provenance documentation; customer-specific EHR write support is not assumed.
- HHS cloud guidance for the limited stated HIPAA/cloud-provider facts.
- Palantir's Ontology overview for the definition of semantic and operational objects/actions.
- Waymo's published safety-case and deployment discussions; proposed designs are not assertions that another system is certified safe.
- Python asyncio and Node.js stream documentation for the runnable service cores.

## Numerical checks and limits

Checked illustrative arithmetic for annotation hours, 100-billion-page storage, request/token load, inference prices, contract review volume/turnaround, and KV memory. All prices and service latency budgets are explicitly hypothetical allocations, not current quotes or measured benchmarks.

The examples are scoped interview reference implementations, not deployed integrations. The entity-matching example states its verified-email uniqueness assumption; the annotation queue is single-process; the rope has randomized expected bounds; trajectory metrics use independent best modes; the streaming proxy still requires deployment authentication and provider payload validation. Clinical/legal and robotics answers describe engineering controls and qualified review, not independent clinical, legal, or safety certification. UI, full-site build, and source-manifest reconciliation are owned by the integrating agent.
