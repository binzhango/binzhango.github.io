# Original interview questions: comprehensive answer review

Reviewed and expanded on 2026-09-23.

## Scope and editorial work

Read and revised all 166 retained original question files, `llm-001` through `llm-167`, excluding `llm-164`, which was already reconciled into `llm-163`. No source snapshot or reconciliation decision was changed.

The review covered each question's prompt, brief answer, detailed answer, equations, existing code, and linked research metadata. The detailed answers now explain the mechanism or decision, work through a concrete example where useful, and state the relevant assumptions and limitations. The amount of detail was selected for the question rather than by applying a common filler paragraph. The retained detailed sections contain approximately 39,333 whitespace-delimited words (198–402 per question; code and notation included). These counts describe the result, not the basis for technical approval.

Specific improvements include:

- ML foundations: independent train/validation/test roles, distribution-shift diagnosis, concrete confusion-matrix metrics, incremental pipelines, and examples distinguishing predictive structure from causal claims.
- Objectives and architecture: numerical NLL, entropy, KL, information-gain and smoothing examples; attention-mask data flow; normalization calculations; FFN parameter budgets; activation values and derivatives.
- Inference: separate prefill, one-step decode, and full-generation complexity; exact cache payload derivation; distinction between architectural head sharing and execution kernels; online-softmax explanation; backend verification and capacity tradeoffs.
- Training: loss-mask and data-format checks, memory-accounting assumptions, adaptation regressions, data mixture effects, distributed strategy selection, and a fully specified illustrative training-compute calculation.
- Application engineering: explicit model/tool/state contracts, history versus selected context, parser validity versus factual correctness, stage-by-stage RAG diagnosis, evidence preservation, document versions, and graph-path provenance.
- PDF and chunking: added a runnable `pdfplumber` command-line extractor and a runnable paragraph/window splitter, with their limitations and boundary behavior explained.
- PEFT: concrete LoRA and bottleneck-adapter parameter calculations, persistent-state versus activation memory, and the distinction between mergeable linear updates and nonlinear added modules.

Duplicate prose caused by the earlier HyDE and PEFT merges was consolidated in `llm-143` and `llm-163`. Accurate existing equations and the six previously executable Python examples were retained. External reading links were moved to frontmatter `sources`; published answer bodies contain no external reference links or reference-only directions. Internal links connecting the BERT loading and continuation examples remain because those snippets share a documented Python session.

## Research checks

Rechecked primary material for DeepNorm residual coefficients and selected initialization projections; FlashAttention's exact dense computation and IO mechanism; GQA head sharing; HyDE and FLARE retrieval behavior; LoRA and classic bottleneck adapters; the compute/data scaling discussion; and current LangChain/LangGraph, TRL SFT, pdfplumber, and GraphRAG documentation. Original primary URLs remain in question metadata for editorial provenance. Historical checkpoint examples are identified as version-specific rather than asserted to describe every current family release. New numerical examples and teaching scenarios are original calculations or explicitly illustrative cases, not reported benchmark measurements.

## Executed verification

All six previously executable examples passed again using the existing `scripts/interview/check-original-examples.py`:

- `llm-058`: cross-attention tensor shapes, normalized weights, and padding mask.
- `llm-061`, `062`, `063`: real cached BERT checkpoint loading, hidden-state indexing, and masked pooling in the documented shared session.
- `llm-070`: categorical and binary cross-entropy outputs.
- `llm-160`: graph evidence join and unknown-entity behavior.

The BERT encoder load reports unused pretraining-head parameters, which is expected when loading the encoder through `AutoModel` rather than its pretraining heads.

The additional [validation script](./comprehensive-review-original-checks.py) passed **48 checks**, recorded in [results](./comprehensive-review-original-results.json):

- 45 independent numerical checks for new worked examples, including dropout expectation, NLL/perplexity, normalization, activations, attention and KV memory, KL direction, loss gradients, entropy, training FLOPs/time, table arithmetic, fusion scores, and PEFT parameter counts.
- `llm-141`: executable assertions plus empty input, invalid arguments, single-word windows, and 45 window/overlap combinations checking coverage and size bounds.
- `llm-139`: actual CLI execution against a generated born-digital PDF containing text, a ruled table, and a blank page; verified table values, page numbering, empty-page handling, JSON output, missing arguments, and missing input file.
- `llm-058`: perturbing only padded source vectors leaves the output unchanged.

Structural checks passed for all 166 files: required authored sections, reviewed date, balanced code fences, and absence of external body links or reference-only directions. Shared application build and browser acceptance are performed by the parent integration task; this report does not claim those were rerun by this editorial worker.

## Reproduction and limits

The original example suite uses PyTorch and Transformers; the BERT checkpoint must be cached or downloadable. The additional script uses PyTorch, pdfplumber, and ReportLab (only for its synthetic PDF test fixture), plus the Python standard library. Both were run in the existing `/tmp/interview-python` environment. The additional suite can be rerun with `python docs/interview/comprehensive-review-original-checks.py` from an environment containing those packages.

The PDF example is a native-text baseline, not an OCR system or a validated parser for every page layout. The chunker is word-based and intentionally does not claim token-budget accuracy or preservation of source offsets. Training, serving, and RAG recommendations are explained design approaches; no new large-model training run, production GPU benchmark, or domain deployment was performed as part of this editorial review.
