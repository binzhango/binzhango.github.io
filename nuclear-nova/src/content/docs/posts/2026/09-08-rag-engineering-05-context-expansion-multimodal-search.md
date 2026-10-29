---
title: "RAG Engineering 05: Context Assembly, Parent Retrieval, and Multimodal Evidence"
date: 2026-09-08T12:00:00-04:00
author: BZ
description: "Turn retrieved candidates into useful context by preserving qualifiers, tracking expansion provenance, budgeting tokens, and separating image retrieval from visual answering."
categories:
  - AI ENGINEERING
tags:
  - rag
  - rag-engineering
  - vector-search
  - context-engineering
  - multimodal
excerpt: "Turn retrieved candidates into useful context by preserving qualifiers, tracking expansion provenance, budgeting tokens, and separating image retrieval from visual answering."
---

<nav class="rag-series-nav" aria-label="RAG Engineering series"><span>RAG ENGINEERING / 05</span><a href="/collections/rag-engineering/">All 10 articles</a></nav>

Retrieval returns a ranked list. The LLM receives a prompt. **Context assembly is the step between them**: expand useful passages, remove duplication, preserve source references, and fit the evidence budget.

A passage can be found by search and still disappear from the final prompt. This article follows that transition, including parent retrieval and multimodal evidence.

<!-- more -->

<figure class="rag-diagram rag-archify">
<p class="rag-visual-title">From a search hit to generation context</p>
<p class="rag-diagram-hint">Scroll to follow the diagram, or open the interactive view →</p>
<div class="rag-diagram-scroll" role="region" aria-label="From a search hit to generation context; scroll horizontally" tabindex="0">
<img class="rag-figure-light no-lightbox" src="/diagrams/rag-engineering/rag-context-expansion.light.png" alt="Window and parent expansion are alternative paths. The parent is a parser-defined element, not necessarily a whole section." loading="lazy" />
<img class="rag-figure-dark no-lightbox" src="/diagrams/rag-engineering/rag-context-expansion.dark.png" alt="Window and parent expansion are alternative paths. The parent is a parser-defined element, not necessarily a whole section." loading="lazy" />
</div>
<figcaption>Window and parent expansion are alternative paths. The parent is a parser-defined element, not necessarily a whole section.</figcaption>
<div class="rag-figure-links"><a href="/tools/rag-context-expansion.html" target="_blank" rel="noopener">Explore interactive diagram · Archify</a><a href="/diagrams/rag-engineering/rag-context-expansion.dataflow.json" download>Editable Archify specification</a></div>
</figure>

## Track three different objects

A search hit, an expanded context item, and a citation target are related but not interchangeable.

<div class="rag-table rag-table--column" role="region" aria-label="Track three different objects comparison" tabindex="0">

| Object | Example | Purpose |
|---|---|---|
| Search hit | Sentence 1 of the P1 section | Explains why the retriever selected the region |
| Context item | Entire P1 section | Supplies the deadline and its exception |
| Citation target | Versioned P1 section with source locator | Lets a reader inspect the supporting evidence |

</div>

If expansion replaces a sentence with its parent and discards the triggering ID, the trace becomes harder to explain. Preserve both the matching child and the returned source region. If the citation points only to the child, it may omit the exception used in the answer.

These **illustrative context records** show the intended separation:

```json
{
  "trigger_id": "support-v2:p1:sentence-1",
  "context_id": "support-v2:p1:section",
  "version_id": "support-v2",
  "expansion": "parent_section",
  "locator": "Priority-one incidents"
}
```

## Compare sentence windows and parent expansion



A sentence window retrieves a narrow unit, then includes a bounded number of neighbors. It is useful when nearby prose supplies definitions or qualifiers. Define what “neighbor” means: within the same chunk, paragraph, page, or section. Crossing a document or version boundary accidentally is not context expansion.

Parent retrieval selects a child and returns a larger structural unit. A heading-aware parent may preserve a complete policy, but a very large section can reintroduce the cost of whole-document prompting. Parent size still needs a limit.

In the Atlas fixture, a one-sentence window after the deadline includes applicability to unplanned outages, but can still miss the maintenance exception two sentences later. Parent expansion captures all three statements if they share a section. This is a property of the example's structure, not a universal advantage of parents over windows.

The repository reconstructs the window from active sentence nodes with the same embedding profile and document version, ordered by chunk ordinal and sentence ordinal. After locating the hit, it takes this slice:

Excerpt from [repository.py](https://github.com/binzhango/local_lke/blob/main/src/local_lke/indexing/repository.py). This is part of the application, not a standalone script.

```python
position = next(
    (index for index, item in enumerate(records) if item.id == hit.id),
    None,
)
if position is None:
    return hit.text, hit.token_count
window = records[max(0, position - radius) : position + radius + 1]
return " ".join(item.text for item in window), sum(
    item.token_count for item in window
)
```

This window can cross chunk boundaries within that version; it is not restricted to the hit’s paragraph. A radius of one can still miss an exception two sentences away. Increasing the radius changes the generation context without changing the sentence that triggered retrieval.

Parent expansion takes a different path: [`parent_context()`](https://github.com/binzhango/local_lke/blob/main/src/local_lke/indexing/repository.py) loads the chunk’s `parent_element_id` and returns that normalized element. Its size depends on parser output; the parent is not automatically an entire heading section. Inspect the actual element before assuming it includes every qualifier.

Multi-granularity indexing adds several search representations—sentence, chunk, section, or summary. A summary can improve discovery while omitting a decisive qualifier. If generated summaries are indexed, retain links to the original text and use the source passages to support factual claims. A summary does not become authoritative merely because it was generated during ingestion.

## Deduplicate source regions before spending the budget

Suppose the top three sentence hits all belong to the same P1 section. Expanding each independently creates three copies of one section. The model receives more tokens but no additional evidence.

Deduplicate by a stable source-region identity that includes the version. Keep the triggering hits as provenance and include the expanded region once. If windows overlap, either merge them within a bounded region or retain an explicit overlap policy.

Do not deduplicate solely by text. Identical text in a current policy and an obsolete policy carries different lifecycle meaning. Similar statements from independently authoritative sources can provide useful corroboration. Conversely, multiple copies of one source are not independent confirmation.

## Allocate the prompt budget explicitly

The model context window must accommodate the complete request and the generated response. The evidence allowance is what remains after other uses are reserved:

$$
B_{evidence}=B_{window}-B_{instructions}-B_{question}-B_{history}-B_{output}-B_{margin}
$$

Use the actual tokenizer and chat-template accounting where possible. A character estimate is useful for early rejection, but it is not an exact token count. Tool descriptions and formatting overhead also consume context when present.

Consider this **calculated packing example**, with an evidence allowance of 900 tokens and a question asking for both the target and escalation owner:

<div class="rag-table rag-table--striped" role="region" aria-label="Allocate the prompt budget explicitly comparison" tabindex="0">

| Candidate context | Assumed token cost | Contribution | Decision |
|---|---|---|---|
| P1 policy section | 520 | Deadline, scope, maintenance exception | Include |
| Another hit expanding to the same section | 520 | Duplicate source region | Exclude |
| Escalation section | 260 | Incident commander responsibility | Include |
| Broad incident handbook | 700 | Mostly redundant background | Exclude: 120 tokens remain |

</div>
<figure class="rag-visual rag-budget">
<p class="rag-visual-title">900-token evidence allowance</p>
<label for="rag-context-budget">780 used · 120 available</label>
<meter id="rag-context-budget" min="0" max="900" value="780">780 of 900 tokens</meter>
<dl class="rag-metrics"><div><dt>P1 policy</dt><dd>520</dd></div><div><dt>Escalation</dt><dd>260</dd></div><div><dt>Remaining</dt><dd>120</dd></div></dl>
<figcaption>Calculated example. The duplicate section adds no evidence; the 700-token handbook does not fit in the remaining allowance.</figcaption>
</figure>

The result uses 780 tokens and covers both requested facts. Blindly taking the top three expanded hits would spend the budget on duplication. Filling every remaining token is not a requirement.

A coverage-aware packer can prioritize evidence for uncovered parts of a question before adding redundant support. The tradeoff is that coverage classification can itself be wrong. Record which subquestion each candidate is believed to support, and inspect errors in that mapping.

## Truncation can remove the answer's conditions

Truncating the end of an oversized section is deterministic, but it is not necessarily faithful. The maintenance exception in our fixture may be the first thing removed. A trace saying “included” would hide that loss.

Record whether context was included in full, truncated, excluded by budget, or replaced by a smaller source region. When possible, split at structural boundaries and preserve the relationships needed for the question. If the necessary evidence does not fit, reduce other context or explain the coverage gap rather than asserting completeness.

Context ordering also deserves evaluation. [Lost in the Middle](https://arxiv.org/abs/2307.03172) found that evidence position affected performance on the models and tasks studied. Treat this as a reason to test order sensitivity on your model, not as proof that one ordering rule always wins.

Two experiments help isolate the issue: keep the context set fixed and change its ordering; then keep the ordering rule fixed and vary the budget. Combining both changes makes it difficult to identify which one affected the answer.

## Images introduce another representation decision

A text embedding does not automatically encode the content of an image. A [CLIP-style dual encoder](https://arxiv.org/abs/2103.00020) places text and images into a shared representation space, enabling a text query to retrieve related images. That can support discovery of a diagram or screenshot without making the system capable of reading every value inside it.

Distinguish three capabilities:

<div class="rag-table rag-table--header" role="region" aria-label="Images introduce another representation decision comparison" tabindex="0">

| Capability | Retrieved or produced object | Additional requirement |
|---|---|---|
| Text-to-image search | Ranked image assets | Compatible text/image encoder pair |
| OCR-assisted retrieval | Recognized text with page or region provenance | OCR quality and reading-order checks |
| Visual question answering | An answer based on image content | A model that consumes the image plus answer evaluation |

</div>

Document-page retrieval systems such as [ColPali](https://arxiv.org/abs/2407.01449) explore richer visual representations than one whole-image similarity vector. They introduce different storage and scoring costs. Use them when the document layout carries information that text extraction loses, and evaluate on those cases.

The same provenance rules apply to every modality: identify the asset version, page or region, triggering representation, and what the generator received. Image similarity is not proof that the retrieved diagram contains the requested answer.

## Measure what survives into generation

Report retrieval coverage both before and after assembly. A reranker may improve candidate order while an expansion policy removes the relevant qualifier. A larger token budget may improve answer completeness while increasing latency. These are different outcomes.

For each failing query, inspect the final context text, source regions, token accounting, and exclusion decisions. The context manifest is the bridge between search metrics and answer quality. Without it, a successful retrieval stage can take the blame for a packing failure it did not cause.
