---
title: "RAG Engineering 01: How RAG Works, from Documents to Answers"
date: 2026-08-27T12:00:00-04:00
author: BZ
description: "Understand how RAG connects documents to an LLM: build an index, retrieve passages, assemble context, and return an answer with traceable sources."
categories:
  - AI ENGINEERING
tags:
  - rag
  - rag-engineering
  - information-retrieval
  - retrieval
  - llm
excerpt: "Understand how RAG connects documents to an LLM: build an index, retrieve passages, assemble context, and return an answer with traceable sources."
---

<nav class="rag-series-nav" aria-label="RAG Engineering series"><span>RAG ENGINEERING / 01</span><a href="/collections/rag-engineering/">All 10 articles</a></nav>

**Retrieval-augmented generation (RAG) connects an LLM to information outside its training data.** The application searches documents or records, puts relevant passages into the prompt, and asks the model to answer using that context.

For an assistant that answers questions about internal APIs, product manuals, or company policies, the engineering work is deciding **what to retrieve, what to send to the model, and how to connect the answer back to its sources**.

<!-- more -->

<figure class="rag-diagram rag-archify">
<p class="rag-visual-title">RAG: from sources to cited answers</p>
<p class="rag-diagram-hint">Scroll to follow the diagram, or open the interactive view →</p>
<div class="rag-diagram-scroll" role="region" aria-label="RAG: from sources to cited answers; scroll horizontally" tabindex="0">
<img class="rag-figure-light no-lightbox" src="/diagrams/rag-engineering/rag-evidence-path.light.png" alt="Conceptual system overview: preparation and request processing meet at the search index." loading="lazy" />
<img class="rag-figure-dark no-lightbox" src="/diagrams/rag-engineering/rag-evidence-path.dark.png" alt="Conceptual system overview: preparation and request processing meet at the search index." loading="lazy" />
</div>
<figcaption>Conceptual system overview: preparation and request processing meet at the search index.</figcaption>
<div class="rag-figure-links"><a href="/tools/rag-evidence-path.html" target="_blank" rel="noopener">Explore interactive diagram · Archify</a><a href="/diagrams/rag-engineering/rag-evidence-path.dataflow.json" download>Editable Archify specification</a></div>
</figure>

## What retrieval adds to a prompt

A language model can answer from its learned parameters and the context you provide. It cannot automatically read a private repository or discover which version of an internal policy is current. Your application needs to supply that information.

<figure class="rag-visual">
<p class="rag-visual-title">Same question, different input</p>
<div class="rag-compare">
<section><p class="rag-panel-label">Question only</p><p><strong>How do I rotate an Atlas API key?</strong></p><p>The model receives the question and its general instructions. It has no supplied evidence about your internal API.</p></section>
<section><p class="rag-panel-label">Question + retrieved documentation</p><p><strong>How do I rotate an Atlas API key?</strong></p><p>The prompt also contains the relevant section from your API guide, including the endpoint, constraints, and source version.</p></section>
</div>
</figure>

Retrieval automates the selection you would otherwise do by finding a document and pasting it into the chat. That becomes useful when the document collection is too large or changes too often for manual selection.

The original [RAG paper](https://arxiv.org/abs/2005.11401) studied a trainable retriever–generator combination. This series uses the broader application pattern: retrieve external information and make it available during generation. Joint training is not required.

## Two paths: build the index, then answer questions

Document preparation runs when sources are added or changed. Query processing runs for each question. Keeping these paths separate lets you update documents without repeating parsing and embedding work for every request.

<figure class="rag-visual"><p class="rag-visual-title">Document preparation</p><ol class="rag-step-flow"><li><strong>Parse</strong><span>Recover text, headings, tables, and source locations.</span></li><li><strong>Chunk</strong><span>Create searchable units while preserving useful boundaries.</span></li><li><strong>Index</strong><span>Store search representations with document and version IDs.</span></li></ol><figcaption>The index may support vector search, keyword search, or both. A vector database is one implementation choice.</figcaption></figure><figure class="rag-visual"><p class="rag-visual-title">Question processing</p><ol class="rag-step-flow"><li><strong>Authorize</strong><span>Determine which sources this caller can access.</span></li><li><strong>Retrieve</strong><span>Find candidate passages for the question.</span></li><li><strong>Pack context</strong><span>Select, deduplicate, and fit passages into the prompt.</span></li><li><strong>Generate</strong><span>Produce an answer and resolve its source references.</span></li></ol><figcaption>Search chooses candidates; context assembly decides which evidence the model actually receives.</figcaption></figure>

The published index joins these paths. [Explore the full architecture diagram](/tools/rag-evidence-path.html) to see the data flow and authorization boundary.

## Follow a passage into an answer

Use a small policy passage to see exactly what crosses each boundary. The following is an illustrative example:

<div class="rag-evidence-demo">
<section><p class="rag-panel-label">Source · support-v2 / P1 incidents</p><p>Atlas acknowledges priority-one incidents within <mark>15 minutes</mark>. The incident commander owns escalation.</p></section>
<section><p class="rag-panel-label">Question</p><p>How quickly does Atlas acknowledge a priority-one incident?</p></section>
<section><p class="rag-panel-label">Context sent to the model</p><p><code>C1</code> Atlas acknowledges priority-one incidents within <mark>15 minutes</mark>.</p></section>
<section><p class="rag-panel-label">Expected answer</p><p>Within <mark>15 minutes</mark>. <code>[C1]</code></p></section>
</div>

The application assigns `C1` to a known source passage. When the answer refers to it, the application resolves that ID to the original document version and section. A generated citation is useful only when that connection exists.

A minimal evidence record keeps the identities separate:

```json
{
  "chunk_id": "support-v2:p1:0",
  "document_id": "atlas-support",
  "version_id": "support-v2",
  "locator": "Priority-one incidents",
  "text": "Atlas acknowledges priority-one incidents within 15 minutes."
}
```

The document ID identifies the source, the version ID identifies its revision, and the chunk ID identifies the retrieved unit. The locator gives the reader somewhere to inspect the claim.

## Follow the implementation on GitHub

My implementation has two entry paths: a small in-memory pipeline for fixture documents, and a collection-backed service with persisted ingestion, indexing, and retrieval. Start with the smaller path so each transition is visible.

Excerpt from [pipeline.py](https://github.com/binzhango/local_lke/blob/main/src/local_lke/rag/pipeline.py). This is part of the application, not a standalone script.

```python
self.prepare()
assert self._vector_store is not None
started = perf_counter()
matches = self._vector_store.similarity_search_with_score(
    question, k=top_k or self.default_top_k
)
elapsed = _elapsed_ms(started, perf_counter())
retrieved = [
    RetrievedChunk(
        chunk=langchain_document_to_chunk(document),
        rank=rank,
        score=float(score),
    )
    for rank, (document, score) in enumerate(matches, start=1)
]
return retrieved, elapsed
```

`prepare()` loads documents, splits them, and builds an `InMemoryVectorStore` once for this pipeline instance. Each query returns the chunk, its rank, its similarity score, and retrieval latency. Those are useful debugging artifacts before inspecting any generated answer.

[Read index preparation](https://github.com/binzhango/local_lke/blob/main/src/local_lke/rag/pipeline.py) · [Read the generation handoff](https://github.com/binzhango/local_lke/blob/main/src/local_lke/rag/pipeline.py) · [Read evidence and citation construction](https://github.com/binzhango/local_lke/blob/main/src/local_lke/rag/pipeline.py)

The handoff constructs request-local citation IDs and carries the original chunk text into `GenerationService`. In this baseline, `sufficient=bool(retrieved)` only checks that retrieval returned something. The collection-backed retrieval service adds coverage heuristics, context packing, and a bounded corrective attempt; Part 6 examines those decisions.

## Retrieve broadly, send selectively

Finding ten candidates does not mean sending ten passages to the model. Some may overlap, repeat the same fact, or consume the budget without helping answer the question.

<figure class="rag-visual"><p class="rag-visual-title">Candidate set → generation context</p><ol class="rag-step-flow"><li><strong>10 candidates</strong><span>Search broadly enough to find useful passages.</span></li><li><strong>3 context items</strong><span>Select distinct evidence within a 1,200-token allowance.</span></li><li><strong>1 answer</strong><span>Generate from the question and selected context.</span></li></ol><figcaption>Illustrative settings, not universal defaults. Candidate count and context budget control different stages.</figcaption></figure>

In a larger system, apply the caller’s access scope before selecting candidates, then pack distinct source regions within the context allowance. The in-memory pipeline above isolates the basic retrieval-to-generation path; the later articles follow the collection-backed implementation.

Two checks matter independently: did retrieval find the required passage, and did context assembly keep it? If the evidence disappeared before generation, a different answer prompt will not recover it.

## Where an answer can go wrong

Use the source-to-answer path to locate the first failing boundary. [Barnett et al.](https://arxiv.org/abs/2401.05856) describe failures across this pipeline; the cards below turn that idea into practical debugging questions.

<div class="rag-debug-grid">
<section><span class="rag-card-kicker">01 / Preparation</span><h3>The fact never reached the index</h3><p>Inspect extracted text and chunk boundaries. A parser can lose a table header; a splitter can separate a rule from its exception.</p></section>
<section><span class="rag-card-kicker">02 / Retrieval</span><h3>The right passage was not found</h3><p>Inspect query encoding, keyword matching, filters, and rank. Check whether the current source version was eligible.</p></section>
<section><span class="rag-card-kicker">03 / Context</span><h3>The passage was found but dropped</h3><p>Inspect expansion, duplicate removal, and token limits. The candidate list and final prompt are different artifacts.</p></section>
<section><span class="rag-card-kicker">04 / Generation</span><h3>The answer misstates its evidence</h3><p>Compare each claim with the supplied passage. A valid citation ID does not prove that the cited text supports the claim.</p></section>
</div>

## Choose the intervention that matches the problem

<div class="rag-table rag-table--header" role="region" aria-label="Prompting, retrieval, and fine-tuning comparison" tabindex="0">

| Need | Investigate | What changes |
| --- | --- | --- |
| Clearer task or response format | Instructions and examples | How the task is expressed |
| Private or changing facts | Retrieval | What information reaches generation |
| More consistent learned behavior | Fine-tuning | The model's learned behavior |

</div>

These approaches can be combined. Fine-tuning can improve evidence use, but it does not remove the need to update source facts. Retrieval can supply a current policy, but the model can still misinterpret it.

For a first implementation, use a small corpus whose answers you can inspect. Record the source version, chunker and embedding settings, retrieved IDs, final context, and prompt. Those records make a change explainable when you later add hybrid search, reranking, or a larger model.

The next article starts with the earliest representation decision: **where to split a document so a relevant fact keeps the context needed to interpret it**.
