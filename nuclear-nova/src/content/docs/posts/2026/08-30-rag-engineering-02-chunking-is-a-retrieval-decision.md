---
title: "RAG Engineering 02: Chunking, Parsing, and the Evidence You Lose"
date: 2026-08-30T12:00:00-04:00
author: BZ
description: "Compare chunk boundaries on the same source, preserve qualifiers and table structure, and evaluate retrieval against stable evidence rather than arbitrary chunk IDs."
categories:
  - AI ENGINEERING
tags:
  - rag
  - rag-engineering
  - chunking
  - retrieval
  - document-processing
excerpt: "Compare chunk boundaries on the same source, preserve qualifiers and table structure, and evaluate retrieval against stable evidence rather than arbitrary chunk IDs."
---

<nav class="rag-series-nav" aria-label="RAG Engineering series"><span>RAG ENGINEERING / 02</span><a href="/collections/rag-engineering/">All 10 articles</a></nav>

Chunking determines the units your system can retrieve. A useful chunk keeps a fact connected to the conditions needed to interpret it: a policy and its exception, a table value and its header, or a code fragment and its surrounding definition.

This article compares boundaries on the same passage, then examines parsing, overlap, and how to measure the evidence that survives.

<!-- more -->

<figure class="rag-diagram rag-drawio">
<p class="rag-visual-title">Parsing and chunking change different things</p>
<p class="rag-diagram-hint">Scroll horizontally to follow the full diagram →</p>
<div class="rag-diagram-scroll" role="region" aria-label="Parsing and chunking change different things; scroll horizontally on small screens" tabindex="0">
<img class="rag-figure-light no-lightbox" src="/diagrams/rag-engineering/02-parsing.svg" alt="A splitter cannot restore a table header that parsing discarded." width="720px" height="550px" loading="lazy" />
<img class="rag-figure-dark no-lightbox" src="/diagrams/rag-engineering/02-parsing.dark.svg" alt="A splitter cannot restore a table header that parsing discarded." width="720px" height="550px" loading="lazy" />
</div>
<figcaption>A splitter cannot restore a table header that parsing discarded.</figcaption>
<div class="rag-figure-links"><a href="/diagrams/rag-engineering/02-parsing.svg" target="_blank" rel="noopener">Open full size</a><a href="/diagrams/rag-engineering/02-parsing.drawio" download>Editable draw.io</a></div>
</figure>

## Start with a source whose boundaries matter

This fictional policy extends the baseline example:

```text
# Priority-one incidents
Atlas acknowledges priority-one incidents within 15 minutes.
The target applies to unplanned production outages.
Scheduled maintenance follows the maintenance agreement instead.
The incident commander owns escalation.
```

For “What is the P1 acknowledgement target?”, the first sentence contains the value. For “Does that target apply during scheduled maintenance?”, the third sentence changes the answer. For “Who owns escalation?”, the final sentence matters. The same document therefore supports several evidence shapes.

A fixed boundary after the first sentence makes the numeric fact easy to retrieve, but leaves its scope elsewhere. A single whole-document chunk preserves the relationship, but may also include unrelated sections. Neither outcome can be judged from chunk length alone.

The claim that larger chunks “average topics” is a useful intuition, not a literal description of every embedding model. Encoders apply learned transformations; the practical risk is that unrelated material changes the representation and competes for the generator's attention.

## Parse structure before splitting text



Parsing recovers a representation of the source. Chunking groups or divides that representation. These are different failure boundaries.

A PDF may contain two columns, repeated headers, footnotes, and a table spanning pages. If extraction interleaves columns, a splitter receives text whose reading order is already wrong. If it drops table headers, an embedding model cannot reliably infer which number belongs to which field.

Preserve the structural signals that survive extraction: section hierarchy, page, element order, source line range, table headers, and warnings. The [Docling technical report](https://arxiv.org/abs/2408.09869) is a useful reference for document-conversion pipelines that recover more than a plain text stream. A parser's supported formats do not establish its accuracy on your particular documents.

A representative parsing check should compare the original page with normalized output. Include scans, native PDFs, multi-column layouts, code blocks, and tables that resemble your corpus. Record failures explicitly. An ingestion job that completes successfully may still have produced unusable evidence.

## Compare strategies on the same passage

<div class="rag-table rag-table--striped" role="region" aria-label="Compare strategies on the same passage comparison" tabindex="0">

| Strategy | Boundary rule | Where it helps | What to inspect |
|---|---|---|---|
| Fixed-size | Split at a size limit | Predictable control experiment | Broken sentences and detached qualifiers |
| Recursive | Prefer stronger separators before weaker ones | General prose with uneven paragraph length | Fallback behavior on oversized elements |
| Structure-aware | Respect headings, sections, or typed elements | Policies, manuals, code, and tables | Missing hierarchy and excessively large sections |
| Similarity-based | Split near changes in adjacent representations | Documents with weak explicit structure | Threshold stability and encoder cost |

</div>

LangChain's [recursive splitter](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter) measures size in characters when using its default length function. A setting of `800` is therefore not automatically 800 tokens. A tokenizer-based length function changes that contract. Always report the unit when comparing settings.

<figure class="rag-visual">
<p class="rag-visual-title">What a chunk boundary separates</p>
<div class="rag-compare">
<section><p class="rag-panel-label">Boundary A · qualifier separated</p><div class="rag-chunk"><strong>Chunk 1</strong><p>Acknowledge P1 within <mark>15 minutes</mark>.</p></div><div class="rag-chunk"><strong>Chunk 2</strong><p>Unplanned outages only. Scheduled maintenance follows a separate agreement.</p></div></section>
<section><p class="rag-panel-label">Boundary B · rule and scope together</p><div class="rag-chunk"><strong>Chunk 1</strong><p>Acknowledge P1 within <mark>15 minutes</mark>, for unplanned outages. <mark>Scheduled maintenance is excluded.</mark></p></div></section>
</div>
</figure>

Illustrative boundaries on the policy above; text is shortened here to expose the dependency.

Boundary B better preserves the dependency needed for the maintenance question. That does not prove it produces better retrieval for every query. The claim must be checked against the labelled questions.

### What a recursive splitter actually does

“Prefer paragraph boundaries” describes a policy. The mechanism is a sequence of increasingly fine separators:

1. Try a paragraph separator and inspect the resulting pieces.
2. Keep pieces that fit the size limit; split an oversized piece again using a finer separator, such as a line break or space.
3. Merge adjacent fitting pieces up to the limit, retaining overlap according to the splitter's configuration.

This matters when one paragraph is much longer than the target chunk size. A separator-only splitter may retain that oversized paragraph. A recursive splitter can descend to smaller units. A character fallback provides a last boundary when the earlier separators cannot divide it.

My ingestion path applies the recursive splitter to each normalized document element. The actual separator order is explicit:

Excerpt from [chunking.py](https://github.com/binzhango/local_lke/blob/main/src/local_lke/ingestion/chunking.py). This is part of the application, not a standalone script.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)
texts = splitter.split_text(element.text)
```


Similarity-based splitting also needs precise naming. The repository’s [`_semantic_segments` implementation](https://github.com/binzhango/local_lke/blob/main/src/local_lke/ingestion/chunking.py) uses adjacent TF-IDF cosine similarity to detect changes in lexical overlap; an embedding-based approach uses a learned representation. Calling both “semantic chunking” without qualification conceals their different assumptions. Neither understands document authority or business exceptions merely by detecting a topic boundary.

## Overlap buys continuity at a cost

Overlap repeats text across adjacent chunks so a boundary is less likely to separate dependent facts. It also increases storage, embedding work, and duplicate retrieval.

For an idealized fixed-window splitter with window size 400 and overlap 80, the stride is 320. On a long input, the representation overhead is approximately `400 / 320 = 1.25`, or 25% more indexed text. This is a calculated approximation; paragraph-aware splitting and short documents will differ.

The more serious cost can occur at query time. If three overlapping chunks contain the same deadline, a top-three result may look well supported while covering only one fact. Deduplicate by source region where possible. Similar text from two different policy versions is a conflict to investigate, not necessarily a duplicate to discard.

## Preserve relationships beyond the chunk

A useful chunk record includes its source version and locator, its parent element, the transformation configuration, and any parser warnings. Inherit headings where they clarify the text, but distinguish original source text from metadata added for retrieval.

Consider a table:

<div class="rag-table rag-table--minimal" role="region" aria-label="Preserve relationships beyond the chunk comparison" tabindex="0">

| Severity | Acknowledge within | Applies to |
|---|---|---|
| P1 | 15 minutes | Unplanned production outage |
| P2 | 4 hours | Degraded service |

</div>

Flattening this as `P1 15 P2 4` loses both units and column meaning. A text representation might repeat headers for each row: “Severity P1; acknowledge within 15 minutes; applies to unplanned production outage.” A structured representation should retain the typed rows as well. Which representation to index depends on whether queries ask for an individual policy or an aggregate across records.

A smaller search unit can point to a larger parent. Sentence retrieval followed by parent expansion allows the retriever to match a focused fact and the generator to receive its qualifiers. That reduces the pressure to solve every context problem with overlap. Part 5 examines expansion and its budget costs.

## Evaluate evidence, not the prettiness of chunks

Changing chunk boundaries changes chunk identities. A benchmark labelled only with old chunk IDs becomes invalid after rechunking. Label stable source evidence—such as a versioned section or annotated span—and map each strategy's chunks back to those labels.

Use the same source snapshot and question set for the comparison. Include exact facts, exceptions, multi-part questions, table questions, and questions that have no answer in the corpus. Hold the embedding and retrieval configuration fixed for the first experiment.

<div class="rag-table rag-table--column" role="region" aria-label="Evaluate evidence, not the prettiness of chunks comparison" tabindex="0">

| Measurement | What it diagnoses |
|---|---|
| Answer-bearing source coverage among candidates | Whether the split made evidence retrievable |
| Coverage after context assembly | Whether useful evidence survived packing |
| Repeated source-region tokens | Prompt space spent on overlap |
| Final context size | Cost of the representation |
| Supported answer and correct exception handling | Whether retained context was sufficient |

</div>

Do not choose a strategy solely because recall improved. It may retrieve more evidence by multiplying near-duplicate chunks and increasing prompt cost. Compare quality and resource use together, then inspect the cases that changed.

A defensible starting point is structure-aware parsing with a simple bounded splitter. Add similarity-based segmentation when measured failures suggest that explicit structure is insufficient. The next article treats the resulting representation as versioned data, so changes to the parser or splitter can be reproduced and rolled back.
