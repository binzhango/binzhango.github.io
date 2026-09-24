---
title: "RAG Engineering 04: Embedding Models and Vector Index Compatibility"
date: 2026-09-05T12:00:00-04:00
author: BZ
description: "Understand vector geometry, pin the full encoding contract, and evaluate approximate search under real filters before migrating an embedding index."
categories:
  - AI ENGINEERING
tags:
  - rag
  - rag-engineering
  - embeddings
  - pgvector
  - vector-search
excerpt: "Understand vector geometry, pin the full encoding contract, and evaluate approximate search under real filters before migrating an embedding index."
---

<nav class="rag-series-nav" aria-label="RAG Engineering series"><span>RAG ENGINEERING / 04</span><a href="/collections/rag-engineering/">All 10 articles</a></nav>

An embedding index stores vectors produced by a particular encoder and preprocessing pipeline. Query vectors must follow a compatible contract. Matching the vector length is only a shape check.

The examples below show how distance metrics change rankings, which settings belong in an embedding profile, and how to migrate an index without mixing incompatible representations.

<!-- more -->

<figure class="rag-visual">
<p class="rag-visual-title">Same vectors, different ranking</p>
<div class="rag-compare">
<section><p class="rag-panel-label">Cosine · direction</p><p><strong>A: 0.800</strong><br>B: 0.707</p><p>A is more aligned with the query direction.</p></section>
<section><p class="rag-panel-label">Dot product · direction and magnitude</p><p>A: 0.800<br><strong>B: 2.000</strong></p><p>B wins because its larger magnitude contributes to the score.</p></section>
</div>
</figure>

## Understand what similarity measures

An embedding encoder maps text to a vector. The retrieval objective is to place useful query–passage pairs near one another according to a chosen metric. Similarity is a learned signal; it is not a general definition of truth, authority, or answerability.

For nonzero vectors, cosine similarity is:

$$
\operatorname{cos}(q,d)=\frac{q\cdot d}{\|q\|\|d\|}
$$

For unit-normalized vectors, cosine similarity and dot product give the same value. Without normalization, dot product also responds to vector magnitude. Squared Euclidean distance between unit vectors is `2 - 2 × cosine`, so it yields the corresponding inverse ordering under those assumptions.

This small **calculated example** shows why the assumptions matter:

```text
query q = (1, 0)
passage a = (0.8, 0.6)
passage b = (2, 2)

cosine(q, a) = 0.8       dot(q, a) = 0.8
cosine(q, b) ≈ 0.707     dot(q, b) = 2.0
```



Cosine ranks A first; dot product ranks B first. Changing the metric without understanding normalization changes the retrieval behavior. A larger score does not mean a better answer across different metrics or encoders.

## Define the full encoding contract

<figure class="rag-diagram rag-drawio">
<p class="rag-visual-title">One vector space, one search contract</p>
<p class="rag-diagram-hint">Scroll horizontally to follow the full diagram →</p>
<div class="rag-diagram-scroll" role="region" aria-label="One vector space, one search contract; scroll horizontally on small screens" tabindex="0">
<img class="rag-figure-light no-lightbox" src="/diagrams/rag-engineering/04-compatibility.svg" alt="Matching vector dimensions alone does not establish compatibility." width="750px" height="595px" loading="lazy" />
<img class="rag-figure-dark no-lightbox" src="/diagrams/rag-engineering/04-compatibility.dark.svg" alt="Matching vector dimensions alone does not establish compatibility." width="750px" height="595px" loading="lazy" />
</div>
<figcaption>Matching vector dimensions alone does not establish compatibility.</figcaption>
<div class="rag-figure-links"><a href="/diagrams/rag-engineering/04-compatibility.svg" target="_blank" rel="noopener">Open full size</a><a href="/diagrams/rag-engineering/04-compatibility.drawio" download>Editable draw.io</a></div>
</figure>

The query and document encoders must be a compatible pair. They need not apply identical text prefixes: asymmetric retrieval models can use different instructions for questions and passages. Compatibility means following the model's intended query–document contract.

<div class="rag-table rag-table--column" role="region" aria-label="Define the full encoding contract comparison" tabindex="0">

| Profile field | Why it belongs in the record |
|---|---|
| Encoder ID and pinned revision | Identify model weights and implementation assumptions |
| Output dimension | Validate storage and query-vector shape |
| Normalization and distance metric | Define how vectors are compared |
| Query and document instructions | Preserve asymmetric encoding behavior |
| Tokenizer, maximum length, truncation | Explain what input the encoder actually consumed |
| Text preparation revision | Track heading prefixes, cleanup, and contextual additions |
| Modality | Separate text-only and image/text representations |

</div>

A profile can be serialized canonically and assigned a digest. Record it on the build and compare it with the query encoder's profile before searching. Do not infer compatibility solely from the advertised model name or vector length.

A changed revision may preserve compatibility or change it materially. The operationally safe assumption is that it requires evaluation; describing every revision change as guaranteed meaningless search would be too strong. Explicit compatibility decisions are preferable to silently mixing vectors.

## What the repository records and checks

Excerpt from [service.py](https://github.com/binzhango/local_lke/blob/main/src/local_lke/indexing/service.py). This is part of the application, not a standalone script.

```python
def profile(self) -> EmbeddingProfileResponse:
    return self.repository.get_or_create_profile(
        modality=EmbeddingModality.TEXT,
        model_id=self.embeddings.model_id,
        revision=self.embeddings.revision,
        dimension=self.settings.embedding_dimension,
        normalized=self.embeddings.normalized,
        document_prefix=self.embeddings.document_prefix,
        query_prefix=self.embeddings.query_prefix,
    )
```

The profile records model identity, revision, dimension, normalization, and separate document/query prefixes. That metadata makes a build inspectable. However, the current [query-time guard](https://github.com/binzhango/local_lke/blob/main/src/local_lke/indexing/service.py) checks vector length against the selected profile; it does not compare every recorded field with the running provider. Two incompatible models with equal dimensions can therefore pass this guard. Full profile matching is a strengthening to implement, not a guarantee this revision already provides.

## Validate provider output before publication

An embedding response can be malformed even when the request succeeds. Check the number of returned vectors, their dimensions, finite numeric values, and the expected association with input items. A missing or reordered result can attach a correct vector to the wrong source text.

The indexing service checks each returned vector against the registered dimension before constructing its database write:

Excerpt from [service.py](https://github.com/binzhango/local_lke/blob/main/src/local_lke/indexing/service.py). This is part of the application, not a standalone script.

```python
for node, vector in zip(batch, vectors, strict=True):
    if len(vector) != profile.dimension:
        raise IndexingError(
            f"Embedding dimension {len(vector)} does not match profile "
            f"dimension {profile.dimension}",
            code="embedding_dimension_mismatch",
        )
```


The excerpt checks dimensions; it does not establish the finite-value and normalization checks discussed above. For cosine search, explicitly handle zero vectors. If normalization is part of the provider contract, check it with a numerical tolerance or normalize at a single documented boundary. Do not let index-time and query-time code make different decisions.

Keep partial writes staged. Once the expected evidence IDs are present and validation succeeds, publish the build using the lifecycle from Part 3. A model download, cold initialization, embedding inference, and database insertion are different costs; record them separately when interpreting a build duration.

## Separate embedding quality from approximate-search quality

An exact search computes distances over the eligible vectors. Approximate nearest-neighbor search reduces the work by visiting a subset of the index. It introduces another possible source of missed evidence.

HNSW and IVFFlat offer different construction, memory, and query tradeoffs. The [pgvector documentation](https://github.com/pgvector/pgvector) describes exact search, both index families, and tuning controls. In particular, filtering an approximate search can leave fewer eligible results than expected; index settings and query plans affect the behavior.

That creates two different evaluation questions:

<div class="rag-table rag-table--minimal" role="region" aria-label="Separate embedding quality from approximate-search quality comparison" tabindex="0">

| Comparison | What it measures |
|---|---|
| Encoder A versus encoder B using exact search | Representation quality under the selected metric |
| Approximate versus exact search using the same vectors | Index approximation loss |
| Unfiltered versus tenant-filtered queries | Behavior under the actual eligibility constraints |

</div>

An index can have excellent approximate recall over the whole corpus and perform poorly for a small filtered collection. Benchmark the shape of real filters, not just unfiltered nearest-neighbor search.

Use exact retrieval on a manageable evaluation corpus as a diagnostic reference. If exact search misses the relevant passage, tuning HNSW search breadth cannot repair the underlying representation or label mismatch. If exact search finds it and approximate search does not, investigate the index and filter interaction before replacing the model.

## Migrate by building a second searchable publication

Suppose profile A encoded the current corpus and profile B is a candidate replacement. Build B separately, evaluate it on the same source snapshot, and preserve A until the migration decision is made.

A migration comparison should hold the corpus and questions fixed, then report retrieval coverage, ranking, context size, latency, and answer quality. If profile B uses longer contextualized chunks, that is a second changed variable; record it rather than attributing all improvement to the encoder.

After acceptance, switch the published profile and ensure query encoding follows that profile. A rolling deployment must either support both profiles during the transition or route requests to a compatible worker. Switching a database pointer while some application instances still encode with A creates a mixed-version failure.

Keep rollback concrete: retain the prior profile, index artifacts, and compatible query configuration. An old index is not a useful rollback target if the service can no longer load its encoder.

## Know when an absent index is a different condition

A small development application may compute embeddings on demand when no index exists. That can be acceptable for a tiny corpus, but it changes query cost and failure modes. It should appear in the trace as a different retrieval path.

An incompatible published index is different. Falling back silently hides a deployment mistake. Expose the incompatibility, preserve the failed build state, and return an actionable error or explicitly configured degraded behavior.

The resulting contract is simple to state and demanding to operate: every retrieval request must use an eligible source publication, a compatible encoder pair, and a known search configuration. The next article examines what happens after those vectors return candidates.
