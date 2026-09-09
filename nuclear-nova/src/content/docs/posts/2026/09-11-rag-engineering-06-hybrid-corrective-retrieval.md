---
title: "RAG Engineering 06: Hybrid Search, RRF, and Bounded Corrective Retrieval"
date: 2026-09-11T12:00:00-04:00
author: BZ
description: "Work through reciprocal rank fusion, distinguish full-text search from BM25, and evaluate reranking and corrective retrieval without treating scores as confidence."
categories:
  - AI ENGINEERING
tags:
  - rag
  - rag-engineering
  - hybrid-search
  - reranking
  - corrective-rag
excerpt: "Work through reciprocal rank fusion, distinguish full-text search from BM25, and evaluate reranking and corrective retrieval without treating scores as confidence."
---

<nav class="rag-series-nav" aria-label="RAG Engineering series"><span>RAG ENGINEERING / 06</span><a href="/collections/rag-engineering/">All 10 articles</a></nav>

Hybrid retrieval combines semantic matching with lexical matching. Dense search can connect a paraphrase to the right passage; keyword search can preserve exact terms such as error codes and product identifiers.

The engineering problem is how to combine the candidate lists, rank them, and decide whether another retrieval attempt is worth its cost.

<!-- more -->

## Separate candidate generation from ranking

<figure class="rag-diagram rag-drawio">
<p class="rag-visual-title">Two retrieval signals, one candidate list</p>
<p class="rag-diagram-hint">Scroll horizontally to follow the full diagram →</p>
<div class="rag-diagram-scroll" role="region" aria-label="Two retrieval signals, one candidate list; scroll horizontally on small screens" tabindex="0">
<img class="rag-figure-light no-lightbox" src="/diagrams/rag-engineering/06-hybrid.svg" alt="Fusion combines ranked lists. A reranker separately evaluates query–passage relevance." width="1025px" height="545px" loading="lazy" />
<img class="rag-figure-dark no-lightbox" src="/diagrams/rag-engineering/06-hybrid.dark.svg" alt="Fusion combines ranked lists. A reranker separately evaluates query–passage relevance." width="1025px" height="545px" loading="lazy" />
</div>
<figcaption>Fusion combines ranked lists. A reranker separately evaluates query–passage relevance.</figcaption>
<div class="rag-figure-links"><a href="/diagrams/rag-engineering/06-hybrid.svg" target="_blank" rel="noopener">Open full size</a><a href="/diagrams/rag-engineering/06-hybrid.drawio" download>Editable draw.io</a></div>
</figure>

Dense retrieval uses a learned representation to connect related wording. Lexical retrieval uses terms in the indexed text. Both can fail: dense search can prefer a thematically similar but incorrect policy, while lexical search can miss paraphrases or mishandle an identifier's punctuation.

Candidate generation should collect a bounded pool with enough recall for later stages. A reranker can improve ordering within that pool. It cannot recover a passage that neither retrieval path returned.

Keep source eligibility consistent across paths: collection, current publication, permissions, language, and explicit metadata constraints. A hybrid union that reintroduces documents excluded by one branch is a correctness and authorization problem.

## Know which lexical algorithm is running

BM25 is a lexical ranking function with term-frequency saturation and document-length normalization. A PostgreSQL full-text implementation using `tsvector`, `plainto_tsquery`, and `ts_rank_cd` is a different scoring path. Calling both “keyword search” is sometimes adequate; calling both BM25 is inaccurate.

PostgreSQL's [text-search controls](https://www.postgresql.org/docs/current/textsearch-controls.html) document parsing, dictionaries, query construction, and ranking. English stemming can help connect related word forms, while stop words and punctuation handling affect which terms survive. A rare identifier should be tested through the actual parser rather than assumed to remain one exact token.

For fields such as error codes, consider a normalized exact-match field alongside full-text search. It provides a clear equality operation when the domain requires it. Semantic retrieval can still contribute surrounding explanations.

## Calculate reciprocal rank fusion

Raw dense and lexical scores generally have different scales. Weighted score fusion can work after calibration, but its weights depend on the distributions being combined. [Reciprocal Rank Fusion (RRF)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) uses positions instead:

$$
\operatorname{RRF}(d)=\sum_{j:d\in L_j}\frac{1}{c+\operatorname{rank}_{L_j}(d)}
$$

Ranks begin at one. A missing candidate contributes zero. The constant `c` smooths the advantage of appearing at the very top of a list; it is different from the final context count often called `top_k`.

Here is a **calculated example** with `c = 60`:

```text
Dense ranking:   A, B, C
Lexical ranking: B, D, A
```

<div class="rag-table rag-table--minimal" role="region" aria-label="Calculate reciprocal rank fusion comparison" tabindex="0">

| Candidate | Dense rank | Lexical rank | RRF score, rounded |
|---|---|---|---|
| B | 2 | 1 | 0.032522 |
| A | 1 | 3 | 0.032266 |
| D | — | 2 | 0.016129 |
| C | 3 | — | 0.015873 |

</div>
<figure class="rag-visual">
<p class="rag-visual-title">Follow B through rank fusion</p>
<div class="rag-ranking">
<section><p class="rag-panel-label">Dense</p><ol><li>A</li><li><mark>B</mark></li><li>C</li></ol></section>
<section><p class="rag-panel-label">Keyword</p><ol><li><mark>B</mark></li><li>D</li><li>A</li></ol></section>
<section><p class="rag-panel-label">RRF result</p><ol><li><mark>B</mark></li><li>A</li><li>D</li><li>C</li></ol></section>
</div>
<figcaption>B appears near the top of both lists and moves to first place after fusion. The score table above shows the arithmetic.</figcaption>
</figure>

B moves above A because its positions across both lists give it more reciprocal-rank mass. These numbers are arithmetic, not measured relevance. RRF rewards agreement between lists, which can also mean agreement on the wrong passage.

The lexical branch of my hybrid retriever adds the reciprocal-rank contribution to the same candidate object used by the dense branch:

Excerpt from [service.py](https://github.com/binzhango/local_lke/blob/main/src/local_lke/retrieval/service.py). This is part of the application, not a standalone script.

```python
for rank, (chunk, score, terms) in enumerate(lexical, start=1):
    item = by_id.setdefault(chunk.chunk_id, Candidate(chunk=chunk))
    item.matched_subqueries.add(subquery)
    item.matched_terms.update(terms)
    if item.lexical_rank is None or rank < item.lexical_rank:
        item.lexical_rank = rank
        item.lexical_score = score
    item.rrf_score += 1 / (self.settings.retrieval_rrf_k + rank)
```

The [dense branch](https://github.com/binzhango/local_lke/blob/main/src/local_lke/retrieval/service.py) uses the same `chunk.chunk_id` key. Scores accumulate across the generated subqueries; final ties are broken by chunk ID. The two-list arithmetic above isolates one subquery so the fusion rule is easy to follow.

Production code should deduplicate identities within each input list and define tie behavior. Fuse by source identity, not the text's position in an array. RRF also discards score magnitude: a barely first-place passage and a dominant first-place passage receive the same rank contribution. That is a tradeoff, not a defect to hide.

### The merge key changes the result

Two retrievers can return different Python objects for the same chunk. If fusion uses `id(document)`, those objects accumulate separate scores and the shared hit may appear twice. The merge key needs to survive loading, copying, and independent retrieval.

The implementation uses `by_id.setdefault(chunk.chunk_id, Candidate(chunk=chunk))` in both branches. Independent results for the same chunk contribute to one candidate. Different chunk IDs remain distinct even if their text matches; near-duplicate removal happens later during context packing.

## Add a reranker only after measuring candidate recall

A bi-encoder computes query and document representations separately. A cross-encoder processes the query and a candidate together, allowing richer interaction at a higher per-candidate cost. The [Sentence Transformers retrieve-and-rerank guide](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) illustrates this division of work.

Choose a candidate budget, rerank that pool, then assemble a smaller context. Record pre- and post-rerank identities, the model revision, inference time, and changes in labelled ranking metrics. A difference between a cosine score and a cross-encoder logit is not a quality gain: they are not commensurate quantities.

Alternatives include an LLM that ranks candidate IDs and late-interaction methods such as [ColBERT](https://arxiv.org/abs/2004.12832). An LLM reranker adds generation cost and output-validation requirements. Late interaction retains token-level representations and adds storage/scoring work. These are useful options when their measured quality benefit justifies the cost; they are not mandatory stages of a mature system.

## Treat correction as a bounded decision

A nonempty candidate list does not prove answerability. A deterministic gate might inspect term coverage, coverage of decomposed subquestions, source authority, and available context. Such a gate is a heuristic whose false acceptances and false abstentions need evaluation. Its score should not be presented as a calibrated probability.

For the question “What is the P1 target and who owns escalation?”, context containing three deadline passages still misses the owner. A corrective attempt might use the uncovered subquestion to retrieve an escalation passage. It should record the gap that motivated the additional work.

<figure class="rag-diagram rag-drawio">
<p class="rag-visual-title">A bounded retrieval retry</p>
<p class="rag-diagram-hint">Scroll horizontally to follow the full diagram →</p>
<div class="rag-diagram-scroll" role="region" aria-label="A bounded retrieval retry; scroll horizontally on small screens" tabindex="0">
<img class="rag-figure-light no-lightbox" src="/diagrams/rag-engineering/06-correction.svg" alt="A retry has a budget and an explicit stop. Similarity alone is not an evidence-sufficiency test." width="985px" height="455px" loading="lazy" />
<img class="rag-figure-dark no-lightbox" src="/diagrams/rag-engineering/06-correction.dark.svg" alt="A retry has a budget and an explicit stop. Similarity alone is not an evidence-sufficiency test." width="985px" height="455px" loading="lazy" />
</div>
<figcaption>A retry has a budget and an explicit stop. Similarity alone is not an evidence-sufficiency test.</figcaption>
<div class="rag-figure-links"><a href="/diagrams/rag-engineering/06-correction.svg" target="_blank" rel="noopener">Open full size</a><a href="/diagrams/rag-engineering/06-correction.drawio" download>Editable draw.io</a></div>
</figure>

In [the implemented corrective path](https://github.com/binzhango/local_lke/blob/main/src/local_lke/retrieval/service.py), the service makes one alternate attempt, switching between dense and hybrid retrieval and enabling step-back rewriting when no rewrite was selected. It keeps the attempt with the higher heuristic answerability score. This is not a loop that automatically searches each missing subquestion.

Define limits on attempts, elapsed time, retrieved tokens, and allowed sources. A rewrite must preserve explicit filters and user intent. If the request says “current production policy,” correction must not broaden it to archived documents merely to obtain an answer.

The [Corrective RAG paper](https://arxiv.org/abs/2401.15884) combines retrieval evaluation with additional actions, including web search and document refinement. A local retry that switches retrieval strategies is a simpler inspired design, not a reproduction of that full method.

## Generated search probes are not evidence

HyDE generates a hypothetical document and uses its representation to retrieve real documents. The [HyDE paper](https://arxiv.org/abs/2212.10496) describes this zero-shot retrieval approach. The generated passage may contain invented facts, so it must remain a search probe rather than a citation source.

Step-back queries broaden the framing; decomposition creates narrower subqueries. Both can help and both can drift. Keep the original question, transformed queries, and candidate provenance in the trace. If correction succeeds, explain which actual source closed the evidence gap.

## Run a useful comparison

Use a fixed corpus with an exact identifier, a paraphrase, a multi-part question, and an unanswerable question. Compare dense, lexical, and hybrid candidate coverage before adding a reranker. Then compare ranking and latency with the candidate pool held fixed.

For correction, measure unsupported answers and false abstentions as well as successful recoveries. A system that retrieves repeatedly until it finds a plausible passage can look helpful while becoming less trustworthy. The objective is a supported response under a bounded policy, not a larger number of search operations.


