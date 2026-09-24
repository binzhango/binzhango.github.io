---
title: "RAG Engineering 09: Evaluate Retrieval and Gate Regressions"
date: 2026-09-20T12:00:00-04:00
author: BZ
description: "Separate retrieval coverage, context quality, and answer support. Calculate ranking metrics and design reproducible regression gates."
categories:
  - AI ENGINEERING
tags:
  - rag
  - rag-engineering
  - evaluation
  - llmops
  - testing
excerpt: "Separate retrieval coverage, context quality, and answer support. Calculate ranking metrics and design reproducible regression gates."
---

<nav class="rag-series-nav" aria-label="RAG Engineering series"><span>RAG ENGINEERING / 09</span><a href="/collections/rag-engineering/">All 10 articles</a></nav>

Evaluate a RAG system at the boundaries where information can be lost: candidate retrieval, context assembly, and generation. A single answer score cannot tell you which stage needs work.

We will calculate ranking metrics on a three-document example, then use per-case comparisons to decide whether a candidate configuration should replace the baseline.

<!-- more -->

<figure class="rag-visual">
<p class="rag-visual-title">Ranking: B → A → C · relevant documents: A and C</p>
<dl class="rag-metrics"><div><dt>Recall@3</dt><dd>1.00</dd></div><div><dt>Reciprocal rank</dt><dd>0.50</dd></div><div><dt>nDCG@3</dt><dd>0.693</dd></div></dl>
<figcaption>Calculated values for one query. All relevant documents are present, but the first result is irrelevant. These are retrieval metrics, not answer-quality scores.</figcaption>
</figure>

## Start with a question set that can expose failures

A useful evaluation case contains more than a question and an expected sentence. It identifies the authorized corpus state, the evidence needed to answer, and the behavior expected when that evidence is absent.

```json
{
  "case_id": "policy-p1-maintenance-exception",
  "question": "Does the 15-minute P1 deadline cover planned maintenance?",
  "corpus_snapshot": "atlas-policy-v2",
  "principal_fixture": "support-reader",
  "required_source_spans": ["support-v2:deadline-and-exception"],
  "expected_behavior": "answer",
  "required_claims": ["planned maintenance is excluded"]
}
```

This is a proposed fixture format. Stable source spans make it possible to compare chunking strategies without treating one strategy's chunk IDs as the permanent ground truth. A build manifest maps those spans to the chunks produced by each representation.

Keep retrieval settings in a separate run configuration. Otherwise changing `top_k` can accidentally create a new dataset version and obscure what the experiment actually changed. Dataset identity describes the questions and labels; run identity describes the system under test. Both must be recorded.

Include paraphrases, exact identifiers, exceptions, obsolete versions, ambiguous questions, aggregate queries, and requests for facts that do not exist. Add permission boundaries as dedicated cases. A corpus containing only easy, answerable questions rewards systems that always answer.

## Measure the stages separately

<div class="rag-table rag-table--minimal" role="region" aria-label="Measure the stages separately comparison" tabindex="0">

| Stage | Question | Useful observation |
| --- | --- | --- |
| Parsing and ingestion | Did the authoritative evidence survive conversion? | Required source span exists with intact qualifiers |
| Candidate retrieval | Did search find the required evidence? | Recall at a stated cutoff, rank of the first relevant result |
| Context assembly | Did the answer model receive that evidence? | Required-span coverage within the final token budget |
| Generation | Does the answer satisfy the task using that evidence? | Claim correctness, support, completeness, abstention behavior |
| Citation handling | Can each citation be resolved and does it support its claim? | Identifier validity and claim-level citation support |
| Operations | What did this behavior cost? | Latency distribution, tokens, retries, errors, resource use |

</div>

Citation validity is a narrow check: an identifier belongs to the supplied registry. Citation support asks whether its content supports the associated claim. Reporting the first as the second makes an incorrect answer look trustworthy.

Similarly, a retrieved relevant chunk does not guarantee that the decisive sentence survived truncation. Record candidate coverage and packed-context coverage independently. Their difference directs you toward retrieval or context assembly instead of another prompt rewrite.

## Calculate retrieval metrics on a tiny example

Suppose the relevant documents are `A` and `C`, and the ranked results are `[B, A, C]`. Use binary relevance for this example.



At a cutoff of three, Recall@3 is `2 / 2 = 1`. At a cutoff of two, Recall@2 is `1 / 2 = 0.5`. Recall tells us whether the required documents were found; it does not reward putting them near the top.

The first relevant document appears at rank two, so the reciprocal rank for this query is `1 / 2 = 0.5`. Mean reciprocal rank averages this value across queries, assigning zero when no relevant item is found within the evaluated ranking. It emphasizes the first useful result. It cannot tell you whether a second required passage was also found.

Normalized discounted cumulative gain rewards relevant documents earlier in the list. Using binary gains:

```text
DCG@3  = 0/log2(2) + 1/log2(3) + 1/log2(4)
       = 1.130930

IDCG@3 = 1/log2(2) + 1/log2(3)
       = 1.630930

nDCG@3 = DCG@3 / IDCG@3
       = 0.693426
```

All three results describe the same ranking. None describes answer quality. If the task needs both `A` and `C`, reciprocal rank alone is inadequate; if only one passage is needed, requiring every loosely related document may be the wrong relevance definition.

Deduplicate results before evaluation according to the unit being measured. Returning three overlapping chunks from document `A` must not count as finding three relevant documents. If you evaluate passages, use passage labels and explain how overlap is handled.

For a no-answer case with no relevant evidence, recall has a zero denominator. Treat it as not applicable and evaluate abstention separately. Do not silently turn it into a perfect recall score. Publish metric definitions with the report, including cutoffs, label granularity, and handling of unjudged results.

### The metric definition in my evaluator

Excerpt from [service.py](https://github.com/binzhango/local_lke/blob/main/src/local_lke/evaluation/service.py). This is part of the application, not a standalone script.

```python
def _ranking_metrics(
    expected: Sequence[str], ranked: Sequence[str]
) -> tuple[float | None, float | None, float | None]:
    relevant = set(expected)
    if not relevant:
        return None, None, None
    deduped_ranked = _unique(list(ranked))
    first = next(
        (rank for rank, item in enumerate(deduped_ranked, start=1) if item in relevant),
        None,
    )
    reciprocal_rank = 0.0 if first is None else 1 / first
    hits = [1 if item in relevant else 0 for item in deduped_ranked]
    recall = len(relevant & set(deduped_ranked)) / len(relevant)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    ideal_hits = min(len(relevant), len(deduped_ranked))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / ideal if ideal else 0.0
    return round(reciprocal_rank, 6), round(recall, 6), round(ndcg, 6)
```

This implementation deduplicates the supplied ranking and returns `None` when no relevant IDs exist. Its cutoff is the list passed into this function; it does not choose `k` internally. The return order is reciprocal rank, recall, then nDCG, rounded to six decimals. That is the code behind the arithmetic above, not a reported benchmark result.

The [regression gate](https://github.com/binzhango/local_lke/blob/main/src/local_lke/evaluation/service.py) checks configured minima, p95 latency, and declines against a baseline run. A passing gate means these configured checks passed; it does not imply every security invariant in the release checklist was evaluated.

## Separate deterministic checks from live model evaluation

Some regressions should be caught without calling a model: a revoked document is still eligible, a citation points outside the registry, a query embedding has the wrong dimension, or a filter disappears from a structured plan. These are contract failures.

Live evaluation covers behavior that those checks cannot settle: whether retrieval finds a paraphrase, a reranker preserves an exception, or an answer accurately explains conflicting sources. Model-based evaluators can help scale review, but their agreement with a human rubric needs measurement. A judge can share the answer model's blind spots and can be sensitive to prompt wording.

[RAGAS](https://arxiv.org/abs/2309.15217) describes a framework for evaluating retrieval-augmented generation with automated metrics. Treat any selected metric as an operational definition with assumptions, not a certificate of correctness. Keep human-reviewed examples of both passing and failing answers, especially for exceptions and multi-source claims.

Use a development set to tune retrieval and prompts. Keep a separate holdout for the release decision. Repeatedly inspecting and optimizing against the holdout turns it into another development set; refresh or protect it accordingly.

## Make the comparison reproducible

<figure class="rag-diagram rag-archify">
<p class="rag-visual-title">Compare changes on the same cases</p>
<p class="rag-diagram-hint">Scroll to follow the diagram, or open the interactive view →</p>
<div class="rag-diagram-scroll" role="region" aria-label="Compare changes on the same cases; scroll horizontally" tabindex="0">
<img class="rag-figure-light no-lightbox" src="/diagrams/rag-engineering/rag-evaluation-flow.light.png" alt="A paired evaluation design: hold inputs fixed, compare run metrics, and apply configured regression thresholds." loading="lazy" />
<img class="rag-figure-dark no-lightbox" src="/diagrams/rag-engineering/rag-evaluation-flow.dark.png" alt="A paired evaluation design: hold inputs fixed, compare run metrics, and apply configured regression thresholds." loading="lazy" />
</div>
<figcaption>A paired evaluation design: hold inputs fixed, compare run metrics, and apply configured regression thresholds.</figcaption>
<div class="rag-figure-links"><a href="/tools/rag-evaluation-flow.html" target="_blank" rel="noopener">Explore interactive diagram · Archify</a><a href="/diagrams/rag-engineering/rag-evaluation-flow.dataflow.json" download>Editable Archify specification</a></div>
</figure>

Record the dataset revision, source snapshot, parser and chunker settings, embedding profile, index build, search parameters, context budget, prompt revision, generation model, evaluator version, and authorization fixture. Save the actual retrieved IDs and packed evidence for each case.

A model name alone is insufficient provenance when a provider can update the implementation behind it. Record any available revision and the execution time. If exact reproducibility is unavailable, state that limitation and use repeated paired runs to estimate variability.

Compare baseline and candidate on the same cases and corpus snapshot. A higher average on an easier question set is not evidence of improvement. Inspect the paired failures: which cases changed from correct to incorrect, and what stage changed first?

## Turn the report into a release decision

Consider this deliberately invented comparison on 100 fixed cases:

<div class="rag-table rag-table--column" role="region" aria-label="Turn the report into a release decision comparison" tabindex="0">

| Observation | Baseline | Candidate | Decision implication |
| --- | --- | --- | --- |
| Required evidence reaches context | 86 cases | 91 cases | Better coverage in this fixture |
| Unsupported answer claims | 3 cases | 7 cases | Investigate before release despite better coverage |
| Cross-tenant evidence exposure | 0 cases | 1 case | Block release and fix authorization |
| p95 end-to-end latency | 1.8 s | 2.4 s | Compare against the product's agreed budget |

</div>

These are examples of a decision process, not suggested targets. Define thresholds from the application and its risk tolerance before evaluating the candidate. Security invariants should not be traded away for a better average relevance score.

With small samples, a percentile can move sharply because of a few requests. Report sample counts and uncertainty; measure under representative load when making capacity claims. Separate time to first visible progress, time to first answer token, and time to a validated answer. They describe different user experiences.

A useful regression report ends with the failed case, the changed evidence path, and a concrete next action. If you cannot explain why a score changed, the evaluation has identified a symptom. The trace should help you find the cause.


