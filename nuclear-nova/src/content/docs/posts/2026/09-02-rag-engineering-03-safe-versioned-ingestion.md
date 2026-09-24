---
title: "RAG Engineering 03: Versioned Ingestion and Safe Index Publication"
date: 2026-09-02T12:00:00-04:00
author: BZ
description: "Design document ingestion around immutable representations, idempotent jobs, and atomic publication so failed updates cannot silently replace searchable evidence."
categories:
  - AI ENGINEERING
tags:
  - rag
  - rag-engineering
  - ingestion
  - document-processing
  - data-quality
excerpt: "Design document ingestion around immutable representations, idempotent jobs, and atomic publication so failed updates cannot silently replace searchable evidence."
---

<nav class="rag-series-nav" aria-label="RAG Engineering series"><span>RAG ENGINEERING / 03</span><a href="/collections/rag-engineering/">All 10 articles</a></nav>

Ingestion turns changing documents into a searchable index. That makes it a versioned data pipeline: retries should not create duplicates, partial work should not become visible, and queries should resolve a complete publication.

The key design is to **stage a new build, validate it, then publish it**. The old build can continue serving ordinary content updates while the replacement is prepared.

<!-- more -->

<figure class="rag-diagram rag-drawio">
<p class="rag-visual-title">Publish a complete version</p>
<p class="rag-diagram-hint">Scroll horizontally to follow the full diagram →</p>
<div class="rag-diagram-scroll" role="region" aria-label="Publish a complete version; scroll horizontally on small screens" tabindex="0">
<img class="rag-figure-light no-lightbox" src="/diagrams/rag-engineering/03-ingestion.svg" alt="Staging can fail without exposing a partially built index to readers." width="945px" height="390px" loading="lazy" />
<img class="rag-figure-dark no-lightbox" src="/diagrams/rag-engineering/03-ingestion.dark.svg" alt="Staging can fail without exposing a partially built index to readers." width="945px" height="390px" loading="lazy" />
</div>
<figcaption>Staging can fail without exposing a partially built index to readers.</figcaption>
<div class="rag-figure-links"><a href="/diagrams/rag-engineering/03-ingestion.svg" target="_blank" rel="noopener">Open full size</a><a href="/diagrams/rag-engineering/03-ingestion.drawio" download>Editable draw.io</a></div>
</figure>

## Model the lifecycle before choosing a worker

Use separate identities for the logical document, its content version, its parsed representation, and its index build. They answer different questions:

<div class="rag-table rag-table--header" role="region" aria-label="Model the lifecycle before choosing a worker comparison" tabindex="0">

| Identity | Example | Question answered |
|---|---|---|
| Document | `atlas-support` | Which source is being updated? |
| Source version | `support-v2` | Which source contents were used? |
| Representation | `support-v2:parser3:chunk2` | How were those contents transformed? |
| Index build | `support-v2:profile-b:build7` | Which searchable representation was published? |

</div>

These example IDs are readable teaching labels, not a proposed wire format. Real systems can use UUIDs or digests, provided relationships remain explicit.

A document can be accepted while its new representation is not yet searchable. Expose that state to operators. Otherwise an upload success message can be mistaken for a freshness guarantee.

## Make retries identify the same work

A byte-level content hash identifies identical input. A transformation hash identifies the parser, chunker, and effective settings used to interpret it. The useful idempotency key is a tuple scoped to the logical source:

```text
(collection_id, document_id, content_digest, transformation_digest)
```

Do not write this as addition of two hashes. Their relationship is a composite identity, and its scope matters. Two teams uploading identical bytes do not thereby grant each other access or create the same logical document.

An illustrative transformation manifest is:

```json
{
  "parser": {"name": "markdown", "revision": "3"},
  "chunker": {"name": "section_recursive", "revision": "2"},
  "size_unit": "characters",
  "chunk_size": 800,
  "chunk_overlap": 100,
  "normalization_revision": "1"
}
```

Canonicalize the serialization before hashing it: stable field order, explicit defaults, and a defined encoding. Include changes that affect output. A configuration hash that omits a parser upgrade cannot explain why identical bytes produced different chunks.

Idempotency also needs concurrency control. Two uploads may pass an existence check simultaneously. Enforce uniqueness in storage and handle the losing transaction by resolving the existing work item. A preliminary `SELECT` alone does not guarantee one representation.

## How the implementation identifies a version

The ingestion service records parser and chunker configuration separately from the uploaded content hash. A repeated upload is looked up using document identity, content hash, and pipeline hash, so changing chunk settings represents different processing work even when the file bytes stay the same.

Excerpt from [service.py](https://github.com/binzhango/local_lke/blob/main/src/local_lke/ingestion/service.py). This is part of the application, not a standalone script.

```python
def hash_pipeline_configuration(configuration: Mapping[str, object]) -> str:
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
```

[Configuration fields included in the hash](https://github.com/binzhango/local_lke/blob/main/src/local_lke/ingestion/service.py) · [Version lookup and duplicate handling](https://github.com/binzhango/local_lke/blob/main/src/local_lke/ingestion/service.py)

There are **two publication boundaries in this implementation**. The ingestion transaction stores elements and chunks, marks the new document version active, and supersedes the old version. Vector indexing runs separately and activates its nodes and embedding profile afterward. This does not provide one atomic switch across ingestion and every search representation.

[Read the ingestion transaction](https://github.com/binzhango/local_lke/blob/main/src/local_lke/storage/repository.py) · [Read the separate vector build and activation](https://github.com/binzhango/local_lke/blob/main/src/local_lke/indexing/service.py)

The complete-build publication pattern below is an operational extension to that implementation. It explains the stronger guarantee to aim for when readers must never observe a source update before every required index is ready.

## Publish a complete build

A practical design stages a new representation while the previous publication remains active:



This is a **proposed lifecycle**, not a claim that every RAG framework implements it. During preparation, version 1 remains the published target. Once all required artifacts for version 2 are durable and validated, a transaction changes the published pointer.

For a database-backed design, the critical section might lock the document row, confirm the staged build is complete, compare the expected previous publication, and update the pointer. Readers resolve one publication ID and use it consistently throughout the request. [PostgreSQL's transaction-isolation documentation](https://www.postgresql.org/docs/current/transaction-iso.html) explains why concurrent visibility needs an explicit isolation and locking strategy.

External object storage and a separate vector service do not participate automatically in that transaction. Make artifacts durable first, publish a manifest referring to them, and use reconciliation to identify incomplete or orphaned work. Do not imply a distributed atomic commit merely because each storage system supports local transactions.

## Work through the interrupted update

Suppose version 2 needs three embedding batches. Batches 1 and 2 succeed; batch 3 times out.

<div class="rag-table rag-table--striped" role="region" aria-label="Work through the interrupted update comparison" tabindex="0">

| Event | Build state | Searchable publication |
|---|---|---|
| Version 2 accepted | Queued | Version 1 |
| First two batches persisted | Building | Version 1 |
| Third batch fails | Failed, partial artifacts retained | Version 1 |
| Retry completes missing nodes | Ready for validation | Version 1 |
| Completeness checks pass | Published | Version 2 |

</div>

Retry should avoid embedding already persisted nodes when their identities and profile are unchanged. That requires checking more than a job status: each node must identify its source representation and embedding profile.

Before publication, verify expected versus actual node identities, dimensions, finite values, and required metadata. A count match alone can hide one missing node and one unexpected node. After publication, a query should record which publication it used so the transition can be investigated.

If a later quality regression requires rollback, change the pointer to a retained compatible publication. If a deletion or access revocation requires immediate removal, override the availability preference: stop serving the revoked evidence even if its replacement is not ready. Keeping the old version searchable is a policy choice, not an unconditional reliability rule.

<figure class="rag-visual"><p class="rag-visual-title">What readers see during an interrupted update</p><ol class="rag-step-flow"><li><strong>Serve v1</strong><span>Queries keep using the published build.</span></li><li><strong>Stage v2</strong><span>A failed batch leaves v2 unpublished.</span></li><li><strong>Complete v2</strong><span>Retry missing work and validate the full build.</span></li><li><strong>Switch once</strong><span>New queries resolve the v2 publication.</span></li></ol><figcaption>A content update can retain the prior version during preparation. Revoked evidence needs a separate exclusion policy.</figcaption></figure>

## Treat parsers and uploads as bounded work

Validate an upload before invoking its parser: allowed formats, size, plausible byte signatures, and safe storage paths. Keep the submitted filename as metadata; generate the storage identifier. File-type checks do not establish that a document is trustworthy or that processing it is cheap.

Set limits on decompression, page count, parsing time, and resource use appropriate to the chosen parser. Retain structured warnings for partial extraction. [OWASP's file-upload guidance](https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html) provides controls for this boundary; the resulting policy still needs to match the service's supported formats and deployment.

A persistent job should expose queued, running, completed, and failed outcomes with a stable error category. For asynchronous workers, add leases or heartbeats so abandoned work can be distinguished from slow work. A lease expiry allows recovery; it does not guarantee the original worker has stopped. Writes must therefore remain safe under retries and duplicate execution.

Separate diagnostic detail from the public error. Operators may need parser traces in restricted logs, while clients need a stable failure code and actionable next step. Neither channel should accidentally expose credentials or another collection's document contents.

## Verify lifecycle guarantees directly

Exercise identical uploads, concurrent duplicates, a parser revision change, a failure between batches, and a restart during processing. Assert the published evidence identities at each step, not just the final job status.

Also test deletion during an in-flight query. Decide whether the request uses its initially authorized snapshot or must recheck access before response delivery. High-sensitivity use cases may require the latter. Record the policy explicitly so caches and citation endpoints do not follow a different rule.

The important operational measurements are publication lag, failed-build rate, retry work, and requests served from older publications. “Upload succeeded” is too early to measure freshness. The next article adds the embedding-profile contract that makes a published index interpretable by its query encoder.

