---
title: "RAG Engineering 07: Query Planning and Safe Structured Retrieval"
date: 2026-09-14T12:00:00-04:00
author: BZ
description: "Translate questions into bounded retrieval plans, preserve metadata constraints, and verify SQL result semantics with a worked incident-table example."
categories:
  - AI ENGINEERING
tags:
  - rag
  - rag-engineering
  - text-to-sql
  - structured-data
  - query-planning
excerpt: "Translate questions into bounded retrieval plans, preserve metadata constraints, and verify SQL result semantics with a worked incident-table example."
---

<nav class="rag-series-nav" aria-label="RAG Engineering series"><span>RAG ENGINEERING / 07</span><a href="/collections/rag-engineering/">All 10 articles</a></nav>

A question about a policy needs relevant passages. A question about average acknowledgement time needs a calculation over eligible records. Retrieving a few similar rows and averaging them answers a different question.

Query planning chooses the required operation and carries the user’s filters into it. Here we build that distinction around a small incident table and a bounded SQL plan.

<!-- more -->

## Route by the operation the answer requires

<figure class="rag-diagram rag-drawio">
<p class="rag-visual-title">Choose the operation the question requires</p>
<p class="rag-diagram-hint">Scroll horizontally to follow the full diagram →</p>
<div class="rag-diagram-scroll" role="region" aria-label="Choose the operation the question requires; scroll horizontally on small screens" tabindex="0">
<img class="rag-figure-light no-lightbox" src="/diagrams/rag-engineering/07-routing.svg" alt="Nearest passages are not a complete population for an aggregate calculation." width="810px" height="845px" loading="lazy" />
<img class="rag-figure-dark no-lightbox" src="/diagrams/rag-engineering/07-routing.dark.svg" alt="Nearest passages are not a complete population for an aggregate calculation." width="810px" height="845px" loading="lazy" />
</div>
<figcaption>Nearest passages are not a complete population for an aggregate calculation.</figcaption>
<div class="rag-figure-links"><a href="/diagrams/rag-engineering/07-routing.svg" target="_blank" rel="noopener">Open full size</a><a href="/diagrams/rag-engineering/07-routing.drawio" download>Editable draw.io</a></div>
</figure>

A router does not need to be a general autonomous agent. It can select from a small set of explicit operations:

<div class="rag-table rag-table--column" role="region" aria-label="Route by the operation the answer requires comparison" tabindex="0">

| Request | Required operation | Suitable evidence |
|---|---|---|
| “What is the P1 acknowledgement target?” | Find a policy passage | Versioned document section |
| “Does that target apply during maintenance?” | Find rule and exception | Related passages with scope |
| “Average acknowledgement time by team” | Aggregate eligible rows | Typed query result |
| “Which service depends on this component?” | Traverse known relationships | Graph or relational dependency records |

</div>

Do not advertise graph retrieval just because a router has a `graph` label. It requires an implemented backend, a schema, authorization, bounded traversal, and evaluation. An unsupported route should be explicit rather than quietly producing a vector-search approximation.

Some questions combine operations. “Which teams exceed the policy target?” requires both an authoritative policy and incident measurements. Preserve which part came from which source, and define the comparison carefully: an average below the threshold does not prove that every incident met it.

## Preserve metadata constraints as part of intent

Take the request: “Search PDFs created after July 31, 2026, using only page 3.” A semantic rewrite should not drop the date because it makes search harder.

This **illustrative plan** separates search text from eligibility conditions:

```json
{
  "query": "priority-one acknowledgement policy",
  "filters": [
    {"field": "media_type", "op": "eq", "value": "application/pdf"},
    {"field": "created_at", "op": "gte", "value": "2026-08-01T00:00:00Z"},
    {"field": "page_number", "op": "eq", "value": 3}
  ]
}
```

The timestamp assumes UTC. A production service must define the timezone and whether `created_at` means source creation, ingestion, or publication. Interpreting an ambiguous date silently can produce a perfectly valid filter for the wrong meaning.

Use allowlisted fields and operators, validate their types, and return the interpreted constraints for inspection. Unknown fields should not disappear quietly. If a user explicitly requested a filter, widening the query should require a documented interaction or policy, not a hidden fallback in the retriever.

Authorization constraints are different from optional user filters. They are application-owned and cannot be removed by a planner, rewrite, or fallback.

## Work through an exact aggregation

Use this fictional incident table:

```csv
incident_id,team,severity,ack_minutes
I-101,Platform,P1,10
I-102,Platform,P1,20
I-103,Payments,P1,12
I-104,Payments,P2,90
```

The question is: **For P1 incidents, what is the average acknowledgement time by team, highest first?**

The planner needs a severity filter, grouping by team, an average over `ack_minutes`, and descending order. Here is an **illustrative intermediate representation**:

```json
{
  "table": "selected_incidents",
  "filters": [{"column": "severity", "op": "eq", "value": "P1"}],
  "group_by": ["team"],
  "aggregations": [{"function": "avg", "column": "ack_minutes", "alias": "average_minutes"}],
  "order_by": [{"column": "average_minutes", "direction": "desc"}],
  "limit": 5
}
```

The application resolves `selected_incidents` to a table already selected and authorized for the request. It validates every column, aggregate, alias, and ordering reference before compiling an executable statement.

The following **SQL example** expresses the expected operation; `:severity` is a bound parameter supplied as `P1`:

```sql
SELECT team, AVG(ack_minutes) AS average_minutes
FROM authorized_incidents
WHERE severity = :severity
GROUP BY team
ORDER BY average_minutes DESC, team ASC
LIMIT 5;
```

The secondary sort makes equal averages deterministic. The calculated result is Platform at 15 minutes and Payments at 12 minutes. The P2 row is excluded. Without the severity filter, Payments would average 51 minutes—a valid calculation answering a different question.

That is why plan validation needs both structural tests and semantic examples. A type-safe query can still omit the most important business constraint.

## Compile a limited language instead of executing model text

<figure class="rag-visual"><p class="rag-visual-title">Keep model output away from raw SQL execution</p><ol class="rag-step-flow"><li><strong>Typed plan</strong><span>Allowed fields, operations, and filters.</span></li><li><strong>Validate</strong><span>Reject unknown identifiers and unsupported expressions.</span></li><li><strong>Compile</strong><span>Application-owned SQL with bound values.</span></li><li><strong>Execute</strong><span>Restricted role, timeout, and result limits.</span></li></ol><figcaption>The plan defines an operation; application code enforces what may run.</figcaption></figure>

An allowlisted intermediate representation is smaller than SQL. It can expose only selected projections, filters, aggregates, grouping, ordering, and limits. The compiler owns identifiers and operators; bound parameters carry values. [SQLAlchemy Core's expression language](https://docs.sqlalchemy.org/en/20/core/sqlelement.html) provides building blocks for this pattern.

Reject raw expressions, arbitrary function names, unselected tables, and unknown columns. Do not accept a free-form SQL string merely because it starts with `SELECT`. A read can expose unauthorized data or consume excessive resources even when it performs no writes.

Use a database role with the minimum read privileges, a statement timeout, result-size limits, and a bounded execution context. A row limit bounds output volume; it does not necessarily bound the work required to sort or aggregate millions of rows. Read-only transactions are another layer, not a replacement for a constrained plan and restricted permissions. [PostgreSQL's transaction settings](https://www.postgresql.org/docs/current/sql-set-transaction.html) describe the database-side controls.

### The compiler boundary in my implementation

The structured endpoint accepts a typed plan. Before compiling SQLAlchemy expressions, it resolves column names against the registered table:

Excerpt from [structured.py](https://github.com/binzhango/local_lke/blob/main/src/local_lke/retrieval/structured.py). This is part of the application, not a standalone script.

```python
def _compile_plan(table: StructuredTableResponse, plan: StructuredQueryPlan, max_rows: int) -> Any:
    sql_table = _sql_table(table.physical_name, table.columns)
    available = {column.name: sql_table.c[column.name] for column in table.columns}
    aggregation_aliases = {aggregation.alias for aggregation in plan.aggregations}
    referenced = {
        *plan.projections,
        *plan.group_by,
        *(item.column for item in plan.filters),
        *(item.column for item in plan.aggregations if item.column),
        *(item.column for item in plan.order_by if item.column not in aggregation_aliases),
    }
    unknown = referenced - set(available)
    if unknown:
        raise RetrievalError(
            f"Plan references unknown columns: {', '.join(sorted(unknown))}",
            code="invalid_structured_plan",
        )
```

An unknown column raises `invalid_structured_plan` before execution. The rest of [the compiler](https://github.com/binzhango/local_lke/blob/main/src/local_lke/retrieval/structured.py) checks aggregation types, builds filter predicates from typed values, and caps the requested row limit. It requests one extra row so the caller can detect truncation. The SQL example above illustrates the intended operation; it is not a raw model-generated query passed directly to the database.

## Data types and business definitions are part of grounding

CSV ingestion needs stable headers, explicit null handling, and recorded type decisions. The string `00123` may be an identifier rather than the integer 123. Dates can be ambiguous across locales. A blank acknowledgement time is not automatically zero.

For the example, `AVG` normally excludes null inputs. A report that omits missing acknowledgements may look better than actual operational performance. Return counts or missing-value summaries when they matter to interpretation. Define whether elapsed time includes weekends, whether reopened incidents count twice, and what period the query covers.

Schema grounding therefore includes more than column names. Include units, allowed values, source ownership, time semantics, and relevant metric definitions. Retrieve those descriptions when the schema is too large for every prompt, then validate the proposed plan against the actual selected schema.

The common “shortest item” failure is another useful test. Setting `limit=1` on a similarity search does not express a minimum. The plan must contain an explicit ordering by the relevant numeric field.

## Return evidence that can be inspected

A structured response should expose the interpreted operation, column names, typed rows, source version or snapshot, and warnings about incomplete data. A generated SQL preview can help an engineer debug the compiler, but ordinary readers usually need the applied filters and metric definition instead.

If a language model summarizes the result, evaluate the summary separately. It may round incorrectly, swap team names, or turn an average into a claim about every incident. The exact query result remains the evidence; the prose is another transformation to validate.

Test the valid aggregation, a missing filter, an unknown column, excessive limits, null values, and ambiguous time language. These cases show whether the system executes the intended operation, not merely whether it can generate plausible SQL.


