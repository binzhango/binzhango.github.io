# Layout modes

## Auto

Use `{"layout":{"mode":"auto"}}` by default. The renderer selects `tree` when any node has multiple children; otherwise it selects `flow`.

Use `direction: "TB"` for decision trees and `direction: "LR"` for pipelines. Override the auto-selected direction when the destination aspect ratio benefits from it.

## Tree

Use `tree` for hierarchies, taxonomies, decision branches, fan-out, and parent/child explanations. A valid tree has exactly one root, one parent at most per non-root node, and `nodes - 1` edges. The renderer keeps each subtree contiguous, centers parents over their descendant span, and constrains smooth edges to the gap between levels. The order of a parent's edges controls the left-to-right or top-to-bottom child order.

```json
{
  "layout": { "mode": "tree", "direction": "TB", "top": 190, "gap": 42 },
  "nodes": [
    { "id": "question", "title": "What is the goal?" },
    { "id": "means", "title": "Compare means" },
    { "id": "rates", "title": "Compare rates" }
  ],
  "edges": [
    { "from": "question", "to": "means" },
    { "from": "question", "to": "rates" }
  ]
}
```

## Flow

Use `flow` for pipelines, lifecycle stages, request paths, and sequential processes. Nodes are placed by graph depth, so small branches still align without forcing a tree contract.

```json
{"layout":{"mode":"flow","direction":"LR"}}
```

## Manual

Use `manual` only when semantic grouping or unusual routing cannot be expressed by graph depth. Every node then requires `x`, `y`, `w`, and `h`. Groups and chips remain manually positioned in all modes; omit them when automatic layout is the priority.

For a dense source diagram that must retain many conditions or alternatives, set `density: "reference"` and use a taller canvas with stable category columns. Keep each column internally ordered, animate one column path at a time, and preserve the supplied taxonomy instead of compressing it into overview cards.

## Sizing controls

The optional fields `top`, `bottom`, `marginX`, `gap`, `nodeWidth`, and `nodeHeight` tune automatic placement. Start with defaults, render the contact sheet, and adjust only when labels or density require it.
