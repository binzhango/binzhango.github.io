# RAG Engineering project conventions

These conventions apply to this series, not to the reusable blog-style skill.

## Content and sources

- Use the RAG Engineering title and established RAG terminology.
- Ground implementation excerpts in the author’s GitHub repository: https://github.com/binzhango/local_lke. Use file links to `main`, without commit labels or line fragments.
- Inspect actual implementation behavior before describing it. Separate conceptual recommendations from implemented guarantees.
- Do not substitute standalone Python demos for implementation excerpts or mention the excluded tutorial in public posts.

## Presentation

Styles live in `nuclear-nova/src/styles/rag-series.css`, scoped under `.post-content[data-series='rag-engineering']`. Reuse the site tokens.

Use existing components where they fit: `.rag-step-flow` for sequences, `.rag-compare` for alternatives, `.rag-evidence-demo` for evidence walkthroughs, `.rag-debug-grid` for failures, `.rag-ranking` for rank changes, and `.rag-budget`/`.rag-metrics` for labelled calculations. Tables use `.rag-table` plus a persisted variant: `.rag-table--header`, `.rag-table--striped`, `.rag-table--column`, or `.rag-table--minimal`. Distribute these with a shuffled rotation for variety, avoiding adjacent repeats. Do not randomize styles on page load.

Preserve full-title `.post-nav` footer cards and collection reading order. The compact `.rag-series-nav` is supplementary navigation.

## Diagrams

Maintain a visible mix: Parts 1, 5, and 9 use Archify-generated inline previews linked to interactive views; other existing figures use native draw.io. Choose tools by the mechanism and reading experience instead of defaulting every inline figure to one renderer. Do not substitute Mermaid unless the author changes this preference.

- Author/export workflow: `nuclear-nova/tools/drawio/rag-engineering/`.
- Draw.io assets: `nuclear-nova/public/diagrams/rag-engineering/`.
- Archify HTML: `nuclear-nova/public/tools/`.
- Figure classes: `.rag-drawio`, `.rag-figure-light`, `.rag-figure-dark`, and `.rag-figure-links`.
- Theme-paired images use `.no-lightbox` to avoid hidden gallery entries. Keep horizontal panning inside a labelled, keyboard-accessible region.

Retain editable sources, verify architecture claims, and inspect readable labels in both themes at desktop and mobile widths.

Draw.io layouts are authored per mechanism in `author-diagrams.py`: parsing rail, release lifecycle, contract comparison, parallel lanes, bounded loop, decision tree, validation cards, and trust boundaries. Keep these compositions distinct when updating figures.
