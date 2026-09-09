# Article presentation patterns

Choose the format from the information's purpose. Combine diagrams, example panels, and short explanations to make an engineering blog easy to enter and scan.

| Reader need | Format | Style |
| --- | --- | --- |
| Understand mechanism or tradeoff | Connected prose | One developed point per paragraph |
| Inspect parallel alternatives | Unordered list | Native bullets, hanging indent, parallel phrases |
| Follow a procedure | Ordered list | Native numbers; action and purpose |
| Learn several short terms | Definition list when useful | Term followed by explanation |
| Compare shared attributes | Table | Clear headers, top-aligned text, restrained rules |
| Reproduce syntax or calculation | Fenced code | Language label; runnable or pseudocode status |
| Notice an assumption or limit | Prose or labelled aside | Subtle border and spacing, never color alone |
| Read an attributed quotation | Blockquote | Exact text within quotation limits and a source |
| Understand topology or branching | Dedicated figure | Tool chosen by purpose; caption and description |

## Lists

Use ordinary lists for parallel content. Number only when order matters. Keep long explanations in prose instead of multi-paragraph bullets. Nest when hierarchy is meaningful. Modest marker color and spacing are sufficient; avoid boxes around each item.

Short comma-separated lists in sentences are acceptable. Use a visible list when readers must inspect items independently. Do not force every product into a chip. Group long inventories by capability and explain selection criteria.

## Tables

Use tables when rows answer the same column questions. Explain the interpretation in nearby prose. Prefer three or four concise columns; move long explanations out of cells. Include units in headers and identify synthetic results.

Use a varied table palette rather than giving every table the same treatment. Inspect existing styles first: options may include an accent header, striped rows, an emphasized label column, and minimal horizontal rules. When the author requests random variety, shuffle suitable options at authoring time and save the selection in the content; avoid adjacent repeats and runtime randomness. Preserve legibility and semantics across variants. Reserve heatmaps and status colors for data with actual magnitude or status meaning.

Wrap Markdown tables in a labelled, keyboard-focusable scroll region:

```html
<div class="article-table" role="region" aria-label="Processing method comparison" tabindex="0">

| Method | Benefit | Limitation |
| --- | --- | --- |
| Batch processing | Amortizes setup cost | Delays individual results |
| Streaming | Produces results incrementally | Requires flow control |

</div>
```

Keep blank lines around the Markdown so the site creates a real table. Use automatic table layout with a readable minimum width and scrolling inside the region. Headers wrap normally. Never shrink columns until technical words fragment. Preserve native semantics, visible keyboard focus, and light/dark contrast.

Use row groups and `scope="rowgroup"` only when grouping improves comparison. Manually authored tables need correct row and column header associations. Never use tables to position prose, cards, or diagrams. For unusually wide data, offer a full-size artifact or a deliberate alternate view.

## Figures

Choose the renderer using the main skill's diagram guidance and the current project's requirements. When the author wants visual variety and has multiple diagram tools, use a deliberate mix across the articles. Make that variety visible inline, rather than relegating one tool to an external link. Choose by topology, editability, and interaction needs; do not randomize technical meaning or redraw every figure in one default renderer. Vary compositions within a tool as well: consider swimlanes for parallel actors, before/after panels for transformations, decision trees for routing, lifecycle loops for retries, and boundary containers for access control. Vary shape treatment, alignment, density, and restrained color emphasis where they clarify the subject. Do not reuse one rounded-box template for every figure or treat palette changes alone as layout variety. Preserve consistent semantic cues and readable typography.

Retain editable sources alongside generated assets using the project's existing directory conventions.

Introduce the figure's question in prose. Conceptual figures must not imply every component exists in the reference implementation. Label stores, transformations, data flow, and control flow consistently. Include failure or authorization paths when central to the lesson.

Interactive embeds need a titled iframe, caption, and direct-open fallback. A readable inline draw.io overview can link to the fuller Archify view; an iframe is not mandatory. Check text size in the article, not just the standalone viewer. If a dense diagram cannot stay readable at article width, simplify the inline overview and link the detailed view. Do not shrink labels to fit or autoplay a static explanation. Use separate light/dark exports when needed. For draw.io, check that exported source reopens as native editable nodes and edges. When a site uses theme-paired images, check its lightbox integration so hidden variants do not produce empty gallery links. If mobile needs horizontal panning, label the region and preserve keyboard access; never let it widen the page.

## Typography and scope

Reuse site text, link, border, and background tokens. Ordinary technical vocabulary uses the body font. Prefer weight and whitespace over gradients, glow, repeated accent panels, and decorative icons.

Scope component classes to the affected article or series using the site's conventions. Keep supplementary navigation visually subordinate to the title.

## Render checks

Inspect around 1280–1440 px and 375–390 px in light and dark themes. Horizontal scrolling belongs to the table or code region rather than the page. Sticky UI must not cover anchored headings. Verify figures load, labels are readable without hover, and focus is visible. Geometry checks complement screenshots; they do not establish visual quality.

## Visual engineering explanations

Choose reusable components by information shape:

- Ordered steps for a sequence, with a layout that adapts to narrow screens.
- Paired panels for alternatives or before/after states, showing inputs and consequences.
- An example walkthrough connecting source input, transformation, and output.
- A diagnostic grid for distinct failure locations and inspection actions.
- Ranked lists for ordering changes, highlighting the same item across views.
- Metric or budget displays with units, denominators, and explicit assumptions.

Place each visual next to the explanation it replaces or clarifies. Do not repeat every detail in both. A diagram should show relationships; a card grid by itself is not an architecture diagram. Preserve the existing article footer navigation.
