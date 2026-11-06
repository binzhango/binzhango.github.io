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

## Turn a dense passage into a reading sequence

First identify the distinct ideas, then choose their presentation. Adding bold to an unchanged wall of text rarely solves its density.

For example, a paragraph might combine a model's two roles, retrieval-model configuration, durable storage, and an inspection UI. Give those separate labels, keep each explanation short, and move sample-only settings into an aside. Use a two-column definition layout only when the ideas are parallel and fit comfortably; stack it on mobile.

Illustrative rewrite:

**Generation — one model, two jobs.** The application generates answers. The memory service uses the model to extract facts. Each role has its own configuration.

**Retrieval — separate models select evidence.** Embeddings represent meaning; a reranker rescores candidates. Selecting a chat model does not configure either one.

**Persistence — memory outlives the agent process.** Durable storage holds the memories. The inspection UI helps during development but is not required for each answer.

These labels expose the distinctions before the reader studies the details. Do not copy this exact set of labels into unrelated topics.

For a process, replace a prose inventory with a labelled sequence, for example: **resolve scope → submit source → extract facts → persist → inspect the result**. Keep input/output examples nearby. A diagram earns its space by showing dependencies or transitions that the labels alone cannot show.

Use descriptive subheadings when the reader's question changes: what the operation returns, how to run it, how to debug it, and what the measurement establishes. Avoid a heading for every paragraph and avoid repeating the same summary in prose, a callout, and a figure.

## Lists

Use ordinary lists for parallel content. Number only when order matters. Keep long explanations in prose instead of multi-paragraph bullets. Nest when hierarchy is meaningful. Modest marker color and spacing are sufficient; avoid boxes around each item.

Short comma-separated lists in sentences are acceptable. Use a visible list when readers must inspect items independently. Do not force every product into a chip. Group long inventories by capability and explain selection criteria.

Start comparable items with short, parallel labels: **Write:** ingest the source; **Read:** retrieve evidence; **Answer:** generate from that evidence. Keep explanations in normal weight. If a bullet grows into several sentences with separate arguments, promote that discussion into prose or a subsection.

### Accent roles

Use an intentional, restrained palette across a long article. One suitable mapping is violet for concepts, teal for procedures, blue for measurements, and amber for caveats. Adapt this to the site's existing palette and any explicit user preference; these exact colors are not mandatory. Use one accent per meaningful group, including its markers and leading labels, rather than a different color for every item.

Ordinary paragraph emphasis can remain the heading/text color. Preserve the site's distinction between links and editorial emphasis. Maintain contrast in light and dark themes, and use explicit words such as “Observed result” or “Limitation” so color never supplies the meaning alone.

Check computed styles on representative list labels and markers. Global rules, especially rules using `!important`, can make supposedly different groups render identically. Prefer scoped variables or narrowly scoped overrides over site-wide changes or escalating specificity everywhere.

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

For simple flows, prefer a compact mobile composition of the same graph over shrinking a wide desktop export. Preserve node meanings, edges, and ordering across variants. Reserve image dimensions or aspect ratios to prevent layout shifts; wait for visible images to load before screenshot review. Verify the active theme actually displays the matching asset and hides the alternate one—an intended CSS rule is not proof of the rendered result.

## Closing takeaways

Give the ending a recognizable closing treatment instead of repeating the preceding section's layout. A single restrained panel, a strong opening recommendation with a divider, or compact labelled guidance can work. Choose from existing site conventions; a custom component, card grid, accent color, or fixed number of items is not required.

Lead with the most useful decision. Follow with concise guidance on application, supported improvement, and when to skip the approach, as defined in the main skill. For a multi-tool article, grouping guidance by tool may read better than three generic boxes. Keep essential caveats beside benefits and make any repeated metric support a recommendation rather than dominate the section.

Use a real heading so the section remains discoverable through anchors and the table of contents. Emphasize a few action phrases, retain ordinary body text for explanation, and stack parallel groups on mobile. Check the transition from the previous section and into footer navigation. The ending should feel distinct without becoming a second hero or repeating the opening summary.

## Typography and scope

Reuse site text, link, border, and background tokens. Ordinary technical vocabulary uses the body font. Prefer weight and whitespace over gradients, glow, repeated accent panels, and decorative icons.

Scope component classes to the affected article or series using the site's conventions. Keep supplementary navigation visually subordinate to the title.

## Render checks

Inspect around 1280–1440 px and 375–390 px in light and dark themes. Horizontal scrolling belongs to the table or code region rather than the page. Sticky UI must not cover anchored headings. Verify figures load, labels are readable without hover, and focus is visible. Geometry checks complement screenshots; they do not establish visual quality.

Review representative prose-heavy sections as well as the title and figures. At mobile width, check whether the paragraph divisions and bold labels still make the argument easy to scan. Inspect distinct accent groups, full figure captions, and the transitions around code blocks. A clean overflow report does not catch repetitive typography, crowded diagram labels, or an oversized text block.

## Visual engineering explanations

Choose reusable components by information shape:

- Ordered steps for a sequence, with a layout that adapts to narrow screens.
- Paired panels for alternatives or before/after states, showing inputs and consequences.
- An example walkthrough connecting source input, transformation, and output.
- A diagnostic grid for distinct failure locations and inspection actions.
- Ranked lists for ordering changes, highlighting the same item across views.
- Metric or budget displays with units, denominators, and explicit assumptions.

Place each visual next to the explanation it replaces or clarifies. Do not repeat every detail in both. A diagram should show relationships; a card grid by itself is not an architecture diagram. Preserve the existing article footer navigation.
