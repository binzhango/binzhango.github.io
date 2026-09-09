---
name: blog-style-formatting
description: Edit technical Markdown and MDX articles for clear engineering explanations, evidence-based examples, and readable typography, lists, tables, and diagrams. Use for blog editorial work and article-scoped presentation changes.
---

# Blog Style Formatting

Write for readers who want to understand a mechanism, make a decision, or debug a failure. Use diagrams, worked comparisons, and concise explanations together so readers can grasp the system before studying its details.

## Scope and workflow

For formatting requests, preserve the author's argument, examples, and technical depth. When the user authorizes a substantive rewrite, correct inaccurate claims, replace weak examples, and reorganize around the learning objective. Do not preserve an obsolete product mention merely because it appears in the draft.

Read the complete article and relevant linked references before editing. For implementation claims, inspect the actual code and identify its revision. Distinguish implemented behavior, proposed design, pseudocode, and measured results. Source code supports an explanation; file inventories alone do not teach a mechanism.

When the author supplies a local reference corpus, read the relevant full chapters, their examples, and linked implementation files. Record the exact reviewed scope; a file listing or reading only the introduction is insufficient. Compare prose with code because a tutorial may lag its implementation.

Identify the reader, mechanism, decisive example, and tradeoff. Open with useful technical substance: what the technology does and how it works. Avoid generic declarations about the audience, invented anecdotes, or a long statement of the series mission. Introduce specialized terms before relying on them. Pair short explanations with inspectable inputs, intermediate states, outputs, and boundary cases. Prefer the teaching sequence mechanism → small example or code → visible output → interpretation → tradeoff. Use it as a reasoning pattern, not identical headings for every article. Avoid introducing an extensive operational checklist before readers can follow the underlying operation.

Read [visual-patterns.md](references/visual-patterns.md) for formatting choices and apply [checklist.md](references/checklist.md) before delivery.

## Engineering voice and evidence

- Explain what changes, why it changes, when the choice applies, and what it costs. Replace vague claims such as “robust” or “production ready” with observable behavior and its limits.
- Write as a thoughtful engineer sharing knowledge. Do not invent personal experience, deployment history, measurements, or quotations to create authority.
- Label fictional fixtures and calculated examples. Include units, denominators, cutoffs, and assumptions for numerical claims. Keep synthetic comparisons distinct from benchmarks.
- Prefer primary papers and official documentation. Link beside the supported claim. A paper's result does not establish the same result for the article's application.
- Distinguish intermediate results from final output, validation from correctness, and a library capability from an application guarantee.
- Use established, searchable terminology in titles and descriptions. Avoid creating a technical category from a project's name.

## Project integration

Inspect the target site's existing components, styles, navigation, and project documentation before changing presentation. Reuse its tokens and conventions. Scope new styles to the affected article or series unless a site-wide change is requested.

Keep project names, stylesheet paths, CSS selectors, asset directories, and series-specific preferences in project documentation. Do not make them requirements of this reusable skill. Follow the author's source-link and diagram preferences for the current task.

## Emphasis and semantics

Use bold sparingly for a decisive term or conclusion. Inline code is for literal identifiers, commands, values, filenames, and syntax; ordinary technical concepts do not need code styling. Use blockquotes for attributed quotations. An author's own takeaway belongs in prose or a labelled aside.

Prefer Markdown for explanations and semantic HTML for useful visual structures. Use connected stage cards for a sequence, paired panels for before/after evidence, ranking boards for order changes, and labelled metric displays for calculated values. Keep heading levels sequential and core explanations visible without hover or animation. Replace prose already communicated by a visual instead of merely adding illustrations to a long text.

## Visual reading rhythm

Put a useful mechanism diagram or worked visual near the beginning. Break dense stretches with content-specific figures, comparisons, and procedures. Keep enough prose to explain why the visual matters; avoid both walls of text and a page made entirely of cards. Mark fictional inputs or calculated results briefly beside the example instead of leading the article with a disclaimer.

Preserve established navigation unless redesign is requested. Check that previous/next links follow the intended reading order and clearly identify their destinations.

## Diagram routing

Choose an available diagram tool from the author's preferences, the information being explained, and the required editability. A source tutorial's renderer does not determine the tool for the article. Apply the selected tool's skill when available.

- Use **archify** for system architecture, data flow, and explorable views. Follow its schema, validation, delivery, and screenshot-review workflow. Retain the JSON specification and delivery receipt. Keep the full-size view directly accessible from the article.
- Use **draw.io** for article-sized flows, decision paths, and transformations that benefit from deliberate layout. Author native shapes, connectors, and labels in `.drawio` XML; retain the editable source and export SVG through draw.io. A pasted screenshot or Mermaid object inside draw.io does not satisfy native editability. Reuse an existing project export workflow when available.
- Use **animate-system-diagrams** when requested motion or timing materially explains the mechanism. A static architecture needs no animation.
- Choose a source excerpt, ranking board, or calculated comparison when that explains the point better than a graph. Tool preference does not mean every visual needs boxes and arrows.

Figures need descriptive titles, scope captions, readable labels, and direct-open links for interactive embeds. Distinguish conceptual designs from code-verified diagrams. Verify arrows, trust boundaries, labels, and alternate paths against the explanation.

## Delivery

Build after content or presentation changes. Run checks relevant to changed behavior and execute examples advertised as runnable. Inspect desktop and mobile, supported light and dark themes, tables, code, anchors, figures, and overflow. Report actual verification and material limitations; build success alone is not visual review.
