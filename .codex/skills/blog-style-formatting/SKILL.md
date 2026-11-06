---
name: blog-style-formatting
description: Write and edit technical Markdown and MDX articles with short focused paragraphs, scannable emphasis, evidence-based examples, purposeful diagrams, and practical closing takeaways. Use for blog editorial work, readability reviews, and article-scoped presentation changes.
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

## Paragraphs and scanning

Use the same readable structure from the first draft through the final article; do not wait for the author to point out a wall of text. Give each paragraph one job and lead with its useful point. Usually two to four sentences are enough. Treat roughly 60–80 words, several separate claims, or a tall mobile text block as a signal to review the paragraph, not a mechanical word limit.

Split at changes in responsibility, operation, evidence, or consequence. Keep causal explanations connected; do not turn every sentence into a separate paragraph or every paragraph into bullets. Move independently comparable facts into a short labelled list or table. Show sequence and branching with a procedure or diagram. Put a local configuration, observed result, or important qualification in a concise labelled aside when it would interrupt the main explanation.

Make two reading speeds work: headings and selective bold should reveal the argument on a quick scan; adjacent prose, examples, and sources should explain it on a close read. Use a short bold lead phrase for list labels, followed by ordinary-weight explanation. Do not bold whole paragraphs or use a generic TL;DR as a substitute for restructuring the body.

For a full-article readability request, review every section, including setup, code introductions, integration caveats, benchmark interpretation, and the conclusion. Preserve the evidence and qualifications while reducing repetition. See the before/after pattern in [visual-patterns.md](references/visual-patterns.md).

## Closing takeaways

End each new article or full-article revision with a **Takeaways** section, before references and footer navigation. Preserve an explicitly requested alternative ending. For a narrowly scoped edit, leave unrelated sections alone rather than expanding the task to retrofit a conclusion.

Help the reader decide what to do after reading. Cover these questions in a compact form suited to the topic:

- **How to use it:** name the situation that calls for the approach and a concrete first action, configuration, or workflow. For multiple tools, explain each tool's role and when combining them helps.
- **What improves:** state the benefit supported by the article and the baseline it improves on. Keep benchmark scope, denominators, tradeoffs, and regressions beside the result. Separate measured gains from expected benefits; if improvement is unproven, say what to evaluate.
- **When to skip it:** name concrete conditions where the approach adds little value, introduces unacceptable cost or failure, or loses to a simpler alternative. Give that alternative when the article supports one.

Adapt these questions for conceptual articles into when to apply the idea, what it helps explain or decide, and where it stops being useful. Do not invent commands, measurements, or recommendations to fill a template. Every recommendation should follow from evidence or reasoning already developed in the body.

Write action-oriented guidance rather than a recap of section titles, a feature inventory, slogans, or a repeated benchmark scoreboard. Select the few decisions that matter; avoid introducing a new tutorial at the end. Make the closing section visually distinct while preserving the site's design language. See the closing-section pattern in [visual-patterns.md](references/visual-patterns.md).

## Visual reading rhythm

Put a useful mechanism diagram or worked visual near the beginning. Break dense stretches with content-specific figures, comparisons, and procedures. Keep enough prose to explain why the visual matters; avoid both walls of text and a page made entirely of cards. Mark fictional inputs or calculated results briefly beside the example instead of leading the article with a disclaimer.

For long articles, use restrained variation between section types. Reuse theme-aware accents to distinguish concepts, workflows, measurements, and caveats where helpful; keep the meaning consistent within the article. Do not give every list the same highlight treatment by accident, or cycle colors arbitrarily per bullet. Labels and typography must carry the distinction without color. Inspect actual rendered styles: shared list rules can override intended emphasis.

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
