---
name: blog-style-formatting
description: Refine technical or explanatory blog posts written in Markdown or MDX for readability without rewriting away the author's content. Use when Codex needs to reorganize dense paragraphs, preserve original meaning while improving structure, add restrained emphasis for terminology, convert heavy prose into clearer lists/tables/quotes, or make scoped article styling changes for better reading UI.
---

# Blog Style Formatting

Preserve the author's ideas, detail, and tone. Improve structure, rhythm, scanability, and article presentation without turning the post into a shorter summary unless the user explicitly asks for compression.

## Workflow

1. Read the full post before editing.
2. Identify the author's actual intent, core technical detail, and places where readability breaks down.
3. Prefer content-preserving edits first:
   - split overloaded paragraphs
   - turn repeated comparisons into lists or tables
   - pull the main takeaway into a short quote or lead sentence
   - standardize terminology formatting
4. Apply emphasis sparingly. Highlight terms and contrasts, not whole paragraphs.
5. Only change CSS when the current renderer makes the article harder to read.
6. Rebuild or validate rendering when the repo supports it.

## Content Rules

- Preserve technical substance. Do not remove important explanation just to make the post shorter.
- Keep the author's original position and nuance.
- Fix awkward sequencing when ideas appear in the wrong order.
- Break long paragraphs when a paragraph contains more than one main idea.
- Keep one paragraph focused on one point whenever possible.
- Prefer explicit section headings for concept shifts.
- Use bullets for enumerations, comparisons, repeated patterns, or multi-part answers.
- Use tables when the reader needs to compare responsibilities, stages, components, or tradeoffs.
- Use blockquotes only for short takeaways or framing lines.

## Emphasis Rules

- Use `**bold**` for key terms, stage names, and important contrasts.
- Use inline code for identifiers, model terms used as literal labels, commands, file names, and exact technical phrases such as `input_ids` or `autoregressive generation loop`.
- Avoid stacked emphasis like bold-italic unless the repo already uses it intentionally.
- Avoid loud highlight styles unless the user explicitly wants a more decorated look.
- If CSS controls emphasis styling, favor restrained underline, weight, spacing, or muted accent color over marker-style highlight backgrounds.

## Markdown and MDX Guidance

- Default to Markdown when standard structure is enough.
- Use MDX only when the post clearly benefits from custom components, callout cards, terminology chips, diagrams, or scoped interactive layout.
- If switching from `.md` to `.mdx`, preserve frontmatter, links, images, and heading structure.
- Keep MDX additions lightweight and editorial, not gimmicky.

## Styling Guidance

- Scope article-specific styling as narrowly as possible.
- Prefer typography, spacing, list treatment, blockquote treatment, and table readability improvements over decorative effects.
- Keep heading hierarchy visually clear.
- Make lists easy to scan.
- Make tables readable on mobile.
- Avoid global style changes when a post-level or article-level rule is enough.
- Preserve existing site design language unless the user asks for a broader redesign.

## Final Checks

- Confirm the revised post still contains the original technical content.
- Confirm terminology is consistent.
- Confirm emphasis is not visually noisy.
- Confirm lists, tables, and quotes improve reading flow instead of fragmenting it.
- If CSS changed, confirm the article still works in both light and dark themes when applicable.

For a reusable editing checklist and common transformation patterns, read [references/checklist.md](references/checklist.md) when the post is long, dense, or technically detailed.
