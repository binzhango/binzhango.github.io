# Article rendering review — September 9, 2026

Reviewed all 64 published article routes, starting with **From Prompt to Response**. The initial fixes and their validation scope are recorded below. A subsequent user screenshot identified an additional PDF presentation problem that the initial checks missed. Article URLs and technical arguments are preserved.

## Follow-up: desktop PDF presentation

The full PDF is embedded inside the article, collapsed by default behind a native “Show PDF preview” / “Hide PDF preview” disclosure. Expanding it opens a locally hosted PDF.js reader with scrolling, selectable text, page navigation, direct page selection, zoom, and fit-to-width controls. Hiding and reopening it retains the current page. The preview no longer links an image to a new tab.

The update callout and PDF control previously used different widths and margins. They now share a reading-width container. The browser's native PDF iframe produced a blank viewer in the in-app browser, so a dedicated reader route renders the actual 603-page document without relying on a browser PDF plugin. Its styles and PDF.js assets are isolated from the article. The arXiv link label also required an escaped colon to avoid interpretation as a Markdown directive.

Verified actual page rendering at desktop and mobile widths, next-page navigation, direct selection of page 20, zoom enlargement and fit-width restoration, keyboard disclosure toggling, and page retention after reopening. The 390px dark layout has no document overflow, and reader controls follow the article theme. JavaScript is required for the reader; a download link is provided when scripts are disabled. The article and native disclosure remain usable without scripts. This is not a claim that every PDF page was visually inspected.

After this follow-up: 19 tests pass; Astro check reports zero errors, warnings, or hints across 66 files; the production build generates 178 routes.

## Findings and repairs

| Area | Confirmed problem | Repair |
| --- | --- | --- |
| Reported article | Bullet lists started farther left than the surrounding paragraphs. | Shared prose and lists now use the same centered reading width. |
| Comparison tables | Mobile columns compressed identifiers into vertical fragments; bold headings could become dark text on a dark header. | Native, labelled keyboard scroll regions preserve readable column widths; header descendants inherit the header color. Existing RAG table treatments remain intact. |
| Checklists | Flex layout separated text, bold phrases, and nested lists into competing columns. | Normal document flow, aligned checkboxes, and less padding in nested mobile callouts. |
| Math | A malformed FFN dimension formula produced a MathJax error. Long inline formulas overflowed the page after answers were expanded. | Corrected the delimiter and constrained formulas to keyboard-scrollable containers. |
| Markdown and callouts | Missing blank lines after HTML blocks left prose, links, and inline code unparsed. Nested LangChain callouts closed prematurely. | Corrected block boundaries and nested fences; removed an unmatched interview fence. |
| Migrated formatting | Emoji shortcodes and highlight delimiters appeared literally. | Supported the existing emoji aliases in prose and replaced legacy highlights with semantic markup. Code examples remain literal. |
| Image viewing | The gallery included a hidden video fallback image. | Excluded media fallback images and preserved authored links and picture elements. |

## Validation

- All 64 routes checked at 390px and 1440px in light and dark themes, with populated disclosures expanded: 256 browser checks, no page overflow or rendered math errors.
- Seven representative articles checked at 768px in both themes. Screenshots inspected for the reported article, nested checklists, equations, tables, diagrams, and code.
- Generated output inspected across all 64 articles: 73 tables, 60 math containers, no remaining detected raw Markdown links, callout fences, or legacy highlight markers. Referenced local media paths were checked for existence.
- Keyboard disclosure operation, contents navigation, table scrolling, code-copy success, and lightbox opening/Escape dismissal checked. With scripts blocked, the reported article retained its content and keyboard-scrollable table (80px of native horizontal travel verified).
- `npm test`: 19 passing tests. `astro check`: 65 files, zero errors, warnings, or hints. Production build: 177 routes. `git diff --check`: clean.

Browser measurements cover every article; visual inspection samples representative sections rather than every paragraph. External destinations and the technical accuracy of historical article claims were not re-audited.

## Source content still unfinished

These are authoring gaps, not hidden or failed renders:

- **LLM Interview Questions** contains 134 empty answer placeholders.
- **Whitening transformation** contains only its initial heading.
- **Spark Dataframe window function** contains a reference and a short outline.

No answers or explanations were invented to fill these gaps.

## Local review

[Open the repaired example article](http://127.0.0.1:4322/posts/2026/03-07-prompt-in-llm/).

Review evidence is saved in `/tmp/article-audit/`, including `browser-after.json`, `render-after.json`, and before/after screenshots. No deployment was performed.
