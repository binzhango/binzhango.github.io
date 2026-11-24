# Astryx Butter migration plan

Replace the blog’s shared interface with Meta’s Astryx components and Butter theme in light and dark mode. Preserve the articles, URLs, anchor links, reading order, and interactive examples. The first reviewable milestone is the homepage and one technical article rendered in both themes. The migration is complete when the archive, collections, interview bank, and standalone tools also pass the checks below.

## 1. Establish a baseline and prove the integration

- Work from the current Astro project in `nuclear-nova`. Account for the in-progress Model API post, explorer, and post-route edits in this checkout. Use the current working tree as the comparison baseline.
- Capture the homepage, a blog index, a collection, the prompt-to-response walkthrough, and an interview page in light and dark at 375 and 1440 px. Record initial JS and CSS transfer sizes for the walkthrough and the full interview bank. Record the generated routes and the sample articles’ heading IDs. The [previous validation report](cohesive-redesign-validation.md) identifies working behaviors to exercise again.
- Resolve compatible versions of React 19+, React DOM, the Astro React integration, Astryx Core, StyleX, and the Butter theme against Node 24 and the project’s Astro version. Pin the resolved set in the lockfile. Start with Astryx’s published core and theme CSS. Check the installed package exports before introducing build plugins. [Astryx installation](https://github.com/meta/astryx) · [Butter package](https://github.com/meta/astryx/tree/main/packages/themes/butter)
- Prototype a real Astryx button, link, and interactive disclosure in the shared shell. Verify static HTML output, keyboard behavior after hydration, and correct theme inheritance inside a dialog or popover. Provide the same theme context to independently hydrated React islands. Load React in the browser only where an interaction needs it.
- Wire Butter to the existing early theme initialization in `ThemeManager.astro`. Preserve the `starlight-theme` storage key, the current default, and stored light, dark, or automatic preferences. Use the documented theme boundary for the installed version. Confirm that its selectors coexist with the blog’s `data-theme` selectors for article media.

**Pass when** a production preview shows Butter at first paint, a keyboard user can operate the control, changing theme updates an open overlay, and the post text and links are present in the initial HTML.

## 2. Build the shared shell and first article

| Surface | Implementation |
|---|---|
| Colors and typography | Map `--canvas`, `--surface`, `--ink`, and the legacy `--sl-*` and `--clay-*` roles to Butter’s surface, text, border, and status tokens. Separate primary action fills from their foregrounds. Butter pairs blue with white in light mode and pale yellow with dark ink in dark mode. Use locally served Outfit for body and headings, JetBrains Mono for code, and Sarina for a short display element such as the wordmark. Preserve readable article sizing. [Butter definitions](https://github.com/meta/astryx/blob/main/packages/themes/butter/src/butterTheme.ts) |
| Header and footer | In `BaseLayout.astro`, replace shared navigation and actions with Top Nav, Link, Icon Button, and the theme control. Preserve skip navigation, accessible icon names, the active page, and mobile menu focus. Apply shared link and typography styles to the footer. |
| Homepage | Use Section and Heading for the hero, Card or Clickable Card for a single destination, and linked List rows for recent posts. Reading paths contain independent chapter links, so use a regular Card with separate links. Preserve the illustrated engineering card, its destination, and reduced-motion behavior. |
| First article | Use the prompt-to-response walkthrough to integrate the post header, article type, contents rail, seven-step reading map, code viewer, figures, and PDF disclosure. Use its real content as the design review surface. |

**Pass when** the homepage, article, and mobile contents are readable in both modes, every hero and chapter link still goes to its original destination, the PDF opens, and switching theme also switches the appropriate article artwork.

## 3. Migrate article rendering and preserve editorial variety

Handle plain Markdown, MDX components, and authored HTML through their actual render paths. Most post tables pass through `rehype-readable-content.mjs`; changing `Table.astro` alone would miss them. Use package components directly where Astro or React controls the markup. For content emitted as static HTML, map the native elements to Butter’s tokens and presentation at build time. Keep Markdown authoring syntax intact.

### Lists

| Content | Treatment | Detail to retain |
|---|---|---|
| Explanatory bullets | Disc at the first level, circle at the next | Hanging indent for wrapped text and enough spacing for multi-paragraph items |
| Ordered procedures and chapters | Decimal markers and real ordered lists | Authored start values, nested steps, numbering, chapter links |
| Published checklists | GFM task-list treatment with checkboxes | Checked, unchecked, and read-only states as authored |
| Table-cell lists | Compact bullets or numbers | Existing nesting, links, and narrow-column readability |
| Recent posts and archives | Markerless List rows with dividers | Excerpt space on the homepage, compact date and title rows grouped by year in archives |
| Article bylines and benchmark facts | MetadataList for label–value pairs | Horizontal byline where space allows, aligned columns for measurements |
| Tags and topics | Linked badges with wrapping | Use OverflowList only in constrained toolbars, with every hidden link reachable |
| Heading navigation and hierarchies | Outline for article anchors, TreeList for genuinely expandable branches | Depth, link destination, keyboard access, and current location |

Map explicit classes, HTML semantics, task markers, and nesting before applying the default article list style. Keep the Hindsight takeaways as a muted summary, a set of measured facts, and a separate guidance list. [Astryx List variants](https://github.com/meta/astryx/blob/main/packages/core/src/List/List.doc.mjs)

### Blocks, code, tables, and media

| Content | Implementation and acceptance check |
|---|---|
| Prose sections and quotes | Use Heading and Stack for reading flow, a muted Section for an authored takeaway, and Blockquote with attribution for a quotation. Preserve ambiguous legacy Markdown quotes as authored. Map explicit note, tip, and warning directives by kind. Recognize existing question-note styling so a question keeps its relationship to its answer. |
| Bold in paragraphs | Markdown `**bold**` uses violet. Apply a specific role with `<strong data-emphasis="process">phrase</strong>`. Use `concept` for violet, `process` for teal timing or behavior, `data` for blue values or interfaces, and `caution` for amber limitations. Use the same role for related phrases. The colors follow Butter in both themes. |
| Code | Use inline Code for identifiers. Use the Affect UI tray and inset code panel for article fences and component-authored examples with Shiki’s Butter syntax colors in both modes. Give a single command a compact copy action. For multiline listings retain language or filename, line numbers, Wrap, Expand, and Copy. Preserve indentation, blank lines, diff markers, highlighted lines, Escape, and focus return. One toolbar owns each block. |
| Tables | Use Table for data-driven components and match its visual treatment on Markdown and authored RAG tables. Preserve table headers, row headers, nested cell content, labelled keyboard-scroll regions, overflow hints, and existing sticky columns. Test a wide table on a phone and a short table on desktop. |
| Evidence and media | Use Card or Grid for discrete worked examples and comparisons. Keep diagram labels, chart encodings, captions, MathJax, and Mermaid rendering. Before replacing GLightbox, verify touch navigation, captions, zoom, authored image links, focus return, and theme-specific picture sources. Carry forward any behavior the installed Lightbox needs help providing. |

**Pass when** the FastMCP checklist, an older LangChain question/answer post, a RAG table-heavy article, and a Hindsight MDX article each retain their distinct structure. Compare copied code with source text and inspect an equation and an authored nested list with JavaScript disabled.

## 4. Implement disclosures and sidebars by purpose

| Surface | Desktop | Narrow screens and interaction |
|---|---|---|
| Blog posts by year | Sticky left SideNav on articles, the blog index, archive, categories, and tags. Astryx’s SideNavCollapseButton retracts the panel horizontally and remains available to reopen it. Search by title or year, or expand individual years to browse posts in reverse publication order. Open the current article or archive year by default, or the latest year on other pages. | Below 721 px, show a closed “Posts by year” disclosure above the content. The search field, year groups, and post links remain available when expanded. |
| Article contents | Sticky Astryx TreeList in the existing right rail. Group anchor links by heading depth, show hierarchy guides, and let readers collapse branches. Highlight the current location. | At the existing 1024 px transition, show the same tree in an inline “On this page” disclosure. Use a trailing chevron and an initially closed state. |
| Interview topics | Compact linked rows with a topic label and question count. Derive the links from the same topic data used by the question bank. | Provide the same counted topic links in a “Question topics” disclosure. Keep direct question and legacy topic hashes working. |
| Collection notes | A labelled Text Input over a divided list of chapters. Selecting a row updates the adjacent preview, selected treatment, and URL hash. | Below the current 768 px transition, stack the picker above the preview. Retain arrow keys, Home/End, filtered previous/next, and the empty state. |
| Single optional block | Use a leading chevron and descriptive label for a PDF, payload, or extended explanation. Set the desired initial state explicitly. | Retain native `details/summary` for static, progressively enhanced content. Ensure the control remains reachable and the revealed content fits the viewport. |
| Interview answer groups | Use the divided, multiple-open CollapsibleGroup treatment. One owner controls expansion, filtering, and bulk actions. | Keep multiple answers readable together, “Expand all” scoped to filtered questions, and print behavior that exposes the answers and restores prior state. Preserve the native disclosure fallback. |
| Browse menus | Year and Topic open navigation menus. “Browse tags” expands linked badges inline. | Preserve current selection, keyboard opening, Escape, focus return, and reachable destinations. |

The left SideNav browses the blog across pages. The right TreeList links to headings within the current article. `BlogShell.astro` places both rails around the reading column. [TreeList](https://github.com/meta/astryx/blob/main/packages/core/src/TreeList/TreeList.doc.mjs) · [Collapsible](https://github.com/meta/astryx/blob/main/packages/core/src/Collapsible/Collapsible.doc.mjs) · [SideNav](https://github.com/meta/astryx/blob/main/packages/core/src/SideNav/SideNav.doc.mjs)

**Pass when** both sides of the sidebar breakpoints work, headings and topics remain reachable by deep link, keyboard focus stays visible, and disclosures announce their names and expanded states in both modes.

## 5. Complete discovery pages, forms, and tools

1. **Discovery pages.** Migrate `BlogIndex`, `PostPreview`, `BrowseControls`, `Pagination`, category and tag badges, archive pages, and collection index/detail pages. Use Breadcrumbs for orientation, Timestamp for dates, and links for navigable rows. Verify first/last pagination states, collection chapter order, pinned posts, and filtered-result counts.
2. **Interview bank.** Use Segmented Control for Read/Practice, Field with Selector and Text Input for filters, Button for study actions, and Empty State for no matches. Preserve saved progress, ratings, reset, print, and keyboard focus. Render questions and answers statically and share interaction code across the bank. Benchmark filtering and transferred JS against the baseline before extending React rendering across hundreds of questions.
3. **Interactive articles.** Review the Model API explorer, Hindsight evidence and decision components, and RAG article styles. Apply Radio List or Tab List to the explorer according to the implemented keyboard model. Use appropriate Card, MetadataList, Table, and Section treatments for evidence and results. Preserve source links, response examples, and diagram meaning.
4. **Standalone tools.** The PDF reader has its own document and theme sync. Apply Butter to its toolbar, page-number field, disabled buttons, and zoom actions while preserving PDF.js rendering. Card Studio is generated from `tools/favicon-card/web/`; edit those sources and regenerate via the existing sync script. Migrate sliders, icon buttons, and front/back selection, then verify rendering, reset, and image export. Inventory standalone HTML diagrams under `public/tools/` and theme their control surfaces at their authoring source where available.
5. **Retire superseded styling.** Remove component-scoped colors, fonts, old toolbar styles, and replaced library imports once their routes pass. Search older posts and specialty stylesheets for remaining uses of compatibility variables before removing aliases. Compare emitted styles and transferred assets against the baseline.

## 6. Verify the migration and prepare it for release

Run these checks from `nuclear-nova` using Node 24. Use the existing scripts so post-build checks cover the deployment manifests, redirects, sitemap, diagrams, and generated Card Studio assets.

```sh
rtk npm test
rtk npx astro check
rtk npm run build
rtk env DEPLOY_TARGET=github-pages npm run build
```

Serve the static production build on port 4323, then run the browser suite in a second terminal. Update tests that intentionally name the old fonts or rely on replaced markup. Keep assertions for copy fidelity, focus, filtering, saved state, and accessible rendering.

```sh
rtk npm run preview -- --host 127.0.0.1 --port 4323
rtk npx playwright test --config scripts/interview/playwright.config.ts
```

| Review | Evidence to collect |
|---|---|
| Light and dark appearance | Matched screenshots at 375 and 1440 px for home, archive, collection, technical article, interview, and tools. Check tablet layouts at 768 px and the article rail at 1023/1024 px. Switch with the real theme control, reload, and inspect a dialog, a code block, a selected row, a disabled field, and topic badges. |
| Accessibility and resilience | Keyboard navigation, accessible names and expanded/selected states, visible focus, reduced motion, readable contrast, print, and a no-JavaScript reading pass. Wide tables and code may scroll within their own regions. Check for page-wide overflow. |
| Content and behavior | Compare generated routes, redirects, article headings and anchors with the baseline. Exercise nested and task lists, syntax and diff colors, code copy, lightbox, PDF navigation, collection search, and saved interview progress. |
| Runtime and release | Compare initial JS/CSS bytes and full-bank filter timings with the baseline. Investigate unexpected changes. Record the verified package versions, browser evidence, and any component that required a native adapter. Review each stage as a separate change so a regression can be isolated. |

The release review can use the component map above as its checklist. A surface is complete when its Butter treatment, content semantics, and reader interactions have been verified together.
