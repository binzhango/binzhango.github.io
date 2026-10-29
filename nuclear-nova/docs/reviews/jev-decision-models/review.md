# Jev decision-model article review

Article: `src/content/docs/posts/2026/10-29-decision-models-jev-langchain.md`.

## Reviewed scope

Read the local project README, concepts, evaluation guidance, recorded-results report and benchmark methodology. Traced the full path through `decisions.py`, `agent.py`, `domain.py`, `runner.py`, `config.py`, and the benchmark's case construction, requests, validation and summary logic. Inspected setup instructions, the primitive example and all three JSON recordings. `source-snapshot.json` pins these local file contents by SHA-256 at the current reviewed commit. The initial review used an uncommitted tree; the update below supersedes that snapshot.

Checked the current official LangChain TypeSafe integration, custom middleware guide, Building a Harness with Jev article, and TypeSafe Choice/Score references on 2026-10-29. Links are adjacent to the relevant article claims. The article distinguishes official interface descriptions, source-verified behavior, invented arithmetic, scripted fixtures, recorded live calls and suggested future evaluation.

## Validation

- `npm run build`: passes; 184 generated pages and the existing card integration check pass.
- Source project: `.venv/bin/python -m pytest -q -p no:cacheprovider` with bytecode disabled: **88 passed**.
- Ran the scripted `missing-item` comparison in an isolated temporary data directory: both paths completed, each passed outcome checks and issued one simulated $9 refund.
- Compiled every Python code block. Executed the standalone invocation with its network method mocked to return the recorded triage response; checked the exact submitted task, fixture and questions. **No new live inference or benchmark was run.**
- Recomputed the benchmark summary from every recorded row using the source's `summarize`: exactly matches the saved summary. Median latency ratio is 73.7453, rounded to about 74 in the article. Checked the example score arithmetic.
- Exported both diagram themes through official draw.io `Graph.getSvg`. Each retains 14 editable vertices and 12 edges; renderer and output hashes are in `tools/drawio/jev-decisions/export-receipt.json`.
- Browser QA at 1440 × 1000 and 390 × 1000 in light and dark themes: images loaded, correct theme variant visible, no page JavaScript errors, no page-level horizontal overflow, all headings have anchors, and all five tables render as semantic tables inside focusable scroll regions. Checked keyboard focus and horizontal scrolling. Screenshot review covered the title/opening, diagram, code/currency paragraph and benchmark table. Screenshots remain in `/private/tmp/jev-blog-qa/`; compact metrics are saved in `visual-check.json`.
- Fixed paired currency amounts being interpreted as math, a diagram edge-label overlap and excessive first-column width in the mobile benchmark table. Final output contains zero unintended MathJax nodes.

## Publication follow-up

The article is added to the local blog and previewed; no commit, push or deployment was performed. Its GitHub source links now pin `7151a23e2e95b58feba3de86c51ee8e928dd656a`, which matches the local `origin/feature_poc` tracking reference, rather than the older `main` branch.

Builds emitted Astro duplicate-content-ID warnings while refreshing cached entries (including the changed article on the second build); the inspected generated route renders the current article. This content task did not modify the site's collection loader or unrelated untracked files.

## Diagram correction

Replaced the assisted-only overview with two explicit paths after author feedback: normal LLM decisions versus Jev-assisted workflow/risk decisions. Kept the same local worker, shared policy, separate mutable state, human-review behavior and outcome comparison visible. The normal agent has no added classifier call. Regenerated both themes and editable sources; rebuilt and repeated desktop/mobile theme checks.

## Readability update

Added an article-scoped `post--jev` treatment in `src/styles/jev-article.css`. Desktop prose now uses a 672 px measure with 19 px type and 1.85 line height; mobile uses 17.5 px type and 1.8 line height. The header is unboxed, taxonomy is quieter, section headings are numbered, inline code uses a neutral background, and technical figures/tables retain the wider article width. Converted the existing responsibility list into definition rows and framed two existing paragraphs as asides. Split selected long paragraphs at sentence boundaries.

Compared normalized word tokens against the pre-edit Markdown: every article word is preserved in the same order. Python snippets and benchmark values are unchanged. The post route adds a class only for this article. Checked another existing RAG article: it retains its 18 px body and original header border. Rebuilt successfully and inspected desktop/mobile light/dark views, prose, definitions, code, tables, notes, images and overflow. See `readability-check.json` for measured layout results. Screenshots are in `/private/tmp/jev-readability-qa/`.

## Source update: 7151a23

Re-reviewed the two follow-up commits after the initial implementation: advisory triage naming/removal of the unused question, unchanged enforced refund gate, environment-configured and validated threshold, pinned integration version, bounded lenient benchmark scoring, first-pass Wilson intervals, local API access checks, and incremental persistence with legacy migration. Traced relevant source and tests; the clean source tree is recorded in `source-snapshot.json`.

Updated the article and both diagram themes to distinguish advisory playbook hints from enforced refund review. Added post-hoc lenient results and uncertainty intervals, historical-recording qualifications, threshold/resume behavior, storage changes, dependency pinning and local access scope. All implementation/recording links now use commit permalinks. Existing readability styling remains intact.

Validation for this revision:
- Current source test suite: **124 passed** in 2.12 seconds, with bytecode and pytest cache disabled.
- Scripted missing-item comparison in a fresh temporary data directory: both paths completed with passing outcome checks and one simulated $9 refund; current Jev trace uses `playbook_hint`.
- Recomputed historical benchmark with `scored_report`: strict 17/18 vs 18/18, lenient 18/18 each; strict Wilson intervals 74.2427%–99.0125% and 82.4121%–100%; lenient 82.4121%–100% for both. Latency values unchanged.
- SHA-256 comparison confirms all three published recordings are unchanged from the original review. No new live inference was run.
- All article Python snippets compile.
- Site build: 184 pages and card integration checks pass. Desktop (1440 px) and mobile (390 px), light/dark checks pass with loaded images, no page errors, no page overflow and no unintended math rendering. Visually inspected the updated two-path diagram and mobile benchmark table. Refreshed both QA JSON reports.

## Style reset after author feedback

Removed the custom header, narrowed prose column, larger typography, numbered section treatments, extra spacing, callout panels, definition rows and boxed rubric styling. The article now inherits the site's standard article typography and layout. Retained only scoped diagram theme switching and accessible table/figure overflow styles. Converted the presentation wrappers to normal Markdown paragraphs and lists; normalized word comparison confirms no article wording changed.

The prior `readability-check.json` describes the rejected treatment, not the current styling. Current verification is in `visual-check.json`: site build and card check pass; desktop/mobile light/dark checks show no page errors, no page overflow, and correctly loaded figures/tables. Visually inspected desktop light and mobile dark prose/code screenshots.
