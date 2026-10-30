# Cohesive redesign validation — September 13, 2026

Local production preview: http://127.0.0.1:4322/

## Automated checks

- Vitest: 24 tests passing across six files.
- Astro check: 71 files, zero errors, warnings, or hints.
- Production build: 178 pages, successful.
- Git diff whitespace check: passed.
- New regression coverage: normalized topic accents, conservative RAG display-title splitting, and code-copy extraction preserving indentation, blank lines, literal escapes, and long lines.
- Existing tests cover pagination, excerpts, taxonomy, accessible content transforms, and the animation-frame fallback (including reduced motion, mobile, and offscreen cleanup).

## Browser checks

54 layout checks: nine routes at 390, 768, and 1440px, in light and dark themes. No page-level horizontal overflow found. Article reading width is 704px when space permits.

Routes: homepage; blog listing; collections index; RAG collection detail; RAG chunking; prompt-to-response walkthrough; RAG security; LLM interview; model training.

Checked interactions: native mobile contents disclosure and anchors; expanded code wrapping, original-line mode, Escape, focus restoration, and unchanged reading position; collection search, filtered previews, and empty state; page-two pagination (10 articles); embedded PDF Show/Hide with rendered pages; standard image lightbox opening and closing; formulas inside interview answer disclosures.

A temporary localhost server applied Content-Security-Policy script-src 'none' to production output. Homepage, listing, collections, and article content remained available. All ten collection previews remained visible; the JavaScript search control stayed hidden; code wrapped and contents disclosures worked natively.

Native homepage scroll-driven transforms were observed moving approximately 15.5, 31, and 46.5px, within the configured 16/32/48px bounds. Reduced-motion behavior was covered by CSS inspection and existing fallback tests; the operating-system setting was not changed.

Copy buttons reported success. Automation clipboard readback returned an empty value, so system clipboard paste fidelity was not independently confirmed; source-text extraction is regression-tested.

## Screenshots

Captured screenshots and layout JSON are in `/tmp/blog-redesign-2026-09-13/`:

- `home-desktop.png`, `home-tablet-dark.png`
- `blog-desktop.png`, `blog-desktop-dark.png`, `blog-mobile.png`
- `article-desktop.png`, `article-mobile.png`, `security-mobile.png`
- `home-collections-desktop.png`, `collection-tablet.png`, `collections-mobile-dark.png`
- `code-expanded-desktop.png`, `code-expanded-mobile-dark.png`
- `pdf-embedded-dark.png`, `math-mobile-dark.png`, `article-no-javascript.png`
- `layout-checks.json`, `no-js-checks.json`

This change builds on existing uncommitted redesign work. No deployment, content migration, article rewriting, or dependency changes were performed in this implementation pass.
