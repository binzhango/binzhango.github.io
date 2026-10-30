# Article reading style refresh

The shared article styles now give long posts a clearer hierarchy: a colored title area, readable section headings, 18px desktop / 17px mobile prose, a consistent 44rem reading measure, and real locally hosted Inter semibold and italic faces. Table containers share the prose width and margins; wide table contents scroll inside that column. Figures may use more room.

Callouts use the shared blue, teal, violet, amber, and rose theme tokens. Nested callouts avoid repeated frames and shrinking type. Lists preserve native document flow, with differentiated markers and leading terms. Table headers use sentence case and larger type; rows have clearer separation and narrow screens get a scrolling hint. Existing RAG table variants remain intact. Legacy body h1 headings use article section sizing rather than the homepage display size.

The LLM inference article also surfaces its existing key idea at the beginning, presents its processing sequence and two comparisons visually, and breaks the responsibilities table into shorter points with proper row headers. Its full embedded PDF and Show/Hide control are preserved. The technical argument and existing heading anchors remain intact.

## Verification

- Five representative routes checked at 390px, 768px, and 1440px in light and dark themes: 30 layout checks, no document overflow or detected MathJax errors in those states.
- Routes: LLM inference, Modern RAG Engineering, LLM Interview Questions, LangChain/LangGraph Q&A, and Model Training.
- Screenshots inspected for the title/takeaway, procedure, comparison panels, general and RAG tables, code, legacy headings, and expanded nested interview/checklist content.
- Mobile table keyboard scrolling, disclosure expansion, and code-copy success verified. The PDF disclosure is still native and the reader remains embedded.
- Existing tests: 19 passing. Astro check: 66 files, zero diagnostic errors, warnings, or hints. Production build: 178 routes. Git whitespace check clean.

This is a shared presentation refresh with targeted formatting in the LLM inference article, not a substantive rewrite or a visual inspection of every paragraph in every post. No deployment performed.

## Table alignment follow-up

The reported responsibilities table now has exactly the same left edge and width as its surrounding paragraph at 390px, 768px, and 1440px in both themes, with no page overflow. The same sizing is applied to Markdown table wrappers and authored RAG table scroll containers. All three interview table wrappers also match their surrounding content widths.

## Reading-flow refinement

The LLM inference walkthrough now uses an unboxed headline and a larger opening sentence, followed by one linked seven-step reading map. On mobile that map replaces the duplicate contents disclosure. Step headings separate their label from the title while retaining the original text and anchors. The attention/FFN comparison has one shared frame. The cache section uses the existing local seven-second animation with native play controls, a poster, and a caption instead of a second pair of cards. Playback is user initiated.

Verified all seven map destinations, keyboard navigation to a chapter, inline animation playback, and continued table/prose alignment at 390px, 768px, and 1440px in both themes; no page overflow. The 20-test suite includes a regression check for preserving heading text, anchors, and inline emphasis. Production build remains 178 routes. This version focuses the content-specific changes on the LLM walkthrough; it does not claim to improve reader retention without reader feedback.
