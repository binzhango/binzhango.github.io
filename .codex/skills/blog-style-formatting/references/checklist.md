# Engineering article review

## Content

- Opening explains the technology or mechanism directly, without generic audience framing or an invented anecdote.
- User-supplied local references were read beyond their introductions; the review scope is stated and prose/code differences are checked.
- Explanations teach mechanisms and decisions rather than list features or files.
- Inputs, intermediate states, outputs, assumptions, and failure cases are inspectable where useful.
- Implementation claims are checked against code; conceptual designs are labelled.
- Fictional data, calculated examples, measured results, and personal experience are represented honestly.
- Primary citations support adjacent claims; terminology is consistent.
- Readers can tell when an approach helps, when it fails, and what to measure.
- New articles and full revisions end with practical takeaways covering how to apply the approach, evidence-supported improvement, and concrete skip conditions or simpler alternatives. Recommendations follow from the body, with benchmark limitations and regressions retained.

## Presentation

- Every section received a density pass, including setup, integration caveats, results, and closing discussion; the quoted problem paragraph was not the only fix.
- Paragraphs develop one point, usually in a few sentences. Longer passages were reviewed for a meaningful split, without forcing arbitrary word limits or sentence fragments.
- A scan of headings and selective bold reveals the argument. Lists use short parallel labels and ordinary-weight explanations.
- Procedures, comparisons, qualifications, and causal explanations use formats suited to their purpose; the article is neither all prose nor all cards/lists.
- A useful mechanism visual appears early; short explanations, diagrams, comparisons, and procedures share the teaching work.
- Visuals reduce repeated prose instead of making an already dense article longer.
- Tables compare shared fields, keep words intact, and scroll inside a labelled region when needed.
- Inline code marks syntax; bold is sparse; quotations have attribution.
- Diagram tools follow the author’s preferences and retain the required editable sources.
- Diagrams use the appropriate renderer, reflect the stated architecture, and have readable labels and captions.
- Interactive figures have direct-open fallbacks; essential information is visible without interaction.
- Section accents vary purposefully where useful, remain consistent within a group, and carry no meaning that is absent from the text.
- Series navigation, titles, descriptions, excerpts, tags, and routes agree.
- Previous/next navigation preserves the established design and intended reading order.
- Takeaways have a distinct, restrained closing treatment, a semantic heading, and a readable mobile layout; they guide decisions instead of repeating a feature list or scoreboard.

## Verification

- Execute runnable examples and check calculated values and units.
- Build the affected site and run relevant existing checks.
- Inspect desktop and mobile, light and dark, tables, code, diagrams, and anchors.
- Confirm intended list-label/marker colors in the rendered page, including shared CSS overrides. Inspect prose-heavy sections, not just decorative components.
- Verify mobile diagram labels, active theme assets, image loading/layout reservation, and complete captions; reflow a simple diagram when scaling makes it unreadable.
- Check overflow and focus; build success is not visual review.
- Report changes, verification, and remaining material limitations.
