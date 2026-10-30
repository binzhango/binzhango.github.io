# Shared article alignment follow-up

The security and governance article exposed a structural mismatch: selected prose,
tables, and code used a centered 44rem column while figures, comparison panels,
and series navigation used the full article width. Fixing one block type at a time
left the other types misaligned.

The `.post-content` container now owns the responsive 44rem reading width and
centering. Removed the selective top-level width rule and reset blockquote inline
margins explicitly. Figures, captions, navigation, custom comparisons, and ordinary
Markdown now inherit the same containing width. The title panel is unchanged.
Wide code, diagrams, and tables retain their internal scroll regions.

## Verification

- Measured visible top-level article block bounds on all 64 routes linked from the
  archive at 390px, 768px, and 1440px, in light and dark themes: 384 route/viewport/
  theme combinations, no differing column edges or widths (1px tolerance), and no
  horizontal document overflow.
- Inspected screenshots of the reported diagram/caption/heading transition on
  desktop, plus mobile diagram and table layouts in light and dark themes.
- Confirmed the focused mobile diagram responds to ArrowRight: its scroll offset
  advanced to 40px while its 608px content stayed inside a 348px scroll viewport.
- Production build succeeded (178 routes); all 20 existing tests passed.

These geometry checks cover top-level alignment, not a visual review of every
nested component or every position in every article.
