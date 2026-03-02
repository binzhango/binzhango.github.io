# Mobile Compatibility Rollback Tasks

## Goal
Roll back custom component overrides one by one to align with Astro Starter defaults and isolate/fix mobile view regressions while preserving only necessary custom behavior.

## Scope
- Project: `nuclear-nova`
- Default baseline: Astro Starter (this project uses the Starlight starter integration).
- Current status: Astro Starter-equivalent defaults are restored for `Header`, `Search`, `ContentPanel`, and `Footer`.
- Active overrides kept:
  - `SiteTitle` (minimal nav links: Home/Blog/Archive)
  - `Head` (kept due metadata/SEO regression with default `Head`)

## Test Matrix (Run Every Step)
- Viewports: `375x812`, `390x844`, `430x932`, `768x1024`
- Routes:
  - `/`
  - `/archive/`
  - one post with long content
  - one `/tags/...` page
  - one `/categories/...` page

## Acceptance Criteria
- No horizontal overflow on key pages.
- Header/nav/search interactions work on mobile.
- Content remains readable without clipped text/components.
- Footer wraps cleanly and links/icons remain tappable.
- Build succeeds (`npm run build`).

## Execution Checklist

1. Baseline capture
- [x] Run `npm run build`.
- [x] Record baseline screenshots/notes for all matrix routes and viewports.
- [x] Log current known issues before next rollback.

2. Rollback Step A: `Search`
- [x] Remove `Search` from `components` in `astro.config.mjs`.
- [x] Run `npm run build`.
- [x] Re-test all matrix routes/viewports.
- [x] Record before/after differences.
- [x] Keep if improved or neutral; otherwise revert only this step.
- [x] Committed.

3. Rollback Step B: `ContentPanel`
- [x] Remove `ContentPanel` from `components` in `astro.config.mjs`.
- [x] Run `npm run build`.
- [x] Re-test all matrix routes/viewports.
- [x] Verify post metadata/content layout is acceptable.
- [x] Keep if improved or neutral; otherwise revert only this step.
- [x] Committed.

4. Rollback Step C: `Footer`
- [x] Remove `Footer` from `components` in `astro.config.mjs`.
- [x] Run `npm run build`.
- [x] Re-test all matrix routes/viewports.
- [x] Verify spacing/wrapping/tap targets.
- [x] Keep if improved or neutral; otherwise revert only this step.
- [x] Committed.

5. Rollback Step D: `Head` (final, controlled)
- [x] Capture required metadata/features currently provided by custom `Head`.
- [x] Remove `Head` from `components` in `astro.config.mjs`.
- [x] Run `npm run build`.
- [x] Re-test all matrix routes/viewports.
- [x] Compare generated page metadata (canonical, OG, Twitter, schema, theme behavior).
- [x] Keep if improved or neutral and metadata is acceptable; otherwise revert only this step.
- [x] Reverted only this step after metadata regression; custom `Head` restored.
- [x] Committed.

6. Final verification
- [x] Run `npm run build`.
- [x] Run `npm test` (optional but recommended).
- [x] Full mobile matrix pass with final screenshots/notes.
- [x] Summarize remaining issues and next minimal fixes.

## Results
- Mobile matrix summary (`20` checks each run): no horizontal overflow in `baseline`, `step-a-search`, `step-b-content-panel`, `step-c-footer`, `step-d-head`, and `final`.
- Footer behavior changed as expected with default `Footer` (`footerLinks` changed from `[3]` to `[0,2]` in matrix diagnostics depending on route/footer composition).
- Step D metadata comparison showed regression with default `Head`, so custom `Head` was restored:
  - `ogCount`: `10` -> `5`
  - `twitterCount`: `7` -> `1`
  - `articleCount`: `3` -> `0`
  - `hasJsonLd`: `true` -> `false`
  - `hasLightboxMarker`: `true` -> `false`
- Artifacts:
  - `analysis/mobile-check/*-summary.json`
  - `analysis/mobile-check/*.png`
  - `analysis/mobile-check/step-d-pre-head.json`
  - `analysis/mobile-check/step-d-post-head.json`

## Progress Log
- [x] Step 0 complete: `Header` reverted to default, build passed.
- [x] Step A (`Search`) complete: kept default.
- [x] Step B (`ContentPanel`) complete: kept default.
- [x] Step C (`Footer`) complete: kept default.
- [x] Step D (`Head`) complete: tested default, then reverted to custom due metadata regression.
