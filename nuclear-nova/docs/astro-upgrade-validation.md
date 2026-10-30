# Astro upgrade validation

## October 30: Vercel browser-install timeout resolved

Production commit `42dbc572` timed out before Astro started. Its logs end after the Playwright 1.58.1 Chromium download reaches 100%; npm and `dnf` had already completed. This matches the upstream [archive-extraction hang](https://github.com/microsoft/playwright/issues/40724), fixed by [Playwright PR 40747](https://github.com/microsoft/playwright/pull/40747).

- Updated and pinned `playwright` and `@playwright/test` to 1.64.1, including the shared Mermaid renderer dependency.
- Changed the Node engine from `>=24` to `24.x`. The hosted preview confirms Vercel switched from Node 26 to Node 24; GitHub Actions already uses Node 24.
- Limited Vercel's browser installer to five minutes, logged Node/Playwright versions, and added a real Chromium launch check before the Astro build.
- On Node 24.22.2, browser installation/launch, 63 unit tests, 27 browser tests, Astro checks (zero errors/warnings; two existing hints), and the production build pass.
- A fresh [Vercel preview](https://vercel.com/binzhangos-projects/binzhango-github-io/8ajCvu1WZCDbWrinxxC7QrvvE3D7) reached READY on the actual hosted Linux image. Chromium extraction and launch both succeeded. This supersedes the hosted-build limitation in the original October 29 notes below.

Verified on October 29, 2026. The npm `latest` tag and [Astro's upgrade documentation](https://docs.astro.build/en/upgrade-astro/) both identified **7.4.4** as the latest stable release.

## Migration

- Updated Astro from 5.17.1 to 7.4.4, with compatible MDX, Vercel, sitemap, check, Shiki transformer, and Open Graph image packages. The lockfile records the resolved versions.
- Added `@astrojs/markdown-remark` and explicitly selected `unified()` to retain the existing math, Mermaid, admonition, interview, and readability plugins. Astro 7 defaults to a different Markdown processor.
- Set `compressHTML: true` to retain HTML-aware inline spacing.
- Migrated the content schema to `astro/zod`, the two-argument `z.record()` API, and `z.url()`.
- Removed the obsolete `astro-og-canvas` `param` option. The current package infers it from the route filename, preserving the existing image URLs.
- Updated the card bundler's esbuild pin to 0.28.2 to satisfy Vite 8's optional peer dependency. A full `npm ls --all` now reports no dependency problems.
- Added Chromium installation to both GitHub Actions jobs because Mermaid rendering requires a browser on clean runners.

References: [Astro 6 migration](https://docs.astro.build/en/guides/upgrade-to/v6/), [Astro 7 migration](https://docs.astro.build/en/guides/upgrade-to/v7/).

## Verification

- Baseline: 63 unit tests passed; Astro check reported zero errors, zero warnings, and two unused-variable hints in card tooling.
- Updated Astro check passes on Node 26.13.0 and Node 24.22.2 (the CI Node major), with the same two existing hints.
- All 63 unit tests and all 27 existing Playwright tests pass. Browser coverage includes article navigation, PDF viewing, math, code copy, interview filtering/practice/storage, no-JavaScript reading, print, and light/dark layouts at 375, 768, 1024, and 1440 pixels.
- Production builds pass on both Node versions, generating 184 pages, sitemap output, social images, and Vercel output. The postbuild check verifies the card studio and its 12 local resources.
- A production browser smoke test passed for the homepage, post listing, archive, collection, Mermaid article, PDF article, and card studio. Sitemap files and an Open Graph PNG return HTTP 200. No browser JavaScript errors were captured.

## Remaining limitations

The dependency audit is not clean: the initial upgraded tree reports 30 advisories (3 low, 10 moderate, 15 high, 2 critical), compared with 50 in the original lockfile. Some concern existing test/rendering dependencies; the newer Vercel adapter also introduces advisory-bearing transitive packages. This is separate from the functional compatibility checks. A forced audit fix is inappropriate here because npm proposes incompatible downgrades for some packages.

Validation is local on macOS. GitHub Actions and remote deployment have not been executed by this change.

## GitHub Actions and Vercel follow-up

### Remote state inspected

- [GitHub Actions run 40975408797](https://github.com/binzhango/binzhango.github.io/actions/runs/40975408797) completed both validation and Pages deployment successfully for commit `5de53a84ebed4f1a5c0d58fb84db2209598b7e32`, before the Astro upgrade. Its logs nevertheless contain missing-Chromium rendering errors for four articles.
- [Vercel's production deployment](https://vercel.com/binzhangos-projects/binzhango-github-io/HusBuczK6d1Jp8bDKs2BsDmDoUx1) also reports success for that same pre-upgrade commit through GitHub's deployment/status APIs.
- The published LangChain article returned HTTP 200 on both `binzhango.net` and `binzhango.github.io`, but contained no Mermaid SVG images. The upgraded local output contains two.
- GitHub Pages is configured to deploy from Actions. The workflow's `binzhango-server` deployment branch matches its latest successful run. Its action versions exist and support GitHub-hosted runners.

### Deployment fixes

- `DEPLOY_TARGET=github-pages` disables the Vercel adapter in the Pages build. This preserves all 22 HTML redirects in `dist`; Vercel continues to use its Build Output API routing manifest. The Vercel adapter otherwise sets `build.redirects: false`, which leaves Pages without redirect files.
- Both GitHub jobs install Chromium and its Ubuntu dependencies before content rendering.
- `vercel.json` sets a reproducible `npm ci` install, followed by `scripts/install-vercel-browser.sh`. On Vercel Linux builds, the script installs Chromium's NSS/GBM dependencies with `dnf`, then downloads the headless shell matching the locked Playwright version. It uses the local CLI, avoiding any implicit package upgrade.
- The Vercel build image is [Amazon Linux 2023](https://vercel.com/docs/builds/build-image); Playwright's Ubuntu-oriented `--with-deps` command is therefore kept in GitHub Actions, not the Vercel installer.
- `postbuild` now checks rendered diagrams in all four affected articles, all 22 target-specific redirects, and the deployment's sitemap origin. These checks fail if diagrams or Pages redirect files are removed.

### Clean-build results and limits

In a temporary checkout containing the working copies of tracked source files and the new deployment scripts, with no initial `node_modules`, Astro cache, generated card assets, or browser cache:

1. `npm ci` passed using Node 24.22.2.
2. Astro check failed as expected with an empty Chromium cache, confirming the prerequisite.
3. The browser installer downloaded the required headless shell into an isolated cache.
4. All 63 tests and Astro check passed (zero errors/warnings; two existing hints).
5. The Vercel-target build passed, including card assets, four Mermaid articles, 22 manifest redirects, and the `https://binzhango.net` sitemap origin.
6. The Pages-target build passed, including card assets, four Mermaid articles, 22 HTML redirects, and the `https://binzhango.github.io` sitemap origin.

These were macOS builds on the CI Node major. The Vercel-only `dnf` branch has **not** been executed locally: Docker was unavailable and access to start it was not approved. Vercel's authenticated dashboard settings/logs were also unavailable. The project Root Directory must be `nuclear-nova` for its `vercel.json` to apply. No deployment or push was performed; a real Vercel preview and GitHub Actions run are still required to confirm hosted Linux behavior for the upgraded commit.
