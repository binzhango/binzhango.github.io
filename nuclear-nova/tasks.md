# Astro Starter Migration Tasks

## Requirement
Treat Astro Starter as the new default and migrate the blog away from Starlight integration.

## Scope
- Project: `nuclear-nova`
- Target: plain Astro Starter-style routing/layout with content collections
- Keep existing blog URLs and taxonomy pages working (`/posts/*`, `/archive/*`, `/tags/*`, `/categories/*`)

## Migration Checklist

1. Replace framework wiring
- [x] Remove Starlight integration from `astro.config.mjs`.
- [x] Keep core Astro integrations (`mdx`, `sitemap`, `vercel`) and markdown plugins.

2. Replace content pipeline
- [x] Replace Starlight `docsLoader/docsSchema` with Astro `glob` collection.
- [x] Define `blog` collection schema in `src/content.config.ts`.

3. Create Astro Starter page shell
- [x] Add `src/layouts/BaseLayout.astro`.
- [x] Add simple site header/nav/footer and theme toggle.
- [x] Keep global styling and lightbox/copy behavior.

4. Restore core routes without Starlight
- [x] Add home page `src/pages/index.astro`.
- [x] Add blog index page `src/pages/posts/index.astro`.
- [x] Add blog detail route `src/pages/posts/[...slug].astro`.
- [x] Keep archive/tags/categories routes and remove `StarlightPage` dependency.

5. Update utilities and metadata routes
- [x] Switch post utilities to `blog` collection and deterministic slug path generation.
- [x] Update OG route to use `blog` entries.

6. Dependency cleanup
- [x] Remove `@astrojs/starlight` from dependencies.
- [x] Remove Starlight-only plugin dependency that was no longer used.
- [x] Run `npm install` to refresh lockfile.

7. Validation
- [x] Run `npm run build` successfully.
- [x] Run `npm test` successfully.
- [x] Confirm generated routes include expected post/taxonomy/archive paths.

## Result Summary
- Framework migrated from Starlight integration to Astro Starter-style app structure.
- Existing blog URLs remain generated under `/posts/<year>/<slug>/`.
- Archive, tag, and category pages continue to build and link to migrated post routes.
- Build/test status: passing.
