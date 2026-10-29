# Bin Zhang — Engineering notes

An interactive author card based on the blog's black-cat favicon, with typography and copy matching its engineering notes.
Created using the project-local RuiC-card-skill.

## Files

- `card.blend`: editable Blender scene with packed textures and animated card rotation.
- `assets/`: registered 1024 × 1536 artwork layers and original favicon reference.
- `card-config.json`: copy, layer depths, scale, and foil settings.
- `web/`: standalone Three.js viewer and exported `assets/card.glb` geometry.
- `renders/`: Blender renders.
- `verification/`: interaction checks and captured browser frames.

## Preview

```sh
npm --prefix ../.. run sync:card
node web/server.mjs
```

Open the printed local URL. Drag to tilt, scroll to zoom, select a finish,
adjust depth, flip the card, or download an image. Arrow keys also tilt the card.

The browser rebuilds the Blender material in GLSL. glTF carries the geometry and
material roles, not Blender's shader graph; the two renderers are not pixel-identical.

## Artwork provenance

Identity reference: `nuclear-nova/public/assets/favicon.ico` (256 × 256), supplied
by the user's blog. The cat cutout and background were generated with the built-in
image tool. The original favicon remains unchanged.

Subject prompt: preserve the cheerful front-facing black crayon cat, upright pink
ears, large white oval eyes, small pink nose, huge pink open smile and white teeth;
remove its original radial background; output a transparent portrait 1024 × 1536
cutout with typography space. No frame, text, foil, or glow.

Background prompt: opaque portrait crayon paper, saffron-yellow, coral-orange and
rose-pink rays around a quiet cream center, quieter top and bottom typography zones;
no character, frame, text, holographic sweeps, or glitter.

`prepare_layers.py` derives line art from the generated cutout, preserving exact
registration, and draws the fixed typography with installed Arial fonts. It also
creates `layer-preview.png`, a flat layer inspection aid (not the 3D export).

Blender is installed only under ignored `tools/`, using the RuiC installer's
SHA-256 verification. The primary download servers returned HTTP 403; the NLUUG
Blender mirror was used for the official release package and published checksums.

`customize_viewer.py` applies English labels to the copied skill viewer. Run it
after rerunning the skill pipeline, then rebuild `web/app.bundle.js` using esbuild.

The project copy of `verify_web.mjs` corrects one optional-layer check in the skill:
it excludes the unused 1 × 1 GPU placeholder when no effects artwork is configured.
The original skill report is preserved as `verification/skill-original-report.json`.
Run `node verify_web.mjs .` from this project for the full interaction checks.

## Blog integration

The homepage renders the GLB card directly into a transparent canvas using `web/home-card.js`. It shares `card-shaders.js` with the full editor. Pointer movement tilts the card without pressing a mouse button; clicking opens `/card-studio/`. The fallback uses the transparent artwork layers, never the Blender render with its dark surround, and also responds to hover. Reduced-motion preferences disable hover motion. Offscreen previews pause rendering.

Effects depth defaults to `0.90` (the artwork currently has no separate effects overlay).

The blog owns the card build. `npm run dev`, `npm start`, and `npm run build` bundle the viewer and homepage renderer, then sync their assets. Three.js and esbuild are pinned in the blog's main package and lockfile, so deployment needs neither the standalone server nor a separate install inside `web/`. To refresh assets during an existing dev session, run `npm run sync:card`.

The editor is an Astro page at `/card-studio/`, with a canonical URL and the card description. A postbuild check verifies that the homepage link, editor document, scripts, and configured model/artwork assets exist in the deployment output. The optional server at port 4173 is only an isolated artwork preview; the integrated blog uses its own origin throughout.

Run homepage hover, transparency, fallback, navigation, mobile and reduced-motion regression checks against the blog preview:

```sh
node tools/favicon-card/verify_web.mjs http://127.0.0.1:4322/ --only homepage --out tools/favicon-card/verification/native-home
```
