---
name: animate-system-diagrams
description: Create production-ready looping technical architecture animations with a refined dark-grid, restrained-neon systems-diagram aesthetic. Use for animated GIF, WebP, or MP4 diagrams for engineering blogs, product sites, launch pages, explainers, system flows, data pipelines, agent architectures, infrastructure diagrams, or when replacing generic slide-like Hyperframes output with crisp art-directed motion graphics.
---

# Animate System Diagrams

Create a diagram that explains a system first and looks impressive second. Use the bundled SVG/HTML renderer for deterministic, resolution-independent frames and FFmpeg for optimized exports.

## Workflow

1. Read `references/art-direction.md` before designing a new scene.
2. Inspect the surrounding article, destination width, theme, and existing assets. Extract one sentence stating what the animation must teach.
3. Reduce the story to 3–7 nodes and one primary path. Split genuinely distinct concerns into labeled groups; do not turn paragraphs into boxes.
4. Copy `assets/starter/` into a project-owned working directory. Never edit the bundled starter in place.
5. Edit `scene.json`. Keep labels short, preserve safe margins, and assign accents by semantic role rather than decoration.
6. Run `python <skill-dir>/scripts/check_scene.py <work-dir>/scene.json`.
7. From the working directory, run `npm install`, then `npm run render`. Use an existing compatible Playwright installation when the project already has one.
8. Inspect `output/contact-sheet.png` at actual display size. Revise overlaps, hierarchy, contrast, timing, and legibility before delivery.
9. Deliver GIF only when required. Prefer animated WebP for articles and MP4 for landing pages; always retain the MP4 master.

## Scene Contract

- Canvas: default 1600×900, 15 fps, 6 seconds.
- `groups`: large labeled regions behind the flow.
- `nodes`: cards with `kicker`, `title`, and optional `detail`.
- `edges`: directional connections; use `from` and `to` node IDs.
- `chips`: compact supporting capabilities, not primary steps.
- `sequence`: ordered node IDs that receive a traveling focus pulse.
- `packetDuration`: optional seconds for a routing dot to traverse one edge; use 1.5–2.5 for brisk flows.

Use `accent` values `blue`, `violet`, `green`, or `neutral`. Add colors only by extending the token map, never as ad hoc inline values.

## Motion Rules

- Keep geometry stable. Animate state, routing, and emphasis—not card positions.
- Use one dominant motion idea per loop: traveling focus, packet flow, staged reveal, or policy interception.
- Maintain a readable static frame at every moment.
- Make the loop periodic; avoid a visible reset or full-scene flash.
- Use 15 fps for GIF/WebP and 30 fps for MP4 only when fine motion demands it.
- Respect reduced motion when embedding a live HTML variant.

## Quality Gate

Reject the render if any check fails:

- The lesson is not understandable within three seconds.
- Body text is unreadable at the intended article width.
- More than two elements compete at the highest emphasis.
- Arrows cross cards, labels collide, or glow obscures text.
- The first/last-frame transition visibly jumps.
- GIF exceeds 8 MB without a documented reason; prefer WebP or MP4.
- Animation conveys decoration but no sequence, state, or causality.

## Resources

- `assets/starter/`: reusable renderer and example scene.
- `scripts/check_scene.py`: structural and layout validation.
- `references/art-direction.md`: visual system, composition, and timing guidance.
