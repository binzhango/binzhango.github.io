# Art Direction

## Design signature

Aim for a precision-instrument interface: near-black navy canvas, barely visible grid, thin colored outlines, soft inner fills, compact uppercase monospace kickers, and restrained glow reserved for active state. The reference style feels expensive because most pixels are quiet.

## Tokens

| Role | Guidance |
| --- | --- |
| Canvas | `#050912` to `#07101c` |
| Grid | blue-white at 3–5% opacity, 72–96 px spacing |
| Primary text | cool white, never pure white |
| Secondary text | slate at 52–68% opacity |
| Borders | 1–1.5 px, 25–48% accent opacity |
| Blue | orchestration, routing, supervision |
| Violet | intelligence, model, agent reasoning |
| Green | policy, approval, validation, success |
| Neutral | external systems or inactive state |

Use one dominant accent and at most two supporting accents. A glow should be detectable, not foggy.

## Typography

- Use Inter/system sans for titles and details.
- Use ui-monospace/SFMono for kickers, group labels, chips, and small telemetry.
- Use sentence case for titles; uppercase only for small labels.
- At 1600×900, start around 28–34 px for node titles, 17–21 px for details, and 15–18 px for kickers.

## Composition

- Use an 80–100 px outer safe area.
- Make the primary flow read left-to-right unless the subject requires another direction.
- Use groups to express ownership or boundaries, not merely to fill space.
- Allow generous empty space; target 45–65% occupied canvas area.
- Align cards to a small number of horizontal baselines.
- Keep edges orthogonal or gently curved and visually behind nodes.

## Motion choreography

Treat the animation as a six-second sentence:

1. Establish the full system in a quiet baseline state.
2. Send a focus pulse through the primary route.
3. Briefly illuminate the consequence or governing layer.
4. Return to the baseline without a cut.

Use smoothstep/ease-in-out interpolation. A node focus should take roughly 600–900 ms; traveling packets should remain visible for at least 350 ms. Stagger neighboring events by 180–350 ms. Avoid bounce, elastic springs, scale pops, and perpetual breathing glows.

## Export guidance

| Format | Use | Target |
| --- | --- | --- |
| MP4 | master, landing pages | H.264, yuv420p, CRF 18–22 |
| WebP | modern blog articles | 15 fps, quality 70–82 |
| GIF | compatibility and Markdown | 12–15 fps, optimized palette |

Render at 2× CSS resolution when thin strokes look soft, then downsample with Lanczos. Check the result against both dark and light page surroundings.

