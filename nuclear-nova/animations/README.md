# System diagram animations

Project-owned sources for the animated architecture diagrams used by the blog.
Each directory contains a validated `scene.json` plus the shared SVG/Playwright
renderer copied from the `animate-system-diagrams` skill.

To rebuild one animation:

```bash
cd animations/<diagram-name>
npm install
npm run render
```

Generated files are written to `output/` and intentionally ignored. Publish the
resulting `diagram.gif` and `diagram.mp4` under `public/assets/images/2026/` using
the directory name as the asset basename.
