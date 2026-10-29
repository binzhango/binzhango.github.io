# Jev article diagram

Native draw.io shapes and connectors compare the normal LLM agent with the Jev-assisted agent. Both use the same worker and hard policy, with independent mutable state. Triage selects advisory prompt guidance without restricting tools; the refund risk score adds an enforced review condition at the configured threshold (default 1.0). Dashed arrows collect observations rather than merge business state. The diagram expands eligible refund handling; denial and unresolved/rejected review stop execution.

`author-diagram.py` regenerates the native source and manifest. It overwrites manual XML edits. Run from `nuclear-nova`:

```sh
python3 tools/drawio/jev-decisions/author-diagram.py
node tools/drawio/jev-decisions/render-diagrams.mjs /private/tmp/jev-drawio-viewer.js
```

Download the renderer runtime from `https://viewer.diagrams.net/js/viewer-static.min.js`. Exports use the official `Graph.getSvg` API. `export-receipt.json` records the runtime hash, SVG hashes and native cell counts. Each theme has 14 native vertices and 12 edges. Inspect exports in the article at desktop and mobile widths.
