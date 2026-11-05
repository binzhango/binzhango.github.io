# Hindsight local-memory experiment — November 5, 2026

Extract this archive into a checkout of https://github.com/vectorize-io/hindsight.
The sample is placed at examples/lmstudio-memory-agents, preserving its editable
client dependency at ../../hindsight-clients/python.

Reviewed local workspace HEAD: 919885bb9a350ba0cf2650af4405ccba4897222e.
That revision adds local learning documentation. The sample and benchmark were
uncommitted local additions at measurement time; their exact files are included.
Upstream code base: ee3bd049ebe3b96b7d1120328429102100c1e28e.
Check out that upstream revision before running uv sync --frozen in the sample.
The running Docker server is pinned independently by digest in compose.yaml.

Install and load Muse Glimmer 30B UD Q6_K_XL in LM Studio, with API identifier
muse-glimmer-30b and a 16,384-token context. Start its server on port 1234.
See the sample README for setup, Docker networking on Mac, and optional auth.
Model downloads and dependency downloads may be needed on a fresh machine.
RTK prefixes in the original README are optional command-output filtering;
remove `rtk proxy` if RTK is not installed.

The final run is benchmarks/2026-11-05-local. Two excluded pilot runs are kept
for protocol transparency. All source facts and user identities are synthetic.
The archive includes only explicitly selected code, docs, lock/config files,
tests, and benchmark outputs. It excludes .env, caches, environments, and secrets.
.env.example contains demo defaults only. No new model benchmark was run when
preparing the blog; the blog reports the preserved experiment.

SHA256SUMS.json identifies all included original sample files. LICENSE.hindsight
is the upstream license accompanying the reviewed repository.
