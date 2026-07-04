---
title: "Reproducing vLLM and LMCache KV Cache Reuse on a CPU-Only MacBook"
date: 2026-07-03
author: BZ
description: "A reproducible CPU-only vLLM and LMCache workspace for understanding KV cache reuse, transfer modes, and local inference debugging on Apple Silicon."
categories:
  - AI ENGINEERING
  - SOFTWARE ENGINEERING
tags:
  - vllm
  - lmcache
  - kv cache
  - cpu
  - uv
---

<!-- more -->

I became interested in LMCache because it sits in the part of LLM serving that feels both very practical and very under-discussed: **KV cache movement**.

Most LLM application work starts at the API layer. Prompt in, text out. But the expensive part of serving is not only model weights or tokens per second. During inference, every request builds a KV cache that represents the context the model has already processed. When requests share prefixes, such as the same system prompt, retrieved documents, or multi-turn conversation history, recomputing that KV cache is waste.

The LMCache team published [vLLM + LMCache: A Starter Guide, No GPU Required](https://blog.lmcache.ai/en/2026/06/23/vllm-lmcache-a-starter-guide-no-gpu-required/). The article caught me because it removes the usual blocker: I do not need a GPU just to learn the vLLM + LMCache integration path. A CPU MacBook is enough to run the core loop, inspect logs, modify code, and build confidence before touching a larger environment.

My reproduction repo is here:

[binzhango/reproduce_vllm_lmcache](https://github.com/binzhango/reproduce_vllm_lmcache)

This post is not a copy of the LMCache guide. It is my version of the learning path: what the guide made me curious about, which parts I reproduced, and what I want to understand next.

## Local Environment

For reproducibility, this is the MacBook environment I used:

| Item | Value |
| --- | --- |
| Machine | MacBook Pro |
| Model identifier | `Mac16,5` |
| Chip | Apple M4 Max |
| CPU cores | 16 total: 12 performance cores and 4 efficiency cores |
| Memory | 64 GB unified memory |
| Operating system | macOS 26.5.2 |
| Build | `25F84` |
| vLLM device mode | CPU backend only |

The hardware matters because the goal of this post is not to show a general production benchmark. It is to show that the core vLLM + LMCache integration path is reproducible on a developer laptop.

The Apple Silicon details also explain several settings later in the post: I force `VLLM_DEVICE=cpu`, keep the KV cache and LMCache L1 budgets small, and pin internal communication to loopback so the whole experiment stays local.

## What LMCache Is Solving

LMCache is a KV cache layer for inference engines such as vLLM and SGLang. The idea is simple to say and hard to execute well:

- The inference engine computes KV tensors for a prompt.
- LMCache receives those tensors through a connector.
- LMCache stores them in an L1 layer, and can also persist or coordinate them through lower storage layers.
- Later requests with matching prefixes can retrieve the KV cache instead of recomputing it.

That matters for workloads with repeated context:

- long system prompts
- RAG prompts with repeated document prefixes
- agent loops that keep reusing instructions and tool context
- multi-turn conversations where earlier context remains stable
- production systems where prefill latency dominates user experience

The part I want to understand deeply is not only "cache hit is faster." I want to understand the data path: how vLLM exposes KV, how LMCache stores it, how chunking works, how transfer modes differ, and how observability tells me whether the cache is actually being used.

## Why a MacBook Is Enough

The source guide's most useful point is that LMCache has a multi-platform design. GPU-specific operations are not scattered everywhere. Device checks, platform behavior, and tensor transfer paths are abstracted behind clearer boundaries.

For local CPU development, that means the core behavior can be exercised through CPU shared memory:

- vLLM runs on the CPU backend.
- LMCache runs as a local server.
- The vLLM KV connector talks to LMCache.
- KV tensors move through multiprocessing and shared-memory paths.
- L1 storage, eviction, metrics, and logs can all be inspected locally.

The production GPU path uses different low-level handles, but the development questions are similar enough to make the laptop workflow valuable. I can debug the connector shape, cache lookup behavior, L1 storage, and the basic cold-to-warm request pattern without waiting for GPU access.

That is why this topic is interesting to me: it makes LLM serving internals approachable.

## My Reproduction Workspace

I turned the guide into a repo with a repeatable layout:

```text
reproduce_vllm_lmcache/
├── LMCache/
├── vllm/
├── src/reproduce_vllm_lmcache/
│   ├── scripts/
│   │   ├── cpu_server_bench_test.sh
│   │   ├── start_lmcache.sh
│   │   └── start_vllm_sh
│   └── test_lmcache_e2e.py
├── pyproject.toml
└── uv.lock
```

The source article uses a shared virtual environment. I used `uv` and made the source checkouts explicit:

```toml
[tool.uv.sources]
lmcache = { path = "./LMCache", editable = true }
vllm = { path = "./vllm", editable = true }
```

That choice is important for my learning style. I do not want a one-time shell history. I want a repo where I can:

- rebuild vLLM locally
- install LMCache in editable mode
- preserve the exact run scripts
- record benchmark output
- change connector code and rerun the same checks

## Installation Notes

My repo assumes Apple Silicon macOS, Python `>=3.13`, and `uv`.

First, sync the root project:

```bash
uv sync
```

Then install vLLM's CPU requirements and build the local checkout:

```bash
uv pip install -r vllm/requirements/cpu.txt --index-strategy unsafe-best-match
uv pip install setuptools-scm setuptools-rust
uv pip install -e ./vllm --no-build-isolation
```

The important lesson from the original article is that normal vLLM wheels are not the right mental model for this setup. On a CPU-only Mac, I want the CPU build path, and I want the local source tree to be the thing Python imports.

I verify that with:

```bash
python -c "from importlib.metadata import version; print(version('vllm'))"
```

My output looked like this:

```text
0.23.1rc1.dev764+g576bf75d0.cpu
```

For LMCache, the important flag is `NO_GPU_EXT=1`:

```bash
NO_GPU_EXT=1 uv pip install --no-build-isolation -e ./LMCache
```

The source guide explains the motivation: on a laptop without accelerator hardware, the install should skip GPU extensions and GPU vendor dependencies. For my repo, this became a rule of the workspace rather than a detail I wanted to remember later.

## Verification Before Running Servers

I added cheap import checks before starting any server:

```bash
python -c "import vllm; import lmcache; print('vllm + lmcache imports ok')"
```

Then I verify the vLLM connector API:

```bash
python -c \
  'from vllm.distributed.kv_transfer.kv_connector.v1.base \
import KVConnectorBase_V1; print("v1 OK")'
```

And the LMCache multiprocessing connector:

```bash
python -c \
  'from lmcache.integration.vllm.lmcache_mp_connector \
   import LMCacheMPConnector; print("connector OK")'
```

These checks make the workflow less mysterious. If imports fail, the problem is installation. If imports pass but runtime fails, I can focus on ports, health checks, transfer config, model download, or vLLM startup.

## Quick Feedback With `server_bench`

The source guide recommends a standalone LMCache server benchmark before doing the full vLLM path. I like this because it gives a smaller feedback loop:

- no model download
- no vLLM server
- no token generation
- no connector integration
- just LMCache server store and retrieve behavior

In my repo, this is wrapped as:

```bash
src/reproduce_vllm_lmcache/scripts/cpu_server_bench_test.sh
```

The script starts LMCache, waits for the HTTP health check, and runs the CPU server bench:

```bash
lmcache bench server \
  --rpc-url tcp://127.0.0.1:5555 \
  --url http://127.0.0.1:8080 \
  --mode cpu \
  --transfer-mode lmcache_driven \
  --num-tokens 512 \
  --end 3
```

My successful run showed:

```text
Total requests: 3
Checksum OK: 3
Checksum FAIL: 0
Pass rate (%): 100.00
```

And the timing summary:

```text
Cold Lookup mean:   2.41 ms
Cold Store mean:   28.40 ms
Warm Lookup mean:   3.16 ms
Warm Retrieve mean: 21.14 ms
```

This benchmark has limits. It uses generated tensors and talks to the LMCache server directly, so it does not prove vLLM integration or token correctness. But it is the right first checkpoint because it tests whether LMCache can move and validate CPU KV-shaped data without corruption.

That is exactly the kind of small, repeatable verification loop I want before changing code.

## Full E2E: LMCache + vLLM

For the full integration, Terminal A starts LMCache:

```bash
src/reproduce_vllm_lmcache/scripts/start_lmcache.sh
```

The script keeps the knobs visible:

```bash
LMCACHE_HOST="${LMCACHE_HOST:-127.0.0.1}"
LMCACHE_ZMQ_PORT="${LMCACHE_ZMQ_PORT:-5555}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-1}"
LMCACHE_EVICTION_POLICY="${LMCACHE_EVICTION_POLICY:-LRU}"
```

The script also checks whether the ZMQ and HTTP ports are already occupied. This is not glamorous, but it matters when I am repeatedly stopping and starting local servers.

Terminal B starts vLLM:

```bash
src/reproduce_vllm_lmcache/scripts/start_vllm_sh
```

The macOS CPU settings are made explicit:

```bash
export VLLM_CPU_OMP_THREADS_BIND=nobind
export OMP_NUM_THREADS=1
export KMP_BLOCKTIME=0
export VLLM_DEVICE=cpu
export VLLM_CPU_KVCACHE_SPACE=1
export VLLM_HOST_IP=127.0.0.1
export GLOO_SOCKET_IFNAME=lo0
```

The source guide calls out this Apple Silicon issue because vLLM CPU startup can hang around OpenMP initialization. I put the environment variables directly into the script so the workaround travels with the reproduction.

Here is what each setting means in this local CPU setup:

| Setting | Meaning | Why I set it this way |
| --- | --- | --- |
| `VLLM_CPU_OMP_THREADS_BIND=nobind` | Tells vLLM's CPU backend not to bind OpenMP worker threads to a specific CPU-core list. With `nobind`, vLLM inherits the standard OpenMP settings instead. | On Apple Silicon, I want the simplest startup path. Disabling vLLM's CPU binding avoids the thread-placement path that can be fragile locally. |
| `OMP_NUM_THREADS=1` | Controls how many OpenMP threads each process uses for CPU compute. | This keeps the reproduction stable and avoids CPU oversubscription. It is not the fastest possible setting, but it is a good first debugging setting. |
| `KMP_BLOCKTIME=0` | Controls how long Intel/LLVM OpenMP worker threads spin-wait after finishing parallel work before sleeping. | Setting it to `0` reduces idle thread spin and helps avoid noisy CPU behavior during local testing. |
| `VLLM_DEVICE=cpu` | Forces vLLM to use the CPU platform instead of trying to discover an accelerator backend. | This repo is intentionally no-GPU. I want vLLM to enter the CPU backend every time. |
| `VLLM_CPU_KVCACHE_SPACE=1` | Reserves CPU memory for vLLM's KV cache pool. vLLM treats the value as GiB, so `1` means roughly 1 GiB. | `facebook/opt-125m` and `--max-num-seqs 1` do not need a large KV budget. Keeping this small leaves memory for LMCache L1, model weights, Python, and macOS. |
| `VLLM_HOST_IP=127.0.0.1` | Sets the IP address vLLM uses for internal process communication. This is not the same as the API server `--host`. | Everything in this reproduction runs on one machine, so loopback is the most explicit and safest choice. |
| `GLOO_SOCKET_IFNAME=lo0` | Tells PyTorch distributed / Gloo which network interface to use. On macOS, `lo0` is the loopback interface. | vLLM's CPU path can still initialize distributed communication pieces. Binding Gloo to loopback keeps those connections local and avoids picking Wi-Fi, VPN, or another interface. |

I group them mentally into three categories:

- **Threading controls:** `VLLM_CPU_OMP_THREADS_BIND`, `OMP_NUM_THREADS`, `KMP_BLOCKTIME`
- **CPU backend and memory controls:** `VLLM_DEVICE`, `VLLM_CPU_KVCACHE_SPACE`
- **Local distributed communication controls:** `VLLM_HOST_IP`, `GLOO_SOCKET_IFNAME`

For a first reproduction, I prefer stability over raw speed. After the pipeline is working, the variables most worth experimenting with are `OMP_NUM_THREADS` and `VLLM_CPU_KVCACHE_SPACE`. Increasing `OMP_NUM_THREADS` may improve CPU throughput, but it can also create contention. Increasing `VLLM_CPU_KVCACHE_SPACE` allows more KV-cache capacity, but it also competes with LMCache's L1 cache and the rest of the machine.

The vLLM script waits for LMCache's health endpoint, then launches vLLM with `LMCacheMPConnector`:

```json
{
  "kv_connector": "LMCacheMPConnector",
  "kv_role": "kv_both",
  "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
  "kv_connector_extra_config": {
    "lmcache.mp.host": "tcp://127.0.0.1",
    "lmcache.mp.port": 5555,
    "lmcache.mp.mp_transfer_mode": "lmcache_driven"
  }
}
```

The important field here is:

```json
"lmcache.mp.mp_transfer_mode": "lmcache_driven"
```

LMCache has two multiprocessing transfer paths:

| Mode | Who drives the transfer? | How to think about it | Main message flow |
| --- | --- | --- | --- |
| `lmcache_driven` | LMCache server drives the data movement. | The worker registers KV-cache memory handles, then LMCache pulls from or writes back to those handles. On GPU this is CUDA IPC; on CPU this can use POSIX shared memory. | `REGISTER_KV_CACHE` + `STORE` / `RETRIEVE` |
| `engine_driven` | The inference engine worker drives the data movement. | The vLLM side gathers/scatters the relevant KV chunks and sends or receives the actual data through the engine-driven context. This is the non-GPU worker-side path. | `REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT` + `PREPARE` / `COMMIT` |

The naming is easy to mix up, so I remember it this way:

- **`lmcache_driven`**: LMCache is active. It uses registered handles and moves KV through the server-driven path.
- **`engine_driven`**: the engine worker is active. vLLM gathers or scatters KV chunks and coordinates the transfer from the worker side.

In LMCache's `auto` mode, CPU workers usually route to `engine_driven` and GPU workers route to `lmcache_driven`. In this reproduction, I explicitly set `lmcache_driven` because I wanted to exercise the same high-level handle-based path discussed in the starter guide, while still staying on a CPU MacBook through shared memory.

There are two related settings that can be confusing:

- `lmcache.mp.mp_transfer_mode` is a **worker / connector setting**. It tells the vLLM LMCache connector which transfer context to create.
- `lmcache server --supported-transfer-mode` is a **server setting**. It decides which transfer modules the LMCache server loads. Leaving the server on `auto` is convenient because it can accept either path.

The model is intentionally small:

```bash
VLLM_MODEL="${VLLM_MODEL:-facebook/opt-125m}"
```

For this task, tiny and debuggable is better than impressive. The purpose is to verify the data path, not to benchmark a frontier model.

## Cache Hit Test

Terminal C sends the same prompt twice:

```bash
python src/reproduce_vllm_lmcache/test_lmcache_e2e.py
```

The test uses a repeated prompt of about 641 tokens and asks for 8 output tokens:

```python
payload = {
    "model": "facebook/opt-125m",
    "prompt": prompt,
    "max_tokens": 8,
    "temperature": 0.0,
}
```

The prompt length is intentional. LMCache chunks prefixes, and a very short prompt may not exercise the reuse path meaningfully. Sending the same longer prompt twice creates a simple cold-request and warm-request pattern.

My recorded float16 run:

```text
round 1 200 1.71s {'prompt_tokens': 641, 'total_tokens': 649, 'completion_tokens': 8, 'prompt_tokens_details': None}
round 2 200 0.42s {'prompt_tokens': 641, 'total_tokens': 649, 'completion_tokens': 8, 'prompt_tokens_details': None}
```

My recorded bfloat16 run:

```text
round 1 200 0.68s {'prompt_tokens': 641, 'total_tokens': 649, 'completion_tokens': 8, 'prompt_tokens_details': None}
round 2 200 0.67s {'prompt_tokens': 641, 'total_tokens': 649, 'completion_tokens': 8, 'prompt_tokens_details': None}
```

The float16 run shows the clearer warm-path speedup. The bfloat16 run is faster on the cold request but almost flat between the two requests. I do not treat those numbers as universal. CPU kernels, PyTorch behavior, dtype support, and vLLM implementation details all affect timing.

What I trust is the shape of the experiment:

1. Start LMCache.
2. Start vLLM with `LMCacheMPConnector`.
3. Send a repeated long-enough prompt.
4. Compare cold and warm behavior.
5. Check LMCache logs for store, prefetch, and retrieve messages.

## Memory Planning

One useful section from the LMCache guide is memory planning. Even with a small model, a laptop run has several memory consumers:

- vLLM's CPU KV cache pool
- LMCache's L1 cache
- model weights
- Python runtime overhead
- the operating system and whatever else is open

My default settings mirror the safe small-workload idea:

```bash
export VLLM_CPU_KVCACHE_SPACE=1
lmcache server --l1-size-gb 1
vllm serve facebook/opt-125m --max-model-len 2048 --max-num-seqs 1
```

If memory is tight, the source guide's advice is to shrink both vLLM and LMCache budgets, then reduce model length and sequence concurrency. That is worth keeping in mind because out-of-memory failures can look like mysterious serving issues when they are really just local resource pressure.

## Model Download Notes

The upstream guide also includes practical model-download fallbacks. I did not need to turn this into a script yet, but it belongs in the mental checklist:

- Use a Hugging Face mirror if the default Hub route is slow or blocked.
- Pre-download the model and pass a local path to `vllm serve`.
- Reuse an existing Hugging Face cache through `HF_HOME`.
- Use LMCache's CI download script as a reference for retry behavior.

For a reproduction repo, I prefer to keep `facebook/opt-125m` as the default because it is small enough for local experiments and familiar enough that failures are usually environment-related.

## What I Want To Learn Next

The LMCache article frames four approachable contribution areas: frontend, L1 eviction, L2 storage, and observability. That framing is exactly why I like this topic.

Here is how I understand the learning path:

| Area | Why it interests me |
| --- | --- |
| Frontend | It is closest to request handling and connector behavior. |
| L1 eviction | It turns cache reuse into a real systems problem, not just a demo. |
| L2 storage | It connects local KV reuse to disk, Redis, object storage, and larger deployments. |
| Observability | It answers the most important practical question: did the cache actually help? |

My reproduction repo is the starting point for that exploration. It gives me a CPU-only baseline where I can change one thing, run `server_bench`, run the E2E script, and inspect what changed.

## Takeaways

The LMCache guide made this topic interesting to me because it changes the barrier to entry. Instead of needing a GPU box before I can learn anything meaningful, I can start with a laptop and still touch the real concepts:

- KV cache reuse
- vLLM's v1 KV connector
- LMCache multiprocessing mode
- CPU shared-memory transfer
- L1 cache behavior
- cold versus warm request verification
- source-level development with editable installs

My repo is not meant to be the final word on vLLM + LMCache performance. It is a learning and reproduction workspace:

[binzhango/reproduce_vllm_lmcache](https://github.com/binzhango/reproduce_vllm_lmcache)

The most important result is that I can now run the core loop locally, explain why each part exists, and use that setup as a base for deeper LMCache work.
