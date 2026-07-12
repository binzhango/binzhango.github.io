---
title: "KV-Centric LLM Serving: vLLM, SGLang, and Disaggregated Attention"
date: 2026-07-05
author: BZ
description: "A practical mental model for KV cache memory, vLLM PagedAttention, SGLang RadixAttention, and prefill/decode disaggregation in modern LLM serving."
categories:
  - AI ENGINEERING
  - LARGE LANGUAGE MODELS
tags:
  - vllm
  - sglang
  - paged attention
  - radix attention
  - kv cache
  - disaggregated serving
  - prefill decode
---

<!-- more -->

The more I look at LLM serving, the more it feels like the main object is not the request, the model, or even the GPU.

The main object is the <span class="term">KV cache</span>.

The old mental model is simple:

![Animated old LLM serving mental model: request to GPU to response](/assets/images/2026/old-request-gpu-response.gif)

The newer serving stack looks more like this:

![Animated KV-centric serving stack with router, KV cache state, and scheduling decisions](/assets/images/2026/kv-centric-serving-stack.gif)

That is the thread connecting <span class="term">vLLM</span>, <span class="term">SGLang</span>, <span class="term">LMCache</span>, <span class="term">Mooncake</span>, <span class="term">Dynamo</span>, <span class="term">DistServe</span>, and many newer systems. They are not only optimizing attention math. They are managing <span class="term">attention state</span>: where it lives, how it is reused, and how quickly it moves through the serving stack.

## The Core Problem: Prefill and Decode Are Different

An autoregressive LLM request has two major phases:

| Phase | What happens | What it stresses |
| --- | --- | --- |
| <span class="term">Prefill</span> | The model processes the prompt and creates the initial KV cache. | Heavy parallel compute, prompt length, input batching. |
| <span class="term">Decode</span> | The model generates one token at a time, reading old KV and appending new KV. | Memory bandwidth, active KV size, per-token latency. |

These phases have very different shapes.

<span class="term">Prefill</span> is compute-heavy because the whole input prompt can be processed in parallel. Long prompts, RAG documents, tool schemas, and few-shot examples all make this phase expensive.

<span class="term">Decode</span> is sequential. Each new token depends on the previous token. The GPU repeatedly reads the existing KV cache, computes one more step, appends the new `K/V`, and repeats. This often underuses raw GPU compute while stressing memory bandwidth and latency.

When both phases run on the same GPU worker, they interfere with each other:

<ul class="risk-list">
  <li>Long prefills can block decode steps and create spikes in <span class="term">TTFT</span>, or time to first token.</li>
  <li>Decode's low compute utilization can waste expensive GPU cycles.</li>
  <li>Prefill and decode compete for the same HBM memory that stores model weights and KV cache.</li>
  <li>One scheduling policy has to satisfy two very different workloads.</li>
</ul>

![Animated prefill and decode phase KV cache transfer](/assets/images/2026/prefill-decode-kv-transfer.gif)

Prefill is parallel and compute-heavy; decode is sequential and memory-bandwidth-heavy. Running both on the same worker creates scheduling and memory contention.

This is the reason people talk about <span class="term">prefill/decode disaggregation</span>.

## The Solution Shape: Disaggregate the Serving Stack

Disaggregated serving splits prefill and decode onto separate worker pools:

![Animated disaggregated prefill and decode worker pools](/assets/images/2026/disaggregated-prefill-decode-pools.gif)

Disaggregation lets prefill and decode scale separately, but it makes KV cache transfer and cache-aware routing part of the critical path.

The prefill cluster can be optimized for prompt throughput. The decode cluster can be optimized for token latency and concurrency. They can scale independently, use different parallelism strategies, and sometimes even use different GPU types.

That sounds clean, but it creates the hard new problem:

> The KV cache produced by prefill must arrive at decode quickly enough that transfer does not erase the benefit of disaggregation.

So disaggregated serving is not just "split the workload." It means the serving stack has to do several things at once:

<ul class="action-grid">
  <li><span class="term">Split the workload</span> between prefill-heavy and decode-heavy workers.</li>
  <li><span class="term">Move KV cache efficiently</span> so transfer latency does not erase the win.</li>
  <li><span class="term">Route requests based on cache locality</span>, not only on open worker slots.</li>
  <li><span class="term">Manage cache tiers under memory pressure</span> across GPU HBM, CPU DRAM, disk, and remote storage.</li>
</ul>

That is why modern systems increasingly feel <span class="term">KV-centric</span>.

## What the KV Cache Actually Stores

During generation, every token leaves behind key and value tensors.

The important detail is that KV is stored <span class="term">per layer</span>. A token does not have one global `K` and one global `V`; it has a `K/V` pair at every transformer layer, and each pair is split by KV heads and head dimension.

Conceptually:

![Animated KV cache stored per transformer layer](/assets/images/2026/kv-cache-per-layer.gif)

The rough memory formula is:

$$
\text{KV bytes per token}
=
2
\times
\text{num layers}
\times
\text{num KV heads}
\times
\text{head dim}
\times
\text{bytes per element}
$$

The `2` is for `K` and `V`.

For a Llama-2-7B-style model with multi-head attention:

| Parameter | Value |
| --- | ---: |
| Layers | `32` |
| KV heads | `32` |
| Head dim | `128` |
| KV dtype | `fp16`, `2 bytes` |
| KV per token | `524,288 bytes`, about `512 KB` |

That means KV cache grows with both <span class="term">context length</span> and <span class="term">concurrency</span>.

| Context length | 1 request | 10 concurrent | 100 concurrent |
| --- | ---: | ---: | ---: |
| `1K` tokens | `0.5 GB` | `5 GB` | `50 GB` |
| `4K` tokens | `2 GB` | `20 GB` | `200 GB` |
| `8K` tokens | `4 GB` | `40 GB` | `400 GB` |
| `32K` tokens | `16 GB` | `160 GB` | `1,600 GB` |
| `128K` tokens | `64 GB` | `640 GB` | `6,400 GB` |

This table uses a standard MHA 7B-class example. Modern architectures reduce the slope:

| Technique | KV cache effect |
| --- | --- |
| `MHA` | Every query head has its own `K/V` head. This is the expensive baseline. |
| `GQA` | Groups of query heads share fewer KV heads, often making KV much smaller. |
| `MQA` | All query heads share one KV head, shrinking cache further with tradeoffs. |
| `MLA` | Compresses KV into a latent representation, as in DeepSeek-style architectures. |
| <span class="term">KV quantization</span> | Stores KV in lower precision, such as `fp8` or `int8`. |
| <span class="term">Sliding window attention</span> | Caps the active KV window, trading long-range recall for bounded memory. |

The exact number matters. A `10K` token prompt on a 70B-class model can mean many GB of KV transfer. With old-style MHA, it can be tens of GB. With a modern GQA model such as Llama-3-70B, it is much smaller, but still large enough to dominate latency if transferred poorly.

So the right lesson is not one fixed number. The lesson is:

> KV cache size is determined by layers, KV heads, head dimension, dtype, context length, and concurrency.

## The KV Transfer Problem

In a disaggregated stack, prefill creates KV on one worker and decode needs it on another.

That makes the transfer path a first-class part of the serving system:

| Transfer path | Why it matters |
| --- | --- |
| `NVLink` | Fast GPU-to-GPU movement inside a node. |
| `PCIe` | Common local device path, slower than NVLink. |
| `InfiniBand RDMA` | High-throughput cross-node transfer in GPU clusters. |
| `RoCE` | RDMA over Ethernet for datacenter fabrics. |
| `CPU DRAM` | Useful warm cache or staging layer, but adds movement. |
| `NVMe SSD` | Larger and cheaper local capacity, much slower than memory. |
| Object store | Huge durable capacity, only useful when latency can be hidden or amortized. |

<span class="term">Mooncake</span> is a good example of this mental model. Its architecture treats KV cache as the central resource: a scheduler decides where work should go, and a transfer engine moves KV cache efficiently, using RDMA-style paths to reduce CPU involvement and overlap movement with compute where possible.

<span class="term">NVIDIA Dynamo</span> makes the same issue visible in a practical serving stack. Its documentation separates the frontend, router, prefill workers, decode workers, and KV transfer path. It also warns that without RDMA or an equivalent fast fabric, KV movement can dominate `TTFT` and throughput.

The key point:

> Disaggregation moves the bottleneck from "can this GPU do both phases?" to "can the system place and move KV cache fast enough?"

## The Four-Tier KV Cache Hierarchy

Once KV cache becomes the main serving object, the system naturally becomes a memory hierarchy.

| Tier | Storage | Role | Tradeoff |
| --- | --- | --- | --- |
| `L1` | GPU HBM | Hot active KV cache used by attention kernels. | Fastest and most expensive; capacity is scarce. |
| `L2` | CPU DRAM | Warm cache and staging layer. | Larger and cheaper, but transfer adds latency. |
| `L3` | NVMe SSD | Cold local cache. | Much larger, useful for reuse and offload, but slower. |
| `L4` | Object store | Durable remote cache. | Huge capacity, highest latency. |

This hierarchy explains why different projects emphasize different things.

At `L1`, <span class="term">vLLM</span>, <span class="term">SGLang</span>, <span class="term">TensorRT-LLM</span>, <span class="term">LMDeploy</span>, and <span class="term">vAttention</span> focus on making GPU-resident KV usable under high concurrency.

At `L2` and below, <span class="term">LMCache</span>, <span class="term">FlexGen</span>, <span class="term">Dynamo KVBM</span>, AttentionStore-style systems, and other KV stores start to matter. They ask: what happens when KV should outlive one GPU worker, one request, or one hot HBM allocation?

The serving engine wants hot KV in HBM, but HBM is scarce. A single GPU may have `40-80 GB`; large model weights consume much of that before user KV cache even arrives. So every serious serving stack needs policies for:

<ul class="chip-list">
  <li>eviction</li>
  <li>promotion and demotion</li>
  <li>prefix reuse</li>
  <li>tenant isolation</li>
  <li>transfer scheduling</li>
  <li>cache-aware routing</li>
</ul>

This is why cache policy is not a minor implementation detail. It decides whether the stack keeps useful attention state close to compute or spends the whole time recomputing and moving it.

## L1 GPU HBM: Where vLLM and SGLang Fit

<span class="term">vLLM</span> and <span class="term">SGLang</span> are easiest to understand as `L1` KV-cache systems with different emphasis.

<span class="term">vLLM</span> starts from <span class="term">memory allocation</span>:

> How do we store dynamic KV cache in GPU memory without wasting huge amounts of HBM?

<span class="term">SGLang</span> starts from <span class="term">prefix reuse</span>:

> How do we automatically find shared prompt prefixes and reuse their KV cache across requests?

Both are attention-serving optimizations, but they sit at different levels.

## vLLM: PagedAttention

vLLM's core idea is <span class="term">PagedAttention</span>.

The analogy is operating-system virtual memory. A process sees a logical address space, but the physical memory pages do not need to be contiguous. vLLM applies this idea to KV cache.

Instead of reserving one large contiguous KV tensor per request, vLLM splits KV cache into fixed-size blocks. A request has logical blocks, and the engine maps them to physical GPU-memory blocks.

![Animated vLLM PagedAttention logical KV blocks mapped to physical GPU KV blocks](/assets/images/2026/vllm-pagedattention-logical-physical-kv.gif)

The request still behaves as if its tokens are contiguous. The attention kernel knows how to gather the physical blocks.

This matters because request lengths are dynamic:

<ul class="risk-list">
  <li>One request may stop after <code>20</code> tokens.</li>
  <li>Another may continue for <code>4,000</code> tokens.</li>
  <li>A new request may arrive while older requests are decoding.</li>
  <li>Parallel sampling may share a prefix and then branch.</li>
</ul>

Without paging, the engine either over-reserves memory or suffers fragmentation. PagedAttention makes KV allocation granular and reusable.

On top of this block system, vLLM supports <span class="term">automatic prefix caching</span>. It hashes full KV blocks using the parent block hash, tokens in the current block, and extra values such as LoRA IDs, multimodal hashes, or cache salts. If another request shares the same prefix blocks, vLLM can reuse the existing KV cache.

The practical implication is simple:

<ul class="signal-list">
  <li>identical stable prefixes help</li>
  <li class="bad">random timestamps near the front hurt</li>
  <li class="bad">inconsistent templates hurt</li>
  <li>block alignment matters because caching happens at block granularity</li>
</ul>

PagedAttention gives vLLM the memory substrate. Automatic prefix caching uses that substrate to avoid repeated prefill.

## SGLang: RadixAttention

SGLang's key idea is <span class="term">RadixAttention</span>.

It stores reusable prompt and generation prefixes in a radix tree. A radix tree is like a compressed trie: edges can represent sequences of tokens rather than single tokens.

Conceptually:

![Animated RadixAttention tree showing shared cached prefixes branching into request-specific suffixes](/assets/images/2026/sglang-radixattention-prefix-tree.gif)

When a new request arrives, SGLang looks for the longest matching prefix. If a matching prefix exists, the runtime reuses the corresponding KV cache and only computes the suffix.

This is especially useful for:

<ul class="use-case-grid">
  <li>shared system prompts</li>
  <li>repeated few-shot examples</li>
  <li>multi-turn conversations</li>
  <li>agent loops with stable tool context</li>
  <li>self-consistency sampling from the same prompt</li>
</ul>

<span class="term">RadixAttention</span> is not a new attention formula. It is a runtime strategy for reusing attention state.

## PagedAttention vs. RadixAttention vs. APC

These terms are easy to mix up, so I like this table:

| Concept | Main question | Data structure | Main win |
| --- | --- | --- | --- |
| <span class="term">PagedAttention</span> | How do we allocate KV memory efficiently? | Fixed-size KV blocks and block tables. | Less fragmentation and better HBM utilization. |
| <span class="term">Automatic Prefix Caching</span> | Can repeated block prefixes be reused? | Hashes over full KV blocks. | Avoids recomputing repeated prefill blocks. |
| <span class="term">RadixAttention</span> | Can repeated token prefixes be found automatically? | Radix tree over token prefixes. | Reuses KV for shared prompt paths. |

The short version:

> vLLM made KV memory behave more like paged virtual memory. SGLang made repeated prompt programs behave more like prefix-tree lookups.

They are not competing mental models. They are different layers of the same KV-cache problem.

## Framework Coverage by Cache Tier

This table is not meant to be a strict feature matrix. It is a mental map of where each system tends to focus.

| System | L1 GPU HBM | L2 CPU DRAM | L3 NVMe / disk | L4 object / remote store |
| --- | --- | --- | --- | --- |
| <span class="term">vLLM</span> | PagedAttention, APC, chunked prefill | CPU swap / offload paths | Mostly through integrations | Mostly through integrations |
| <span class="term">SGLang</span> | RadixAttention, paged KV, chunked prefill | Can integrate with lower-tier cache systems | Backend-dependent | Backend-dependent |
| <span class="term">LMCache</span> | Integrates with engines such as vLLM | Primary warm-cache use case | Disk/offload support | Remote/persistent cache patterns |
| <span class="term">Mooncake</span> | KV-aware serving with hot GPU cache | Uses cluster memory resources | Uses SSD resources | KV-centric cluster-level design |
| <span class="term">Dynamo KVBM</span> | Works with backend engines | KV cache offloading / staging | Backend and integration dependent | Blob/object style integrations |
| <span class="term">FlexGen</span> | Limited GPU memory assumption | CPU offload is central | Disk offload is central | Not the main focus |
| <span class="term">vAttention</span> | Virtual paging for KV cache | Demand paging model | Not the main focus | Not the main focus |
| <span class="term">LMDeploy</span> | TurboMind, continuous batching, KV quantization | Less central | Less central | Less central |
| <span class="term">TensorRT-LLM</span> | Optimized NVIDIA kernels, paged KV, in-flight batching | Serving-stack dependent | Serving-stack dependent | Serving-stack dependent |

The important correction to my own thinking: "framework support" is not binary. A serving engine may own `L1`, while a cache layer or distributed runtime owns `L2-L4`. In production these are composed.

## Metrics Improved by Disaggregation

Disaggregation is valuable only if it improves the metrics users and operators care about.

| Metric | Coupled serving | Disaggregated target |
| --- | --- | --- |
| <span class="term">TTFT</span> | High variance when long prefills block decode. | Lower and more predictable. |
| <span class="term">TPOT</span> | Decode latency can degrade during heavy prefill. | More stable per-output-token latency. |
| <span class="term">GPU utilization</span> | Decode can waste compute while prefill creates bursts. | Each pool can run closer to its best utilization profile. |
| <span class="term">Cost</span> | Every worker needs to handle both phases. | Different GPU types or replica ratios can serve different phases. |
| <span class="term">Scalability</span> | Prefill and decode scale together. | Prefill and decode scale independently. |

DistServe calls the real target <span class="term">goodput</span>: how much traffic can be served while meeting both TTFT and TPOT constraints. That is a better target than raw throughput because a system that produces many tokens but violates latency SLOs is not actually good serving.

## Major Systems and Their Approach

Here is the landscape from my notes, rewritten as a clearer map.

### Mooncake

<span class="term">Mooncake</span> is a KV-cache-centric disaggregated serving architecture used for Moonshot AI's Kimi service.

The useful mental split:

<ul class="role-list">
  <li><span class="term">Conductor</span>: global scheduler</li>
  <li><span class="term">Transfer Engine</span>: RDMA-based KV movement</li>
  <li><span class="term">KV-centric cache</span>: treats KV cache as the first-class cluster resource</li>
</ul>

Mooncake is especially interesting for long-context workloads because repeated large prompts make KV placement and reuse extremely valuable.

### NVIDIA Dynamo

<span class="term">Dynamo</span> is a distributed inference serving stack with:

<ul class="chip-list">
  <li>frontend and router components</li>
  <li>backend workers for vLLM, SGLang, and TensorRT-LLM</li>
  <li>KV-cache-aware routing</li>
  <li>disaggregated serving</li>
  <li>KV cache offloading integrations</li>
</ul>

The key term in Dynamo is <span class="term">NIXL</span>, a transfer abstraction around paths such as NVLink, InfiniBand, and PCIe. The router can make decisions based on KV locality rather than only worker load.

### DistServe

<span class="term">DistServe</span> formalized the prefill/decode split as a goodput optimization problem.

It argues that prefill and decode should be placed and parallelized separately because they have different latency targets:

<ul class="signal-list">
  <li>prefill affects <code>TTFT</code></li>
  <li>decode affects <code>TPOT</code></li>
</ul>

The core lesson is that disaggregation is not only about utilization. It is about meeting both latency constraints at the same time.

### vLLM Disaggregated Prefill

vLLM has experimental disaggregated prefilling support and examples that separate prefill and decode workers. The key abstraction is a transfer connector: the runtime needs a way to move KV state from the prefill side to the decode side.

In practice, this area connects vLLM to systems such as LMCache, NIXL/Dynamo-style transfer, and Mooncake-style serving architectures.

### Splitwise

Splitwise explores heterogeneous serving: use different hardware for different phases.

The motivation is intuitive:

<ul class="signal-list">
  <li>prefill may benefit from high-end compute-heavy GPUs</li>
  <li>decode may run cost-effectively on different GPUs if memory and latency targets are satisfied</li>
</ul>

This matters because disaggregation is not only a latency trick. It can also be a cost-structure trick.

## Framework Landscape by Problem

Another way to organize the ecosystem is by the problem each system is trying to solve.

| Problem | Systems | Main idea |
| --- | --- | --- |
| <span class="term">KV reuse / prefix caching</span> | vLLM APC, SGLang RadixAttention, LMCache, CacheGen | Avoid recomputing repeated prompt prefixes or compress/move KV more efficiently. |
| <span class="term">Prefill/decode disaggregation</span> | Mooncake, DistServe, Splitwise, Dynamo, vLLM disagg prefill | Split prefill and decode into specialized pools. |
| <span class="term">Memory management / paging</span> | vLLM PagedAttention, vAttention, FlexGen | Reduce fragmentation or offload KV beyond GPU HBM. |
| <span class="term">Full serving engines</span> | vLLM, SGLang, TensorRT-LLM, LMDeploy, TGI | Provide batching, scheduling, kernels, APIs, and production serving behavior. |
| <span class="term">Distributed KV sharing</span> | Dynamo KVBM, LMCache, AttentionStore-like systems | Make KV reusable across workers, tiers, sessions, or restarts. |
| <span class="term">Research cache policies</span> | ChunkAttention, Pensieve, Quest, Scissorhands, MagicPIG | Explore better sharing, migration, retrieval, and eviction policies. |

This table is the bigger picture behind the vLLM/SGLang comparison. vLLM and SGLang are not isolated tools; they are two important nodes in a larger KV-management ecosystem.

## Application Design Still Matters

Infrastructure can only reuse what the application gives it.

<div class="prompt-compare">
  <section class="prompt-panel">
    <div class="prompt-title">Good prompt structure improves cache reuse</div>
    <ul class="prompt-list">
      <li>Put stable content first.</li>
      <li>Keep system prompts byte-for-byte identical.</li>
      <li>Keep tool schemas in a stable order.</li>
      <li>Put volatile request-specific fields later.</li>
      <li>Reuse few-shot examples consistently when possible.</li>
    </ul>
  </section>
  <section class="prompt-panel">
    <div class="prompt-title">Bad prompt structure destroys reuse</div>
    <ul class="prompt-list bad">
      <li>random request IDs before the shared system prompt</li>
      <li>shuffled tool definitions</li>
      <li>inconsistent whitespace or templates</li>
      <li>RAG chunks inserted before stable instructions</li>
      <li>tenant-specific noise mixed into otherwise shared prefixes</li>
    </ul>
  </section>
</div>

The engine sees tokens, not your intent. If two prompts do not tokenize to the same prefix, the cache cannot help.

## Final Mental Model

The attention formula tells us what the model computes:

$$
\mathrm{Attention}(Q, K, V) =
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

The serving stack asks a different set of questions:

<ul class="question-list">
  <li>Where is the KV cache?</li>
  <li>Is the useful prefix already cached?</li>
  <li>Is it in GPU HBM, CPU DRAM, NVMe, or remote storage?</li>
  <li>Should this request go to the least busy worker or the worker with the best KV locality?</li>
  <li>Is it cheaper to recompute KV or transfer it?</li>
  <li>Which KV blocks should be evicted?</li>
</ul>

That is the shift:

> Modern LLM serving is becoming KV-cache management.

vLLM's <span class="term">PagedAttention</span> solves the hot HBM allocation problem. vLLM's <span class="term">automatic prefix caching</span> reuses repeated block prefixes. SGLang's <span class="term">RadixAttention</span> organizes repeated prompt paths for runtime reuse. Disaggregated systems such as <span class="term">Mooncake</span>, <span class="term">Dynamo</span>, and <span class="term">DistServe</span> move the same idea up to the cluster level.

A rough cost instinct is:

$$
\text{Serving cost}
\propto
\frac{
\text{KV bytes computed or moved}
\times
\text{requests}
}{
\text{KV reuse}
}
$$

It is not a pricing equation. It is a compass.

The goal is to compute less repeated KV, move fewer KV bytes, keep hot KV close to the GPU, and schedule requests around cache locality. Once that clicks, vLLM attention, SGLang attention, LMCache, and disaggregated serving stop feeling like separate topics.

They are all different layers of the same system:

> A system for managing expensive attention state.
