---
title: "Attention Dilution"
date: 2026-03-15
author: BZ
description: "Why longer LLM context windows can make answers worse: softmax dilution, positional bias, and attention sinks."
categories: ["LLM"]
tags: ["attention", "context", "transformers", "agents", "rag"]
---

<!-- more -->

**Attention dilution** (also called **context dilution**) is one of the fundamental limitations of transformer-based LLMs when dealing with long contexts or extended agent memory.

It explains why simply giving an LLM *more* history, retrieved documents, tool outputs, or conversation turns often makes the model **worse** at answering the current question, even when the relevant information is still technically inside the context window.

> **Long context is not magic memory.** It is finite, biased, and easy to dilute.

In short, the model's attention mechanism has a fixed budget of focus: `softmax` weights always sum to `1`. As the number of tokens grows, that budget gets spread thinner. Relevant signals weaken, irrelevant or positional noise steals focus, and the model becomes distracted.

This is why long-context agents frequently ignore key past facts, repeat old behaviors, or fail to connect the current query to buried memory.

## The Mathematics of Dilution

Recall the core self-attention formula:

$$
\mathrm{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

For any query token, the attention weights $a_i$ over keys are:

$$
a_i = \frac{\exp(s_i)}{\sum_j \exp(s_j)} \quad \text{where} \quad s = \frac{QK^\top}{\sqrt{d_k}}
$$

This is where dilution begins. The `softmax` is **zero-sum**: the total attention mass is always exactly `1`. If you increase the number of keys in context, and the similarity scores are not extremely peaked on the relevant token, each individual weight shrinks.

In a simple simulation with one relevant token given a logit advantage of `+2.0` and all other tokens neutral, the effect is immediate:

| Context size | Attention on relevant token | Effect |
| --- | ---: | --- |
| `N = 10` | `~45.1%` | Baseline |
| `N = 100` | `~6.9%` | `~6.5x` dilution |

The relevant token is still present. It is just no longer dominant.

Even a tiny positional sink bias on the first token, say a logit bonus of `+0.1`, starts pulling weight away. That is exactly what happens in real LLMs at `128k+` tokens: the relevant past question or memory fact must compete with thousands of distractors and often gets diluted into near-irrelevance.

## Positional Bias and Lost in the Middle

Dilution is not uniform. It is strongly shaped by **positional bias**.

The 2023 paper [**"Lost in the Middle: How Language Models Use Long Contexts"**](https://aclanthology.org/2024.tacl-1.9/) (Liu et al.) showed a consistent **U-shaped performance curve** across GPT-3.5-Turbo, Claude, Llama variants, and others:

- Best performance when the key fact or document is at the **very beginning**.
- Best performance also when it is at the **very end**.
- Worst performance when it is buried in the **middle**.

In one representative result for GPT-3.5-Turbo over 20 documents, accuracy dropped from **76.8%** when the answer-bearing document was first to **53.8%** when it was in the middle.

In some setups, middle placement performed **worse than the closed-book baseline**, meaning worse than providing no documents at all.

This is the critical point: longer windows improve **readability** of long inputs, but not necessarily **usability** of the information inside them.

Why does this happen? Partly because `RoPE` introduces effective distance decay, and partly because training creates emergent **primacy** and **recency** biases. Middle tokens tend to receive weaker effective attention.

## Attention Sinks and Primacy Bias

There is another layer on top of dilution: **attention sinks**.

LLMs learn that the first few tokens, even meaningless ones like newlines, attract disproportionately high attention across many layers. This is not primarily semantic. It is a structural consequence of how attention is distributed.

When no token is strongly relevant, the model still has to allocate the full attention budget of `1.0`. Early tokens are visible to every later token during training, so they become stable places for excess attention mass to accumulate.

This is why removing those early tokens from a pure sliding-window `KV cache` can cause performance to collapse. The [**"Efficient Streaming Language Models with Attention Sinks"**](https://openreview.net/forum?id=NG7sS51zVF) paper shows that keeping only a few sink tokens is often enough to recover performance across extremely long sequences.

That behavior helps explain both the strong primacy bias and why stuffing important memory into the middle of an agent's context is so brittle.

## Agent-Level Failure Modes

In an agent with long-term memory, such as a system combining a vector DB, `RAG`, tool outputs, and conversation history, context often ends up arranged like this:

1. The system prompt, early turns, and initial tool results sit at the **beginning**.
2. The latest user request and most recent actions sit at the **end**.
3. Everything else gets pushed into the **middle**.

That middle region is where old questions, failed attempts, weak retrievals, distractor chunks, and stale tool output pile up.

The result is **context distraction** or **context rot**. The agent may:

- over-rely on recent noise instead of the original goal
- ignore a critical fact from 20 turns ago, even if it was retrieved
- repeat past behaviors because diluted history still looks pattern-like
- follow distractors that share keywords but not meaning
- hallucinate because relevant evidence is present but too weakly weighted

Recent 2025 studies reinforce this pattern. Chroma's [**"Context Rot: How Increasing Input Tokens Impacts LLM Performance"**](https://research.trychroma.com/context-rot) report shows that performance can degrade sharply as context grows, and that distractors become more damaging as input length increases.

Agentic systems are especially vulnerable because their histories grow organically rather than being curated upfront.

## Why Bigger Context Windows Still Fall Short

Bigger windows help, but they do not remove the core problem.

Three forces are still working against us:

- **Architecture:** `softmax` competition, `RoPE` decay, and multi-layer attention dynamics still create dilution.
- **Training:** pretraining mixes tasks that favor short-term recency with others that reward broader recall, producing emergent positional bias.
- **Zero-sum attention:** more tokens still mean more opportunities for irrelevant tokens to steal weight.

Newer models with `128k+`, `1M`, or larger windows reduce the pain, but they do not eliminate the U-curve. The issue is not just absolute context length. It is the **relative position** of useful information within the window.

## Practical Mitigations

You usually cannot fix this at the architecture level, but you can engineer around it.

- **Curate ruthlessly.** Summarize, compress, and prune history so only high-signal memory remains.
- **Reposition critical facts.** Put the current question and the most relevant evidence near the end. Prompts that explicitly ask the model to recite evidence first can help.
- **Use selective retrieval.** `RAG` with reranking and contextual metadata can reduce dilution and improve hit quality. Anthropic's [**"Introducing Contextual Retrieval"**](https://www.anthropic.com/engineering/contextual-retrieval) reports lower retrieval failure rates when contextual chunk metadata and reranking are combined.
- **Adopt hierarchical memory.** Keep a short working memory, and retrieve episodic or semantic memory only when needed.
- **Apply calibration techniques.** Some work suggests that subtracting positional bias from attention scores can recover meaningful accuracy.
- **Prefer smaller focused contexts.** Checkpoint clean state, split tasks, or use multi-agent patterns when a single giant prompt becomes noisy.

## How Agent Frameworks Manage the Problem

Agent frameworks do not solve attention dilution inside the transformer. What they do instead is **manage the prompt boundary**: they decide what stays in active context, what gets stored elsewhere, and what gets pulled back in later.

The safest way to describe this is not "frameworks fix memory," but rather:

> Frameworks provide **documented memory-management primitives** such as trimming, retrieval, checkpointing, scoped recall, and long-term storage.

### Shared Strategies Across Frameworks

Here are the main techniques widely adopted in production agent systems:

#### 1. Context Selection & Selective Retrieval

Retrieve only the most relevant memories, documents, or facts for the current step using semantic search, reranking, or metadata filtering.

Avoid dumping full history or all tool outputs into the prompt.

Many frameworks now support dynamic retrieval inside agent nodes or individual turns.

#### 2. Compression & Summarization

Periodically, or when nearing token thresholds such as `70%` to `95%` of the context window, summarize conversation history, tool results, or long documents.

Common techniques include recursive summarization, hierarchical summarization, abstractive distillation, and LLM-based compaction that preserves goals, decisions, and key facts.

Examples include Claude Code-style auto-compaction and LangGraph summarization middleware or nodes.

#### 3. Pruning & Trimming

Remove stale, redundant, low-relevance, or conflicting messages and tool outputs using heuristics such as keeping only the last `N` turns, or using learned pruners.

Offload raw heavy data externally and keep only references, summaries, or metadata in the active prompt.

#### 4. Repositioning & Ordering

Place critical information such as the current question, key facts, and instructions at the beginning or end of the prompt to benefit from primacy and recency effects.

Retrieved chunks can also be strategically ordered, and in some systems iteratively reordered based on attention-related signals.

#### 5. Hierarchical / Structured / External Memory

**Short-term or working memory** keeps a clean, focused context for the current turn and is often backed by checkpointed state.

**Long-term memory** stores facts, episodes, and user preferences in vector stores, key-value stores, or profile and collection-based memory systems, then retrieves them on demand.

Agents may also decide what to remember, forget, or summarize over time.

#### 6. Context Isolation

This is especially important in multi-agent systems.

Sub-agents or specialized agents handle separate concerns, such as one agent for research and retrieval and another for final synthesis.

This prevents cross-contamination and helps keep each agent's context small and focused.

This pattern is common in CrewAI, AutoGen, LangGraph multi-agent workflows, and Google's ADK.

#### 7. Offloading & Checkpointing

Store verbose tool outputs, full histories, or intermediate states outside the LLM context, such as in agent state, the filesystem, or a database.

Inject only summaries or pointers when needed.

LangGraph's persistent checkpointers are especially useful for long-horizon tasks.

#### 8. Agentic / Self-Managing Memory

In more advanced setups, the agent itself decides when to compress, prune, retrieve, or reflect on its own context through tools, memory policies, or reflection loops.

Emerging directions include plan-aware frameworks such as `PAACE` and RL-trained memory agents.

#### Additional Techniques

Additional techniques include attention calibration to offset positional bias, structured note-taking, and keeping toolsets minimal to reduce noise.

### Documented Framework Primitives

The broad strategies above are more general than any one framework. The table below shows which memory-management primitives are explicitly documented in framework docs:

| Framework | Documented mechanism | Why it helps | Limitation |
| --- | --- | --- | --- |
| [LangGraph](https://docs.langchain.com/oss/python/langgraph/add-memory) | short-term thread memory, long-term store, checkpoints, and explicit options to **trim**, **delete**, or **summarize** messages | keeps conversations within the context window and lets applications retrieve state across turns | you still need to choose what to trim, summarize, or search |
| [AutoGen](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/memory.html) + [model contexts](https://microsoft.github.io/autogen/dev/user-guide/core-user-guide/components/model-context.html) | bounded chat contexts such as `BufferedChatCompletionContext`, plus a `Memory` protocol with `query` and `update_context` | avoids replaying the full transcript and supports retrieval-based injection into context | bounded context can drop useful older facts; retrieval can still bring back distractors |
| [CrewAI](https://docs.crewai.com/en/concepts/memory) | unified `Memory` with composite ranking over **semantic similarity**, **recency**, and **importance**, plus scoped memory views | narrows recall and injects relevant context before tasks | memory quality still depends on scope design and ranking quality |
| [LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/memory/) | token-limited short-term memory plus optional long-term `MemoryBlock`s such as `FactExtractionMemoryBlock` and `VectorMemoryBlock` | flushes older chat into structured long-term memory instead of keeping everything in the active window | extracted facts and retrieval blocks are still lossy or selective |
| [Google ADK](https://google.github.io/adk-docs/sessions/) + [memory tools](https://google.github.io/adk-docs/sessions/memory/) | explicit separation of `Session`, `State`, and `Memory`, with `PreloadMemory` and `LoadMemory` tools and `add_session_to_memory` | separates current conversation state from longer-term memory and supports on-demand or turn-start retrieval | memory still has to be curated, and more retrieved context can still dilute the prompt |

### How the Patterns Show Up in Practice

The first pattern is **bounded short-term context**.

LangGraph documents short-term memory at the thread level and then explicitly offers message trimming, deletion, and summarization to manage long conversations. AutoGen documents `BufferedChatCompletionContext` as a way to keep only the most recent messages, specifically to avoid context overflow. LlamaIndex documents a token-limited short-term memory that keeps only the last messages that fit within a configured budget.

The second pattern is **long-term storage outside the active prompt**.

LangGraph provides a long-term store. LlamaIndex provides long-term `MemoryBlock`s. ADK distinguishes between current-session context and longer-term memory. In all three cases, the idea is the same: do not keep the entire raw history in the prompt if it can be stored externally and retrieved later.

The third pattern is **retrieval-based reinjection**.

AutoGen's memory API is explicit here: `query` retrieves relevant information and `update_context` injects it into the agent's model context. CrewAI documents recall using composite scoring, and says agents recall relevant context before each task. ADK provides `PreloadMemory` to retrieve memory at the beginning of each turn and `LoadMemory` to retrieve it only when the agent decides it is useful.

The fourth pattern is **structured or scoped memory**.

CrewAI documents scoped memory views so agents can use either shared crew memory or a narrower private scope. LlamaIndex documents multiple memory block types with priorities and insertion behavior. ADK splits `Session`, `State`, and `Memory` into separate roles. This structure helps reduce prompt clutter by keeping different kinds of state from collapsing into one undifferentiated transcript.

The fifth pattern is **checkpointing and offloading**.

LangGraph's checkpointing model is especially relevant here: agent state can persist across steps and sessions without forcing every intermediate detail back into the prompt. ADK's distinction between `Session`, `State`, and `Memory` serves a similar purpose by keeping longer-lived state outside the current model input.

The sixth pattern is **system-managed memory policies**.

LangGraph, AutoGen, and LlamaIndex all expose mechanisms that let the application decide when to trim, summarize, flush, or retrieve memory. That is not the same as solving dilution automatically, but it does move memory management from an accidental transcript into an explicit policy layer.

Two other strategies matter at the system level even when they are not always first-class framework APIs: **repositioning** and **context isolation**.

Repositioning is usually implemented in prompt assembly rather than as a named memory primitive: the system decides where retrieved evidence, summaries, and instructions appear in the final prompt. Context isolation often appears through graph nodes, scoped memory, or sub-agent boundaries that keep separate tasks from sharing one giant running transcript.

### Practical Takeaway

So the documented framework response to attention dilution is not "just use a bigger context window." It is:

- keep a smaller active context
- store additional information outside the prompt
- retrieve only some of it later
- optionally summarize or transform history before reinserting it
- isolate tasks or scopes when shared context would become noisy
- treat memory management as an explicit policy, not an append-only log

That is valuable, but it is still only **context management**, not a fix for the zero-sum attention mechanism itself.

### Limits Frameworks Still Cannot Remove

Even with these features, the underlying problem remains.

- A trimmed buffer may remove an older but critical constraint.
- A summary may omit nuance you later need.
- A retrieval layer may surface keyword-matching distractors.
- A memory store may grow large enough that poor recall policy simply reintroduces dilution in a different form.

So the practical lesson is: frameworks help by giving you **better memory plumbing**, but the developer still has to decide **what belongs in working memory**, **what should be externalized**, and **what deserves to be brought back into the prompt at each step**.

## Bottom Line

Long context is best thought of as **finite, biased, dilutable RAM**, not perfect memory.

The best agents treat context as a scarce resource that must be actively managed. Without deliberate context engineering, more memory often just means more distraction.

That is why, in production systems, **smaller and smarter context often beats bigger context**.

## References

- Liu, Nelson F., et al. [*Lost in the Middle: How Language Models Use Long Contexts*](https://aclanthology.org/2024.tacl-1.9/). *Transactions of the Association for Computational Linguistics*, 2024.
- Xiao, Guangxuan, et al. [*Efficient Streaming Language Models with Attention Sinks*](https://openreview.net/forum?id=NG7sS51zVF). ICLR 2024.
- Hong, Kelly, Anton Troynikov, and Jeff Huber. [*Context Rot: How Increasing Input Tokens Impacts LLM Performance*](https://research.trychroma.com/context-rot). Chroma Technical Report, July 14, 2025.
- Anthropic. [*Introducing Contextual Retrieval*](https://www.anthropic.com/engineering/contextual-retrieval). Engineering post, September 19, 2024.
