---
title: "From Prompt to Response: A Step-by-Step Walkthrough of LLM Inference"
date: 2026-03-07
author: BZ
description: "A cleaner mental model for LLM inference: prompt templating, tokenization, scheduling, prefill, KV cache, decode, and the split between the inference engine and the model."
categories:
  - LARGE LANGUAGE MODELS
tags:
  - llm fundamentals
---

<!-- more -->

## What Happens Between a Prompt and the Final Response

From input to output, a prompt generally goes through seven steps: **request packaging**, **tokenization**, **inference scheduling**, **prefill**, and **decode** before the result is returned.

At the core, the model first processes the full input during the **prefill** stage, computing contextual representations and building the **KV cache**. It then enters an `autoregressive generation loop`, predicting only the next token at each step.

This inference pattern is fundamentally **autoregressive**: the model predicts the next token based on the preceding context, without updating its parameters during the run. The **Transformer**, meanwhile, provides the computational backbone through its internal attention and feed-forward network structure.

Why is this framing important? Because it cleanly separates the **system layer** from the **model layer**, and it also distinguishes **prefill** from **decode**.

Many people struggle to answer this question clearly not because they lack understanding, but because they mix different layers together, making the explanation feel messy and unstructured.

Let's explain each step in detail.

> A simple mental model:
> the inference engine decides **how** the request runs efficiently, while the model decides **what** token comes next.

## Step 1: From Raw Prompt to Token IDs

What we type into the chat box is natural language, but the model does not consume raw text directly.

Before the input is passed to the model, the serving layer typically formats the conversation messages, such as system, user, and assistant turns, into a model-specific prompt template, often with special tokens or delimiters added.

After that, the text goes through a tokenizer and is split into a sequence of tokens. For example, OpenAI's open-source `tiktoken` explicitly states that it is a **BPE tokenizer** for models.

In other words, from the model's perspective, text is first converted into a sequence of discrete **token IDs** rather than being treated as "sentences" directly.

This layer is easy for many people to overlook, but it is critical. All subsequent inference is built on top of the token sequence.

Whether you input a Chinese sentence, an English paragraph, or a block of code, the model's first step is always to convert it into token IDs.

## Step 2: Serving, Scheduling, and Batching Before Model Execution

In real production systems, once a request arrives, it usually does **not** go straight onto the GPU for execution.

In real-world deployments, requests often first go through an inference serving framework such as **TGI** or **vLLM** before reaching the model. These systems handle request queuing, dynamic batching, cache management, streaming output, and other inference-time orchestration tasks.

The Hugging Face TGI documentation explicitly lists **continuous batching**, **token streaming**, **Flash Attention**, and **Paged Attention** as core features. The Transformers documentation on continuous batching also explains that the purpose of this kind of dynamic scheduling is to improve GPU utilization, reduce latency, and allow requests to join and leave batches dynamically at each step.

So from a systems perspective, the pipeline usually looks like this:

1. user input
2. prompt template expansion
3. tokenization
4. request scheduling and batching
5. model execution
6. detokenization and return

The significance of this step is that model inference is not a "bare run" of a single request in isolation. Instead, it is jointly organized and optimized by the inference engine together with other requests.

Strictly speaking, the tokenization we discussed in the previous stage is not part of the Transformer's forward inference itself, because the model only receives `input_ids`.

But in modern inference services, the tokenizer is often tightly integrated with the serving engine. From an engineering perspective, it may appear as though the inference engine is processing raw text directly.

For example, vLLM supports both text prompts and pre-tokenized prompts, and both modes can be executed.

Users usually send raw strings to the backend. The inference service on the backend typically owns the tokenizer, first encodes the string into token IDs, and then passes them to the model for **prefill** and **decode**.

In some architectures, tokenization is performed earlier on the client side or in a separate preprocessing layer.

## Step 3: Embeddings and Positional Information

Once the input reaches the model, the first step is not "starting to answer," but mapping token IDs into high-dimensional vectors.

This step is called **embedding lookup**. Each token looks up its own vector representation in a large embedding table. At this point, the model has truly entered numerical computation in continuous space.

The foundational Transformer paper, *Attention Is All You Need*, defines exactly this kind of attention-based sequence modeling approach.

However, token vectors alone are not enough, because the model also needs to know which token comes before and which comes after.

Early Transformers used **positional encoding**, while many later large models use **RoPE** (*Rotary Position Embedding*). The key value of RoPE is that it integrates positional information directly into the attention computation, allowing the model to preserve relative position information while processing tokens.

## Step 4: Transformer Blocks - Self-Attention, FFN, and the Core Computation of Inference

This is the most central part of the entire question.

In a typical decoder-only LLM, each layer mainly performs two things:

1. **Self-Attention**
2. **FFN / MLP** (*Feed-Forward Network*)

These are combined with residual connections and normalization. This is exactly the main structure given in the original Transformer paper.

You can think of it like this:

- **Attention** is responsible for reading the group chat.
- **FFN** is responsible for thinking it through privately and organizing the information.

### What does Self-Attention do?

It can be understood like this: the token at the current position needs to determine which tokens in the context are most relevant.

The model projects the current hidden state into three groups of vectors: **Query**, **Key**, and **Value**. It then computes attention weights using the similarity between the Query and all Keys, and finally performs a weighted sum over the Values.

The Transformer paper defines this as **Scaled Dot-Product Attention**.

For generative language models, there is another point that must be emphasized: the **causal mask**.

This means that the current position can only see itself and the tokens before it; it cannot peek at future tokens. This is what makes the model naturally autoregressive: it can only predict the next token based on the context that already exists.

Even the few-shot and in-context learning discussed in the GPT-3 paper are fundamentally built on top of this autoregressive prediction mechanism.

A simple way to understand **Q**, **K**, and **V** is:

- **Q** = what I am currently trying to find
- **K** = the index label attached to each word
- **V** = the actual information each word carries and can provide

The most intuitive analogy is a library retrieval system.

You currently have a question in mind. That is **Q** (*Query*). The subject labels on the catalog cards of books on the shelf are **K** (*Key*). The actual contents inside the books are **V** (*Value*).

The system first takes your question **Q** and compares it against all the labels **K** to see which ones are similar and relevant. For the books with higher relevance, more of their content **V** is retrieved, and these are then combined into the information needed at the current step.

In essence, the Transformer paper defines attention as: a query matches against a set of key-value pairs, and the output is a weighted sum of the values.

### What does the FFN do?

If attention is responsible for moving information in from the context, then the **FFN** is more like further processing the current position.

It does not interact across positions. Instead, it applies an independent nonlinear transformation to the representation of each token, further refining and strengthening its features.

The Transformer paper calls this a **position-wise feed-forward network**.

So a Transformer block can be roughly understood as:

> first decide which parts of the context I should attend to, and then perform a deeper feature transformation on the information I retrieved.

One important thing to notice is that throughout the entire process, both the **prefill** stage and the **decode** stage go through **self-attention** and **FFN**.

But be careful: "both go through them" does **not** mean they are executed in exactly the same way.

- **Prefill** sends the entire prompt into the model at once. At this point, every layer performs masked self-attention over that whole batch of tokens, then passes them through the FFN. Because the entire prompt is already known from the beginning, many tokens can be processed in parallel within a single request. Hugging Face also describes prefill this way: it processes the full input and builds the KV cache.
- **Decode** begins generating one token at a time. Each newly generated token still goes through every layer in the same pattern: one self-attention operation and one FFN operation.

However, decode does **not** rerun attention and FFN for all old tokens from scratch.

Once KV cache exists, the **K** and **V** of old tokens are cached. When a new token arrives, the model only needs to compute the layer representations required for this new token, perform attention against the historical K/V, and then pass the result through the FFN.

Hugging Face's official caching documentation explicitly says that during subsequent generation, only the new unprocessed token is passed in, while key/value tensors are written to and read from the cache.

In short:

The **FFN** is the feed-forward network inside each Transformer layer that comes right after self-attention, and in essence it applies an MLP independently to each token.

In standard LLMs, both **prefill** and **decode** go through self-attention and FFN. The difference is that prefill processes the entire known token sequence, while decode processes only the current new token and reuses the historical KV cache.

## Step 5: Prefill - Processing the Entire Prompt in Parallel

Many people mistakenly think that the model starts generating word by word as soon as the input arrives. In fact, that is not what usually happens.

Before generation, there is typically a very important stage called **Prefill**.

Prefill means that the model first runs the entire prompt through a full forward pass all at once.

During this stage, the model computes the hidden states at every layer for all input tokens and also builds the **KV cache** that will later be used during decode.

Hugging Face's caching documentation clearly states that the KV cache stores the key-value pairs produced by previous tokens in the attention layers so they can be reused directly during subsequent generation, thereby avoiding repeated computation.

One important characteristic of prefill is that it is usually highly parallelizable.

Because the entire input is already given, the GPU can execute many matrix operations together. So prefill is more like reading the whole question first, and its throughput is usually higher.

The vLLM documentation also explicitly classifies prefill as a stage that is more compute-bound.

You can think of prefill as a person taking an exam: prefill is the moment when they are reading the question carefully, loading the full problem into their mind, and filling in the context before they begin writing the answer, token by token.

## Step 6: KV Cache - Reusing History Instead of Recomputing It

Without **KV cache**, every time the model generates a new token, it would have to recompute the entire historical context from scratch, which would be extremely expensive.

With KV cache, however, the **K** and **V** computed for historical tokens at each attention layer are stored. At the next time step, the model only needs to compute the new **Query**, **Key**, and **Value** for the new token, and then use the new Query to match against the historical Keys stored in the cache.

Hugging Face's official documentation explains this very clearly: the goal of KV cache is to eliminate redundant computation and accelerate autoregressive generation.

In one sentence:

- **Without KV cache**: it is like rereading the entire article every time.
- **With KV cache**: it is like having already taken notes on the earlier part and now only adding the final sentence.

### Why does KV cache store only K and V, but not Q?

Whether something is worth caching does not depend on whether it is important, but on whether it will be reused later.

KV cache stores only **K** and **V**, not **Q**, not because **Q** is unimportant, but because **Q** is useful only once at the current step.

**K** and **V**, by contrast, will continue to be reused at every later step.

This is exactly how Hugging Face explains the caching mechanism: the K and V of past tokens can be cached and reused, while during inference you only need the query of the latest token to compute the current step's representation.

## Step 7: Decode - Autoregressive Token-by-Token Generation

Once prefill is complete, the model has effectively read and understood the entire input.

Next, the system takes the hidden state at the last position and maps it through the output layer into **logits** over the entire vocabulary, that is, scores for the next token.

Then, through **softmax** and a decoding strategy, the model decides which token to output next. Both the Transformer output logic and Hugging Face's generation documentation describe this process.

There is another commonly asked question here: how is the next token selected?

It is not always chosen simply by taking the token with the highest probability. Common decoding strategies include:

- **greedy decoding**
- **sampling**
- **top-k**
- **top-p**

Different strategies affect the stability, diversity, and creativity of the generated text. Hugging Face's generation strategy documentation provides a systematic explanation of this.

After that, the process enters a loop:

1. append the newly generated token to the context
2. reuse the KV cache
3. run the forward computation only for this new token
4. produce new logits
5. generate the next token

And this loop continues until a stopping condition is met.

## Inference Engine vs. LLM: Who Does What in the Generation Pipeline?

An LLM request inference process is, in essence, the following:

First, the prompt is templated and tokenized. Then it is scheduled by the inference service and sent to the GPU. The model uses embeddings and multiple Transformer blocks to complete the **prefill** stage in parallel, building contextual representations and the **KV cache**.

It then enters the **decode loop**, where it performs attention, feed-forward computation, and sampling token by token based on the cached history until generation is complete. Finally, the token sequence is detokenized back into text and returned.

This entire pipeline reflects:

- the computational mechanism of the **Transformer**
- the **autoregressive generation** paradigm
- the engineering optimizations of modern inference systems in batching, caching, and attention kernels

Doesn't that make it seem like almost all of this is the inference engine's job?

From the perspective of the overall workflow, it is indeed the inference engine handling most of the process, so that interpretation is reasonable.

But we should take one more step forward:

- From the perspective of **workflow orchestration**, the LLM itself is indeed relatively passive.
- From the perspective of **core computation and semantic generation**, the LLM is the most irreplaceable part of the entire pipeline.

If we break the full pipeline apart, the responsibilities are roughly as follows:

| Layer | Main responsibilities |
| --- | --- |
| **Inference engine / serving system** | receiving HTTP requests, performing tokenization and input processing, scheduling batching, managing the KV cache, coordinating GPU workers, streaming results back, and handling part of the sampling logic and system-level optimizations |
| **LLM model** | taking `input_ids`, converting them through embeddings, passing them through self-attention and feed-forward networks inside multiple Transformer blocks, and producing logits for the next token |

The official vLLM documentation describes these layers quite directly: at minimum, there is one API server responsible for HTTP, tokenization, and input processing, one engine core responsible for the scheduler and KV cache management, and N GPU workers responsible for executing the model's forward computation.

The Transformer paper defines the core structure as **attention plus FFN**, and the Transformers documentation also makes it clear that causal language modeling is essentially next-token prediction conditioned on the left-side context, where the logits in the model output are the predicted scores for each token in the vocabulary.

So, the inference engine determines **how to run** the process efficiently, while the model determines **what to generate**.

The former is mainly about orchestration and optimization; the latter is mainly about semantic computation and content generation.
