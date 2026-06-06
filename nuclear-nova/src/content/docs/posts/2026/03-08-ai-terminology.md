---
title: "AI Terminology: Agents, Skills, RAG, MCP, and the Layers Beneath the Hype"
date: 2026-03-08
author: BZ
description: "A practical mental model for AI terminology: prompt, context, memory, agents, RAG, function calling, MCP, workflows, skills, and sub-agents."
categories:
  - AI ENGINEERING
tags:
  - agents
  - rag
  - mcp
---

<!-- more -->

How many of these terms do you actually recognize?

If most of them feel familiar-but-fuzzy, that is exactly the problem this post is trying to solve.

In AI discourse, new names appear constantly: **agent**, **memory**, **RAG**, **function calling**, **MCP**, **workflow**, **skill**, **sub-agent**. They sound like separate inventions, but many of them are just different ways of packaging the same underlying pattern.

This is the mental model I want to argue for:

> Most AI terminology is just naming different ways of adding **context**, **code**, and **control** around a language model.

Some of the hype is real. Some of it is just old wine in new bottles. Either way, once you see the layers clearly, most new terms stop feeling mysterious.

## Start With the Old Primitive: the LLM

The starting point of all this chaos is the **language model**.

Early language models were honestly pretty dumb. As parameter counts kept increasing, though, something changed: the outputs started to look intelligent enough that we felt the need to rename the category. Put **large** in front of it, and now we have **Large Language Models**, or **LLMs**.

At its core, an LLM still only does one thing: it predicts the next token.

If you use it only as a raw next-token machine, it still looks limited. But the moment you artificially split interaction into two roles, question and answer, it starts to feel like a conversation.

That is the first trick in this whole story: the model itself did not fundamentally change, but the interface around it did.

## Prompt, Context, and Memory Are Just Naming Layers of Input

Imagine the LLM as an employee. I will call it **Little L**.

Little L has one unusual constraint: it is fundamentally a **one-question, one-answer** worker. Ask something, get one response, and that interaction is over.

So the real challenge becomes: how do you squeeze more value out of this limited interaction mode?

The first step is to name the interaction itself:

- Each request becomes a **prompt**.
- The background information inside that prompt becomes **context**.
- Previous conversation history, when stuffed back into later prompts, becomes **memory**.

That is already most of the magic people attribute to "chat."

The model is not truly remembering anything across calls. You are simply replaying earlier information by injecting it into the next prompt. If the history gets too long, you summarize it and compress it, and now you have a more efficient form of memory.

In other words, a lot of "multi-turn intelligence" is just careful prompt construction.

## When the Model Is Not Enough, You Add Code Around It

Soon you hit the next limitation: Little L cannot browse the web, call APIs, or search your local files by itself.

Even if you give it access to a computer, the model still only emits text. It does not independently execute logic. So the first naive workflow looks like this:

1. Ask the model what it needs.
2. Perform the external action yourself.
3. Feed the result back into the next prompt.

That works, but it also makes the human the middleware layer.

So you automate the middleware. You write a program that sits between you and the model, handles browsing or tool execution, and forwards results back and forth.

Now the outside world sees:

`You -> Program -> LLM`

Give that program a science-fiction name and suddenly it becomes an **agent**.

This is why I say a lot of so-called agent behavior is really just the parts that **do not require intelligence**. The intelligent part is still mostly the model. The rest is orchestration.

## RAG Is Just Another Way of Injecting External Context

Once the agent can use tools, another question appears: can it search local documents, databases, or knowledge bases?

Yes, and that is where **RAG** enters.

**Retrieval-Augmented Generation** sounds grand, but the practical idea is simple:

1. retrieve relevant information from outside the model's parameters
2. inject that information into the prompt
3. let the model answer with better context

Web search is one version of this. Document retrieval is another. Vector databases are just one implementation strategy for finding semantically similar chunks instead of doing only exact matching.

So the right mental model is not "RAG is a magical new intelligence layer." It is closer to:

> **RAG is search plus prompt injection of retrieved context.**

## Agents Need Contracts, Not Just Natural Language

Once you have an agent between the user and the model, another engineering problem shows up.

If the model tells the agent what to do using unconstrained natural language, the agent becomes painful to implement. Parsing free-form text is brittle. The agent needs structure.

So you define a strict agreement: the model must express tool intent in a rigid format, often something like structured JSON.

That agreement is called **function calling**.

It is best understood as a contract between the **model** and the **agent**. The model says, in a machine-readable format, what function should be called and with what arguments.

But what if the tools are no longer embedded directly inside the agent process? What if they live in separate services?

Now you need a different contract:

- a way to list available tools
- a way to invoke them
- a way to pass results back

That agreement is **MCP**, the **Model Context Protocol**.

Function calling and MCP are often mixed together, but they live at different layers.

| Term | What it really is | Boundary |
| --- | --- | --- |
| **Function calling** | a structured output contract | **model <-> agent** |
| **MCP** | a tool discovery and invocation contract | **agent <-> tools/services** |

Once you see that distinction, a lot of confusion disappears.

## The Agent Is Mostly a Messenger

At this point, the architecture looks much clearer:

- The **LLM** is the part that produces language and fuzzy judgment.
- The **agent** translates model intent into actions and routes information around.
- **Tools** or **MCP services** do the deterministic external work.

That is why the agent often feels smart while actually doing a lot of mechanical work.

If the LLM is the philosopher, the agent is the messenger:

> "I do not create the knowledge. I move it."

And from the user's side, the interface can take many forms:

- a CLI
- an IDE plugin
- a desktop assistant
- a chat app

The UI changes. The basic architecture often does not.

## Workflows, LangChain, and Skills Sit on the Same Spectrum

Now suppose the task is:

Extract text from an English PDF, translate it into Chinese, and save the result as Markdown.

You could let an agent plan this freely, but that is often wasteful. Some steps are deterministic:

- PDF extraction
- file conversion
- saving output

Only the translation part genuinely benefits from model intelligence.

That means the whole pipeline may not need a free-form agent at all. A fixed program or workflow could be more stable and cheaper.

This is where concepts like **LangChain**, **workflow tools**, and **skills** start appearing. They are not separate universes. They are different points on the same control-versus-flexibility spectrum.

When requirements become messy, the spectrum becomes useful. Maybe the input can be PDF, Word, TXT, or PPT. Maybe the output can be Markdown, HTML, PDF, or even an image. Hard-coding every permutation becomes ugly, but letting the model improvise everything is also unstable.

So you meet in the middle:

- keep reusable scripts for deterministic work
- write instructions that explain when to use which script
- let the model choose dynamically when flexibility is needed

That instruction bundle is often what people now call a **skill**.

A skill is not mystical. In many cases, it is basically:

- a prompt
- some instructions
- a small collection of scripts or tools

## Sub-Agents Are Mostly a Context Management Trick

As tasks get larger, context windows get crowded.

The obvious response is to create **sub-agents**: smaller isolated agents responsible for specific subtasks. Their main benefit is often not magical specialization, but **context isolation**.

Instead of one giant conversation polluted by everything, you create smaller workspaces:

- one agent for searching
- one for code changes
- one for document parsing
- one for synthesis

That is often the whole trick.

## One Table for the Whole Terminology Pile

Here is the compressed version of the landscape:

| Term | Practical meaning |
| --- | --- |
| **Prompt** | the current request sent to the model |
| **Context** | background information included in that request |
| **Memory** | prior conversation replayed or summarized into later prompts |
| **Agent** | a program that wraps the model and coordinates actions |
| **RAG** | retrieving external information and injecting it into context |
| **Function calling** | structured communication between model and agent |
| **MCP** | structured communication between agent and external tools |
| **Workflow** | a mostly fixed pipeline of steps |
| **Skill** | reusable instructions plus scripts/tools for a task pattern |
| **Sub-agent** | a separate agent context for a subtask |

## The Unifying Methodology

If I had to compress the whole article into one sentence, it would be this:

> Nearly all of these concepts are just ways to automatically stuff more useful context into prompts while moving deterministic work out of the human's hands.

That is why I said agents are made of everything that does not need intelligence.

- Anything deterministic belongs in **code**.
- Anything fuzzy belongs in the **model**.

And the systems we keep naming are mostly different ways of drawing that boundary.

## Why This Still Matters

Today, token cost is still a real constraint. That is one reason people care so much about workflows, memory compression, RAG, and orchestration.

But if production-grade models eventually run cheaply on personal machines, a lot of this vocabulary may stop feeling profound. Many of today's grand concepts will look, in hindsight, like transitional engineering patterns.

When that day comes, the terminology pile will feel much less mysterious.

And a lot of it will turn out to have been obvious all along.
