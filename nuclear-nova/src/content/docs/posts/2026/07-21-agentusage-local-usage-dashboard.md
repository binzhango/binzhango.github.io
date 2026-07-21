---
title: "Standing on Open Source: How a Codex Edge Case Became Agentusage"
date: 2026-07-21
author: BZ
description: "How two OpenUsage projects inspired me to investigate a Codex JSONL edge case and build a local-first dashboard for AI coding agents."
categories:
  - AI ENGINEERING
  - SOFTWARE ENGINEERING
tags:
  - codex
  - coding agents
  - rust
  - observability
  - developer tools
---

<!-- more -->

Agentusage would not exist without open source.

Two OpenUsage projects first showed me how useful a unified view of AI coding
subscriptions could be. Their work gave me a starting point, a set of ideas to
learn from, and the confidence to explore a problem in my own workflow.

One day, I opened my usage dashboard expecting a quick answer: **how much Codex
quota do I have left?**

The number was stale.

My recent sessions were on disk. Codex had written new events. Yet the latest
status was not making it to the dashboard. What looked like a small display bug
turned into a deeper question: what does “current usage” actually mean when the
source is an append-only stream of JSONL events?

This was not a failure of the open-source work I had learned from. Provider
formats change, local event histories contain edge cases, and every tool makes
reasonable choices about what to support. For me, the missing status became an
invitation to understand the data more deeply.

That exploration became
[Agentusage](https://github.com/binzhango/agentusage)—my local, open-source
dashboard for understanding how AI coding agents consume tokens, cache, credits,
and quota.

![Agentusage browser dashboard showing provider cards, usage trends, and model breakdowns](https://raw.githubusercontent.com/binzhango/agentusage/main/docs/images/browser-dashboard.svg)

## The open-source projects that made this possible

I did not begin from a blank page. I was fortunate to learn from two thoughtful
open-source projects with the same name:

- [robinebers/openusage](https://github.com/robinebers/openusage) brings usage
  and subscription limits into a polished native macOS menu-bar application.
- [janekbaraniewski/openusage](https://github.com/janekbaraniewski/openusage)
  takes a terminal-first approach, combining multiple providers, local storage,
  reports, and quota tracking.

I am grateful to both projects and their contributors. They openly shared not
only working software, but also product ideas, provider integrations, and
different ways to think about the same problem. Reading their code and using
their tools saved me an enormous amount of discovery work.

> Agentusage is not an attempt to erase that lineage. It is my contribution to
> the same open-source conversation.

The two projects proved something important: once coding agents become part of
daily work, their usage needs its own observability.

My workflow eventually pulled me toward a different set of design choices. I
wanted a small Rust tool that could read local agent history, preserve the
original events, normalize them into one data model, and serve the result
through a CLI, terminal UI, browser dashboard, and JSON API.

Most importantly, I wanted it to handle the Codex status that had gone missing
in my local data.

## The event that changed the design

Codex stores session history in JSONL files. Each line is a JSON object, and new
events are appended as the session continues.

That sounds straightforward until you inspect a real session. The file is not a
table of identical usage rows. It is a timeline containing different kinds of
information:

- session metadata and turn context;
- user and assistant messages;
- tool calls and code changes;
- cumulative token counters; and
- occasional rate-limit or quota snapshots.

The latest line is not necessarily the latest **usage** record, and the latest
usage record is not necessarily the latest **status** record.

That distinction explained what I was seeing in my local data. Historical
activity and current quota are related, but they are not the same data product.
A parser can count tokens correctly while still missing the event that says how
much of a quota window has been used and when it resets.

> An event stream records both **what happened** and **what is true now**. Those
> questions need different read paths.

Once I understood that, Agentusage stopped being a collection of parsers and
became a small ingestion system.

## How Agentusage reads the data

The architecture has two paths after ingestion. Provider history and live hook
payloads enter through separate adapters, then converge on the same normalized,
idempotent storage path.

<figure style="margin: 1.5rem 0;">
  <iframe
    src="/tools/agentusage-architecture.html?embed=1&theme=dark"
    title="Interactive Agentusage architecture diagram"
    loading="lazy"
    width="100%"
    style="display: block; width: 100%; height: auto; aspect-ratio: 950 / 1358; border: 0; border-radius: 12px;"
  ></iframe>
  <figcaption style="margin-top: 0.75rem; text-align: center;">
    <a href="/tools/agentusage-architecture.html?theme=dark">
      Open the full interactive Agentusage architecture diagram ↗
    </a>
  </figcaption>
</figure>

The interactive view lets you search components, follow the guided ingestion
and delivery paths, inspect relationships, switch themes, or export the diagram.

For historical reporting, Agentusage turns provider-specific records into a
common model: input and output tokens, reasoning tokens, cache activity, model,
project, session, tools, cost, and code changes.

For current status, it keeps the raw provider payloads and finds the newest
event that actually carries quota information. From that event it can recover
values such as the primary usage percentage, quota-window length, and reset
time.

The Codex reader also remembers a byte offset for every file. After the first
sync, it resumes where it stopped instead of scanning the complete history
again. Stable deduplication keys make repeated syncs safe, including when the
dashboard refreshes in the background.

## What I wanted the tool to feel like

I did not want a telemetry platform that required an account before showing me
my own data. Agentusage is deliberately local-first:

- **SQLite by default.** Each provider gets a local database.
- **One executable.** The browser dashboard is embedded in the Rust binary.
- **No hosted frontend.** Charts render locally and do not upload usage data.
- **Several ways in.** Use a browser, terminal dashboard, CLI report, or JSON
  API depending on the moment.
- **PostgreSQL only when requested.** A shared backend is available, but never
  required for a personal setup.

The current release reads local history from Codex, Claude Code, OpenCode, and
GitHub Copilot. The exact detail depends on what each provider exposes, but the
normalized view can include:

| Question | Agentusage can show |
| --- | --- |
| Where did the tokens go? | Input, output, reasoning, cache read/write, and totals |
| What caused the spike? | Model, project, client, tool, and language breakdowns |
| How active was I? | Requests, prompts, sessions, and code changes |
| What did it cost? | Estimated cost, AI units, and Copilot credits when available |
| Am I near the limit? | Latest known quota usage, window, and reset time |

The result is less about producing one impressive number and more about making
the number explainable.

## Try it in two minutes

Install both the full `agentusage` command and its shorter `au` alias:

```bash
cargo install --git https://github.com/binzhango/agentusage --locked --bins
```

Sync the agents you use, then launch the browser dashboard:

```bash
au sync codex
au sync claude_code
au sync copilot
au server --open
```

The dashboard runs locally at
[http://127.0.0.1:8787](http://127.0.0.1:8787). If I only want a quick answer in
the terminal, I use:

```bash
au daily --provider codex
```

And because the same data is exposed through a local API, it is easy to connect
a script or another dashboard:

```bash
curl 'http://127.0.0.1:8787/api/summary?provider=codex&window=30d'
```

## The useful lesson was not about JSONL

The JSONL event was the trigger, but the broader lesson is about observability—and
about the way open source turns a personal problem into shared learning.

A coding agent generates at least three layers of information:

1. **History** explains what happened during sessions.
2. **Normalization** makes different agents comparable.
3. **Current state** answers whether a limit is approaching right now.

Flatten those layers too early and the dashboard may look correct while telling
an incomplete story. Keep them separate, and both the history and the status
become easier to test, debug, and trust.

Agentusage is still early access, and provider formats will keep changing. That
is exactly why raw-event preservation and regression fixtures matter. If you
find a provider event that the parser misses, a sanitized example is incredibly
useful.

I want to end where the project began: with thanks to
[robinebers/openusage](https://github.com/robinebers/openusage),
[janekbaraniewski/openusage](https://github.com/janekbaraniewski/openusage), and
the contributors who make their work available to everyone. Open source gave me
something useful, taught me how others approached the problem, and made it
possible for me to build and share my own interpretation.

The project is open source under the MIT license. Try
[Agentusage on GitHub](https://github.com/binzhango/agentusage), open an issue,
or tell me which coding agent I should support next.
