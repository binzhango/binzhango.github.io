---
title: "A Git-Native Message Channel for Local Coding Agents"
date: 2026-06-18
author: BZ
description: "How I use a Git ref as a local message channel between Claude Code and Codex for implementation, independent review, and iterative bug fixing."
categories:
  - AI ENGINEERING
  - SOFTWARE ENGINEERING
tags:
  - coding agents
  - git
  - multi-agent systems
---

<!-- more -->

## Before: One Agent Writes and Reviews Its Own Code

My previous local development workflow was simple:

1. Ask Claude Code to implement a feature.
2. Ask Claude Code to review the same implementation.
3. Ask it to fix anything it found.

This works, but it has a weakness: **self-review is not independent review**.

The same agent that made the design decisions also reviews those decisions using the same context. It may repeat its original assumptions and overlook the same edge cases, incorrect behavior, security vulnerabilities, or missing tests.

I could manually copy Claude Code's result into Codex, ask Codex to review it, then copy Codex's feedback back into Claude Code. For a complex task with several implementation and review rounds, that becomes tedious and error-prone.

I wanted the two agents to communicate without making me their message broker.

## After: Claude Code Implements, Codex Reviews

My new workflow gives the agents separate responsibilities:

- **Claude Code is the developer.** It implements features and fixes defects.
- **Codex is the code reviewer.** It reviews correctness, edge cases, security risks, and test coverage without changing the code.

Both agents work in the same local Git repository. They communicate through messages stored under a custom Git ref.

```text
refs/agents/<task-id>/messages
```

This ref does not appear in the normal branch list. It does not add a file to the working tree, and the messages are not part of the product branch's commit history.

The workflow needs only two message operations:

- **Read:** read the latest message or the most recent three to five messages.
- **Write:** append a new JSON message without changing previous messages.

At the workflow level, messages are append-only. Git's reflog provides the message history, while the current ref points to the newest message marker.

## Message Format

Every message uses a consistent JSON structure:

```json
{
  "id": 12,
  "from": "claude",
  "type": "developer",
  "reply_to": null,
  "msg": {
    "review": "Implemented token refresh and updated the authentication middleware.",
    "risk": [
      "Concurrent refresh requests may race.",
      "Expired refresh tokens need an explicit integration test."
    ],
    "done": [
      "Implemented refresh-token rotation.",
      "Added unit tests.",
      "Ran the authentication test suite successfully."
    ],
    "pending": [
      "Add a lower-priority metric for refresh failures."
    ]
  },
  "session_id": "<claude-or-codex-session-id>",
  "goal": "Implement refresh-token rotation for the authentication service.",
  "code_commit": "<git-commit-sha>",
  "timestamp": "2026-06-18T09:30:00-04:00"
}
```

The fields have specific purposes:

- `id` is an auto-incrementing message ID.
- `from` identifies the agent that wrote the message.
- `type` identifies the agent's role, such as `developer` or `reviewer`.
- `reply_to` connects a review or fix to an earlier message.
- `msg.review` explains what the agent changed or reviewed.
- `msg.risk` records possible defects, vulnerabilities, and uncertain behavior.
- `msg.done` lists completed work and verification results.
- `msg.pending` lists incomplete or lower-priority work.
- `session_id` makes it possible to trace the originating Claude Code or Codex session.
- `goal` preserves the original request, or the approved plan when the agent is working in plan mode.
- `code_commit` identifies the exact code revision associated with the message.
- `timestamp` records when the message was created.

`code_commit` is important. The message describes the work, while the commit contains the actual diff. A reviewer can inspect exactly the revision the developer intended, even if the branch moves later.

## Workflow in Motion

The complete loop looks like this:

![Animated workflow showing Claude Code implementing, Git refs carrying JSON messages, Codex reviewing, and the agents repeating until verification](/assets/images/2026/agent-review-workflow.gif)

### 1. Claude Code Starts the Task

Before changing code, Claude Code reads the most recent three to five messages from the task's ref.

This gives it a compact view of:

- the original goal
- the current implementation status
- previous review findings
- completed verification
- remaining work
- the commit it should continue from

Claude Code can then focus on implementation without loading another agent's entire conversation history.

### 2. Claude Code Appends a Developer Message

After completing the implementation, Claude Code commits the code and appends a new message.

The message explains:

- what changed
- which files or behaviors were added, updated, or deleted
- known risks
- completed tests and verification
- unfinished low-priority work
- the associated commit SHA

It does not overwrite an earlier message. Each implementation round creates a new entry.

### 3. Codex Reviews the Implementation

In a separate session, Codex reads the latest developer message first. It then reviews the referenced code revision.

Its instructions are deliberately narrow:

1. Read the message.
2. Review the code for correctness, edge cases, security risks, regressions, and test coverage.
3. Do **not** change the code.
4. Append a reviewer message whose `reply_to` points to the developer message ID.

Keeping Codex in a review-only role matters. The implementation and review responsibilities remain separate, so review findings are easier to attribute and evaluate.

### 4. Claude Code Fixes the Findings

Claude Code reads Codex's review message, fixes the confirmed defects, runs verification, commits the new code, and appends another developer message.

That message replies to the review it addressed and distinguishes:

- findings that were fixed
- findings that were investigated but not reproduced
- risks that remain
- tests that now prove the behavior

### 5. Codex Reviews Again

Codex reads the new developer message and reviews the new commit.

The loop continues until the reviewer finds no blocking issues:

```text
implement → message → review → message → fix → message → review
```

Complex tasks can take several turns, but each agent receives only the context it needs for its current responsibility.

## Why Git Refs Work for This

Git already provides most of the local coordination mechanism:

- a custom ref can live outside `refs/heads/*`, so it is not a normal branch
- updating a ref does not modify tracked project files
- a reflog records the history of ref updates
- each message can reference an exact code commit
- no server, database, or message queue is required

The write operation creates a unique marker commit and updates the custom ref with `git update-ref --create-reflog`. The JSON message is stored in the reflog entry. The read operation uses the reflog to return the latest entry or a short recent history.

There is one important technical distinction:

> The ref stores the current message pointer; the reflog stores the message history.

Therefore, "never change old messages" is a rule enforced by the wrapper command: it appends a new ref update and never rewrites an earlier entry during an active task.

Reflogs can eventually expire during Git maintenance, so this is a local working channel rather than permanent audit storage. If I need a permanent human-readable record, I can export the final conversation summary before cleaning up the task ref.

## Benefits

### Independent Cross-Validation

An agent is often less effective at challenging its own implementation assumptions. Claude Code and Codex provide two independent passes over the same task:

- Claude Code focuses on building and fixing.
- Codex focuses on finding defects and missing evidence.

This does not guarantee perfect code, but it reduces the chance that one agent's blind spot survives every round.

### Separate Context Windows

The agents do not share one continuously growing context window.

Claude Code keeps an implementation-focused context. Codex keeps a review-focused context. The Git ref messages transfer only the goal, status, risks, verification, and commit references needed for the handoff.

This reduces distraction and prevents a long implementation transcript from weakening the review.

### Less Human Copy and Paste

I no longer need to copy Claude Code's summary into Codex and then copy Codex's findings back into Claude Code.

My role changes from transporting messages to supervising the workflow and making decisions when the agents disagree.

### Easier Human Review and Rollback

Every implementation message points to a Git commit, and every review message points back to the implementation it reviewed.

Before deployment, I can read the short message history and final summary to understand:

- what changed
- what risks were found
- which defects were fixed
- what verification passed
- what remains pending

The commits still contain the complete code changes when I need deeper inspection or rollback.

### No Agent Framework Required

This workflow does not need a complex multi-agent framework.

It does not need:

- a message queue
- a database
- an orchestration server
- shared conversation memory
- another service running in the background

For my local coding workflow, a small wrapper around Git's ref and reflog commands is enough.

## Practical Rules

<p class="takeaway-lede">Ten guardrails keep the agent loop reliable without turning it into a framework.</p>

<ol class="takeaway-grid" aria-label="Ten practical rules for the agent workflow">
  <li class="takeaway-card">
    <span class="takeaway-number" aria-hidden="true">01</span>
    <div><strong>Keep messages valid</strong><p>Write compact, single-line JSON that every agent can parse.</p></div>
  </li>
  <li class="takeaway-card">
    <span class="takeaway-number" aria-hidden="true">02</span>
    <div><strong>Never reuse an ID</strong><p>Increase message IDs monotonically within each task.</p></div>
  </li>
  <li class="takeaway-card">
    <span class="takeaway-number" aria-hidden="true">03</span>
    <div><strong>Make replies explicit</strong><p>Point <code>reply_to</code> at the exact developer message under review.</p></div>
  </li>
  <li class="takeaway-card">
    <span class="takeaway-number" aria-hidden="true">04</span>
    <div><strong>Pin the code revision</strong><p>Record the associated commit in every implementation message.</p></div>
  </li>
  <li class="takeaway-card">
    <span class="takeaway-number" aria-hidden="true">05</span>
    <div><strong>Keep roles separate</strong><p>The reviewer finds problems but does not edit the code.</p></div>
  </li>
  <li class="takeaway-card">
    <span class="takeaway-number" aria-hidden="true">06</span>
    <div><strong>Prove completion</strong><p>Report verification results instead of merely claiming the task is done.</p></div>
  </li>
  <li class="takeaway-card">
    <span class="takeaway-number" aria-hidden="true">07</span>
    <div><strong>Keep context lean</strong><p>Read a small recent message window unless older context is necessary.</p></div>
  </li>
  <li class="takeaway-card">
    <span class="takeaway-number" aria-hidden="true">08</span>
    <div><strong>Append, never rewrite</strong><p>Add a new message while preserving every earlier message.</p></div>
  </li>
  <li class="takeaway-card">
    <span class="takeaway-number" aria-hidden="true">09</span>
    <div><strong>Write atomically</strong><p>Use compare-and-swap ref updates so concurrent writers cannot silently collide.</p></div>
  </li>
  <li class="takeaway-card">
    <span class="takeaway-number" aria-hidden="true">10</span>
    <div><strong>Keep a human gate</strong><p>Summarize the final state for approval before deployment.</p></div>
  </li>
</ol>

## Final Result

This creates a small local multi-agent development loop:

- Claude Code writes the code.
- Codex independently reviews it.
- Git refs carry structured messages between them.
- Git commits preserve the complete implementation history.
- I supervise the result instead of manually moving text between sessions.

The idea is intentionally simple. I am not trying to build a general-purpose agent platform. I only want two coding agents, with different roles and separate context windows, to collaborate reliably inside the same local repository.

For that job, Git is already enough.
