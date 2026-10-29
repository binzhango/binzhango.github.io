---
title: "Decision Models for LangChain Agents: Jev, Tool Review, and Measured Results"
date: 2026-10-29T12:00:00-04:00
author: BZ
description: "How Jev adds advisory playbook hints and an enforced refund-risk review gate to a LangChain agent, how Python turns them into decisions, and what local-agent and decision-only experiments actually show."
excerpt: "Separate predictions, policy, and execution with Jev and LangChain. Follow a refund through middleware, inspect review thresholds, and compare recorded decision latency with full-agent outcomes."
categories:
  - AI ENGINEERING
tags:
  - decision-models
  - jev
  - langchain
  - agent-harness
  - human-in-the-loop
  - evaluation
---

An agent can generate a valid refund tool call and still make the wrong decision. The amount might fit the schema and stay below the monetary limit, while the customer has supplied no reason to receive a refund.

A **decision model** addresses the judgment between understanding a request and choosing what happens next. In this project, Jev reads supplied context and explicit questions, then returns typed predictions: a support category, a probability, or a score on a defined rubric.

Python uses the support category to select an advisory playbook hint and the proposed-refund risk score to require human review. The generative LLM still chooses actions, proposes tool calls, and writes the final response.

I built [Jev Decision Lab](https://github.com/binzhango/jev_langchain) to make that separation inspectable. It runs the same support task through a normal LangChain agent and a Jev-assisted agent, using the same local LLM and independent copies of the same order data. The useful question is specific: **does adding these decision points improve the agent's observed behavior, and what does it cost?**

<!-- more -->

## Where a decision model fits

A typical tool-using agent repeats a loop: read the conversation, produce a response or tool call, receive the tool result, and continue. Some judgments happen inside a generation the LLM already needs. Others can be isolated into questions such as “Which workflow fits?” or “Should this proposed action receive human review?”

TypeSafe calls Jev a *System One* model. Its interface evaluates state against questions and returns structured decisions rather than conversational text. That makes it a complementary component for bounded judgments within an agent harness—the code surrounding model calls, tools, state, and execution. The project takes inspiration from LangChain's [Building a Harness with Jev](https://www.langchain.com/blog/building-a-harness-with-jev).

The comparison has two paths. In the normal path, the local LLM chooses its workflow and next action within its generation. In the assisted path, Jev supplies triage and risk predictions, Python applies the rules, and the same local LLM still proposes actions and writes responses. Both paths use one configured worker model; Jev does not choose a different worker model in this lab.

Jev triage happens before the assisted worker begins, but it only adds prompt guidance: both paths retain the same tools and all four playbooks. The enforced difference is the refund-risk gate. Jev assesses a proposed refund after hard-policy checks, using concrete arguments and verified evidence; Python pauses it for review when the score reaches the configured threshold or is unavailable.

<figure class="jev-figure">
<p class="jev-figure-title">One task, two paths: LLM decisions and Jev-assisted decisions</p>
<p class="jev-scroll-hint">On narrow screens, scroll inside the diagram to compare both paths.</p>
<div class="jev-diagram-scroll" role="region" aria-label="Normal LLM agent and Jev-assisted agent comparison, horizontally scrollable" tabindex="0">
<img class="jev-light no-lightbox" src="/diagrams/jev-decisions/refund-path.svg" alt="The same task splits into two independent runs. Left: the local LLM chooses the workflow and action, followed by Python policy and review. Right: Jev triage guides the same LLM, and Jev risk adds a review signal after hard-policy checks. Both record simulated tool effects and LLM responses for comparison." width="906" height="1120" />
<img class="jev-dark no-lightbox" src="/diagrams/jev-decisions/refund-path.dark.svg" alt="The same task splits into two independent runs. Left: the local LLM chooses the workflow and action, followed by Python policy and review. Right: Jev triage guides the same LLM, and Jev risk adds a review signal after hard-policy checks. Both record simulated tool effects and LLM responses for comparison." width="906" height="1120" />
</div>
<figcaption>Two independent runs with the same local LLM and shared policy rules; they execute sequentially. This diagram expands the refund branch using the default risk threshold of 1.0. Triage is advisory; it does not restrict tools. Hard denials and pending or rejected reviews do not execute refunds; other tools skip this gate. The normal path has no separate classifier call. Dashed arrows collect observations for comparison.</figcaption>
<p class="jev-figure-links"><a href="/diagrams/jev-decisions/refund-path.svg" target="_blank" rel="noopener">Open full size</a> · <a href="/diagrams/jev-decisions/refund-path.drawio" download>Editable draw.io source</a></p>
</figure>

This separates three responsibilities:

- **Prediction:** estimate which category or risk level matches the evidence.
- **Policy:** choose a prompt hint or enforce review from the returned prediction.
- **Execution:** validate the actual action and apply its effect.

A risk score is evidence consumed by policy. It does not grant permission, establish ownership, or prove that a refund is justified.

## Start with a concrete \$9 refund

The lab's `missing-item` scenario contains this task:

```text
The edamame was missing from ORD-1001. Please refund that item.
```

Its simulated order contains a \$16 bowl marked present and \$9 edamame marked missing. The worker can look up the order, propose a refund, or escalate to a person. Both agent paths receive the same initial facts and policy instructions.

The recorded Jev-assisted run on October 28, 2026 followed this sequence:

<div class="jev-table" aria-label="Recorded nine dollar refund walkthrough">

| Stage | Observed evidence | Consequence |
| --- | --- | --- |
| Triage | Category `refund`; confidence `1.00` | Python adds the refund playbook as a prompt hint |
| Worker proposal | Order lookup, then a \$9 refund | Middleware inspects the proposed arguments |
| Hard policy | Amount and order pass deterministic checks | Semantic risk assessment can proceed |
| Jev risk | Score `0.01`; review threshold `1.00` | This assessment adds no review requirement |
| Execution | One successful \$9 refund record | The LLM receives the tool result |
| Final response | Reports the executed refund | Response text remains separately assessable |

</div>

These values come from the [saved live comparison](https://github.com/binzhango/jev_langchain/blob/7151a23e2e95b58feba3de86c51ee8e928dd656a/docs/recordings/missing-item-live.json), not a fabricated model response. “Live” means real model inference; every business operation in the lab is simulated.

Now change the situation: the order arrived complete, but the customer asks for \$9 anyway. The amount still satisfies the hard monetary limits. Semantic justification becomes the important distinction. The agent can decline or clarify; if it nevertheless proposes a refund, the assisted path has a separate risk assessment before execution. That extra opportunity to catch an unsupported action is a design hypothesis to evaluate, not a guarantee that the classifier always catches it.

## Choice, Noul, and Score answer different questions

The [LangChain TypeSafe integration](https://docs.langchain.com/oss/python/integrations/providers/typesafe) exposes Jev through `TypeSafeClassifier`. Each invocation carries the state and named questions together. A response preserves the answers, resolved model identity, usage, and request ID.

<div class="jev-table" aria-label="Decision primitive meanings">

| Primitive | Meaning | Use in this lab |
| --- | --- | --- |
| `Choice` | Select among named alternatives; return a distribution and confidence | Select a prompt hint for status, refund, human help, or clarification |
| `Noul` | Probability that a statement is true, from 0 to 1 | Assess expressed urgency |
| `Score` | Position on an ordered rubric, with a distribution and confidence | Assess frustration or proposed-refund risk |

</div>

### Choice defines the available categories

The category question in [decisions.py](https://github.com/binzhango/jev_langchain/blob/7151a23e2e95b58feba3de86c51ee8e928dd656a/src/jev_harness/decisions.py) defines four support categories for selecting a playbook hint. This excerpt shows the actual question structure:

```python
CATEGORIES = {
    "order_status": "Checking delivery status or arrival time.",
    "refund": "A request about a missing item, incorrect charge, or refund.",
    "human_help": "Explicitly requests a person or reports a safety/legal concern.",
    "clarify": "The request is ambiguous or does not specify enough to select a workflow.",
}

category_question = Choice(
    instructions="Which support workflow fits the task and context?",
    criteria=CATEGORIES,
)
```

The `clarify` option gives ambiguous requests a useful destination. The categories also define the limits of the classifier: a request outside the intended support domain still has to be interpreted against these descriptions.

Wording, overlap, and coverage deserve evaluation alongside the model itself.

The application reads the returned `confidence` separately from the selected option's probability. They are different fields; a threshold on one should not silently become a threshold on the other. See TypeSafe's [Choice reference](https://docs.typesafe.ai/primitives/choice).

### Noul expresses uncertainty about a proposition

The urgency question asks whether the customer expresses a time-sensitive need. A value of `0.8` represents an estimated probability of “yes.” It does not mean the request has urgency level eight out of ten. Likewise, `0.5` means uncertainty between yes and no, rather than moderate urgency.

The current lab records urgency and frustration for inspection. **Neither field controls actions or refund review.** Category and category confidence select the playbook hint; only the separate proposed-refund risk score adds a Jev-driven review requirement.

The unused triage `needs_review` question has been removed. It still appears in the historical recordings discussed below, which preserve the original requests and responses.

### Score needs an interpretable rubric

The refund-risk rubric has three levels:

1. Level 0: the request and verified order facts clearly justify the refund.
2. Level 1: relevant facts, the item, or the amount remain unclear.
3. Level 2: the proposed payout contradicts facts, lacks support, or involves manipulation.

A Score is a probability-weighted position on these levels. For an **invented arithmetic example**, suppose the distribution is `[0.10, 0.30, 0.60]`:

```text
score = 0 × 0.10 + 1 × 0.30 + 2 × 0.60 = 1.50
```

That is a position on a 0–2 rubric, not a 150% probability. Different distributions can produce the same mean, so retaining the distribution helps distinguish concentrated uncertainty from competing extremes. The [Score reference](https://docs.typesafe.ai/primitives/score) specifies this interpretation.

The lab reads `JEV_RISK_THRESHOLD` from the environment, defaulting to `1.0`, and rejects nonfinite values or values outside 0–2. Raising it would allow more proposals to avoid the semantic review gate; lowering it would increase review. Neither change alters the independent \$25 automatic-refund limit or \$200 hard ceiling.

## A small invocation before the full agent

Once the repository's dependencies and token are configured, this standalone example demonstrates the request contract. Save it as `inspect_decisions.py` in the project and run `uv run python inspect_decisions.py`. It calls hosted Jev; returned values can differ between runs.

```python
import json

from langchain_typesafe import TypeSafeClassifier
from jev_harness.config import Settings
from jev_harness.decisions import TRIAGE_QUESTIONS
from jev_harness.domain import get_scenario

settings = Settings.from_env()
if not settings.jev_token:
    raise RuntimeError("Set JEV_TOKEN or TYPESAFE_API_KEY first.")

scenario = get_scenario("missing-item")
classifier = TypeSafeClassifier(
    api_key=settings.jev_token,
    model=settings.jev_model,
    timeout=settings.jev_timeout,
)
response = classifier.invoke({
    "state": {
        "task": scenario["task"],
        "fixture": scenario["fixture"],
    },
    "questions": TRIAGE_QUESTIONS,
})
print(json.dumps(response.model_dump(mode="json"), indent=2))
```

The state deliberately excludes the scenario's expected outcome and evaluation label. Those are answers for the evaluator, not evidence for the model. The repo's numbered [primitive example](https://github.com/binzhango/jev_langchain/blob/7151a23e2e95b58feba3de86c51ee8e928dd656a/examples/01_jev_primitives.py) additionally wraps the response with the submitted state, questions, timing, and source label.

This setup uses a local generative LLM plus a hosted decision model. The task and fixture are sent to Jev for triage; refund assessment also sends the proposed arguments, verified order, and up to the last 12 conversation messages. The teaching fixtures are synthetic. A real application would need to choose that outbound evidence deliberately.

## Connect predictions to LangChain middleware

The integration uses custom `AgentMiddleware` in [agent.py](https://github.com/binzhango/jev_langchain/blob/7151a23e2e95b58feba3de86c51ee8e928dd656a/src/jev_harness/agent.py). The three hooks align with the moments when new evidence becomes available. LangChain documents these lifecycle points in its [custom middleware guide](https://docs.langchain.com/oss/python/langchain/middleware/custom).

### Before the run: classify and select a prompt hint

`before_agent` performs triage only for the Jev-assisted path and reuses its recorded result when present. The application then applies this rule, abbreviated from the source:

```python
def playbook_hint(triage):
    answer = triage["answers"]["category"]
    if answer["confidence"] < 0.6:
        return "clarify", "Category confidence is below 0.60."
    return answer["choice"], "Use the selected category."
```

The threshold belongs to the application. It expresses a preference for clarification when the category confidence is low; it is not a demonstrated optimal cutoff.

### Around generation: supply the selected playbook

`wrap_model_call` appends the selected playbook hint to the assisted worker's system message. For refunds, the playbook asks it to look up the order, identify the item and amount, and clarify or request review when evidence is insufficient.

Both paths already have all four playbooks in their base prompt. Jev supplies an explicit prompt hint; it does not give the assisted agent exclusive access to policy knowledge. Both also request one tool call per step, and the harness stops after eight model steps.

There is a practical limit here: choosing a playbook changes instructions, not the tool allowlist. A clarification hint still leaves the refund tool available. The hard tool boundary must therefore validate actual actions even when the selected hint sounds conservative.

### Around tools: inspect the proposed refund

`wrap_tool_call` checks refund eligibility before asking Jev about semantic risk. An oversized or invalid refund can be rejected using ordinary code without an additional inference request.

For an eligible refund, risk assessment sees the user's task, the actual tool arguments, verified order data, and recent conversation. The core review condition is:

```python
score = risk.get("answers", {}).get("risk", {}).get("score")
needs_review = score is None or score >= settings.risk_threshold
```

This is an excerpt of the decision rule, not the complete middleware. The full implementation combines it with the hard-policy review requirement and the worker's optional `request_review=True`. A low risk score cannot erase either of those reasons to review.

The review action is a LangGraph `interrupt`, backed by SQLite checkpoints. The reviewer can approve, edit, or reject. Resuming supplies a `Command(resume=...)`, and edited arguments pass through hard policy again before the tool executes.

## Keep authorization in deterministic code

The [domain layer](https://github.com/binzhango/jev_langchain/blob/7151a23e2e95b58feba3de86c51ee8e928dd656a/src/jev_harness/domain.py) applies the same rules to both paths. These rules check facts and amounts that do not need model judgment.

<div class="jev-table" aria-label="Refund policy and semantic review boundaries">

| Condition | Implemented behavior |
| --- | --- |
| Wrong order or customer | Deny |
| Nonpositive, nonfinite, or fractional-cent amount | Deny |
| More than \$200 or remaining order value | Deny |
| An existing successful refund for this order | Deny another refund, including a split payment |
| More than \$25, within hard limits | Require human review |
| Worker explicitly requests review | Require human review |
| Jev risk missing or at least the configured threshold (default `1.0`) | Require review on the assisted path |
| Otherwise eligible | Execute only after any required review is resolved |

</div>

The tool implementation checks policy again while applying the simulated payment. Middleware also caches completed tool results by call ID so checkpoint replay can return an existing result rather than repeating an effect.

This is useful protection within the sandbox; connecting a real payment service would also require durable idempotency at that service boundary.

The hard policy deliberately does **not** prove missing-item entitlement. It does not independently enforce that the proposed amount equals a particular missing item's price. That judgment remains with the worker, the risk assessment, and any reviewer. If a business can express entitlement as a deterministic rule over trusted records, it should enforce that rule directly too.

Reviewed edits are rechecked against monetary and ownership constraints. The current implementation does not run a fresh Jev assessment of the edited proposal: the person takes responsibility for the semantic change. That boundary should be explicit when extending the demo.

### An unavailable decision is not a low-risk decision

If triage fails, the assisted path records the error and selects the clarification playbook. A failed or missing risk assessment requires review. The run is marked degraded and excluded from automatic decision-quality scoring, while its actual tool effects remain inspectable.

This fallback still has limits. A triage failure alone does not mechanically forbid every refund: clarification is prompt guidance, and subsequent refund gating follows its own rules. The trace makes that behavior visible instead of silently treating an outage as successful classification.

## Compare two different experimental questions

The lab supports two experiments because decision quality and agent outcomes are different measurements.

The **full-agent comparison** starts both paths from the same task and fixture. They share tools, worker model, generation settings, policy knowledge, and review capability.

Each receives independent mutable business data and checkpoints. The runner executes them sequentially to avoid making two requests compete for the same local server.

This experiment measures the whole Jev-assisted harness. It adds a playbook prompt hint and a refund-risk gate together. It does not demonstrate enforced workflow routing or isolate the effect of either addition. After the first divergent action, the agents may have different histories. A later disagreement is therefore not necessarily a disagreement over identical evidence.

The **decision-only benchmark** removes the agent loop. Jev and the existing local LLM receive the same state and question definitions across 18 authored cases: six intent classifications, six urgency judgments, and six risk scores.

Labels stay outside inference inputs. The local LLM returns minimal prompted JSON; it is not asked to generate Jev's full distributions. Strictly invalid responses count as failures without a hidden repair call. A separate lenient score applies bounded normalization to the same recorded text, keeping the strict result visible.

This isolates the decision interface more closely, although the hardware, hosting, network, and decoding methods still differ.

The [benchmark implementation](https://github.com/binzhango/jev_langchain/blob/7151a23e2e95b58feba3de86c51ee8e928dd656a/src/jev_harness/benchmark.py) records shuffled case order, counterbalanced provider order, a dataset hash, and warmups excluded from the reported measurements.

## What the recorded runs show

### Full-agent outcomes: both paths completed the refunds

Two live integration comparisons from October 28 used the worker model ID `qwen3.8-27b-mlx` in LM Studio and resolved Jev model `jev-1.14.0`. Each scenario ran once, normal path first. A test driver approved the \$40 refunds on both paths.

<div class="jev-table" aria-label="Recorded full agent comparison timing">

| Scenario | Normal active time | Assisted active time | Actual effects |
| --- | --- | --- | --- |
| Confirmed missing \$9 item | 26.30 s | 29.41 s | Both issued exactly one \$9 refund |
| Confirmed missing \$40 dinner | 24.30 s | 23.17 s | Both paused for approval, then issued \$40 |

</div>

Each path used three worker calls per scenario. The assisted path added two Jev calls, totaling approximately 450 ms for the \$9 case and 405 ms for the \$40 case. Active time excludes queueing and human review waiting. These are integration examples with no counterbalanced scenario repetition, so their timing differences do not establish an agent speedup. The [observed-results report](https://github.com/binzhango/jev_langchain/blob/7151a23e2e95b58feba3de86c51ee8e928dd656a/docs/observed-results.md) links both complete recordings.

Action correctness also differed from response correctness.

After successfully issuing the \$9 refund, the normal agent incorrectly described it as pending approval. The assisted response correctly reported execution but added payment-processing language unsupported by the fixture. Passing action checks did not prove that either response was completely grounded.

### Decision-only calls: lower latency and one fewer format failure

The October 29 recording ran each of the 18 decision cases once, with one excluded warmup per provider. It used the same local worker model and hosted Jev. Results below are recomputed from the [complete benchmark recording](https://github.com/binzhango/jev_langchain/blob/7151a23e2e95b58feba3de86c51ee8e928dd656a/docs/recordings/decision-benchmark-live.json).

<div class="jev-table jev-results" aria-label="Recorded decision benchmark results">

| Metric | Local LLM | Jev |
| --- | --- | --- |
| Valid and correct / all attempts | 17 / 18 | 18 / 18 |
| Correct / valid responses | 17 / 17 | 18 / 18 |
| Correct after lenient parsing (post-hoc) | 18 / 18 | 18 / 18 |
| Strict accuracy, 95% Wilson interval | 74.2%–99.0% | 82.4%–100% |
| Lenient accuracy, 95% Wilson interval | 82.4%–100% | 82.4%–100% |
| Formatting failures | 1 | 0 |
| Median valid-response latency | 11.158 s | 0.151 s |
| p95 valid-response latency | 45.328 s | 0.219 s |
| Median latency, including failures | 12.414 s | 0.151 s |
| Noul Brier score, valid responses | 0.000 | 0.001 |
| Risk-score MAE, valid responses | 0.000 | 0.132 |
| Reported output tokens, all attempts | 5,275 | 499 |

</div>

The local LLM's single invalid answer was `{"decision": 1.0}` where the contract required `{"noul": 1.0}`. This demonstrates a format failure, not an incorrect urgency judgment.

The updated benchmark reports a second result after stripping a complete JSON code fence or unwrapping a sole `decision` field. It then applies the same type and range checks, rejecting duplicate or conflicting keys, arbitrary prose, and type coercion. This recovers the one field-name error and gives **both providers 18/18 under lenient parsing**. It makes no additional inference call and does not alter the original strict failure.

That normalization rule was added after inspecting this failure. The historical lenient result is therefore a **post-hoc sensitivity analysis**, not a new benchmark run or an independently established reliability gain. The 95% Wilson intervals use only the first recorded repetition of each of the 18 authored cases; repeating a case does not increase that sample size. They illustrate finite-sample uncertainty, without establishing population accuracy, paired statistical significance, or generalization.

Every valid discrete decision was correct for both providers. For risk, “correct” means choosing the expected side of the `1.0` review threshold, not matching the exact rubric value.

Brier score measures squared probability error against the binary urgency labels; MAE measures absolute distance from authored risk levels. Lower is better for these metrics. The local model matched those labels exactly on its valid answers, while Jev returned softer values. This run observed lower request latency for hosted Jev and one fewer strict output-contract failure. It does not establish superior semantic judgment or a general reliability advantage.

The valid-response latency medians differ by about 74 times on this setup. That ratio compares medians; it is neither a per-case average speedup nor a full-agent speedup. Hosted Jev and the local 27B generative model use different hardware, serving stacks, and decoding behavior. The LLM's reported output includes reasoning tokens, and the providers use different tokenizers. These measurements do not establish intrinsic model speed. Eighteen simple teaching cases cannot establish broad calibration or monetary savings.

## Decide whether the extra model helps

Adding a fast classifier is most promising when there is a bounded judgment with clear downstream consequences: selecting one of several workflows, escalating an ambiguous request, or reviewing a concrete action. The benefit comes from making that boundary measurable and possibly improving its decisions.

It can be unnecessary for a direct lookup or a rule already determined by trusted structured data. It also adds a dependency and inference time. In the full-agent recordings above, Jev did not eliminate any worker calls.

A useful timing model is:

```text
assisted active time
  = worker generations + Jev requests + tool execution + harness overhead

user elapsed time
  = active time + queueing + review waiting
```

To improve the experiment, expand the authored cases with ambiguous intent, conflicting facts, mixed requests, near-threshold scores, and unsupported instructions embedded in customer text. Separate development examples used to tune questions or thresholds from held-out evaluation cases. Compare false allows, unnecessary reviews, task completion, and response grounding rather than optimizing a single aggregate accuracy figure.

Calibration needs its own evidence. Among many cases assigned a probability near 0.8, roughly 80% should satisfy the proposition if predictions are calibrated for that population. A confident individual answer or a low Brier score on six easy cases cannot establish that property. Typed answers make integration easier; their meanings still require validation.

### Preserve the rules and evidence across review

A paused action may resume after the process restarts. The runner saves a non-secret execution identity that includes the worker settings, Jev settings, and risk threshold. Changing `JEV_RISK_THRESHOLD` prevents resuming a pending comparison under different rules; the reviewer continues the original experiment rather than silently changing its policy.

The current [SQLite store](https://github.com/binzhango/jev_langchain/blob/7151a23e2e95b58feba3de86c51ee8e928dd656a/src/jev_harness/store.py) separates comparison metadata, individual run fields, and append-only event rows. Updates persist changed fields and newly appended events. Older JSON-based databases migrate transactionally on startup, with a `comparisons.pre-fields.db` backup. Legacy `workflow` fields and events become `playbook_hint` in storage, while exported documentation recordings remain unchanged.

## Run the lab and inspect the evidence

The repository uses Python 3.12 or newer and a locked `uv` environment. The inspected lockfile includes LangChain `1.4.7`, `langchain-typesafe` `0.0.1a3`, and LangGraph `1.2.18`. The project now explicitly pins `langchain-typesafe==0.0.1a3`; this is a tested prerelease, not a stable API guarantee. Validate upgrades before relying on identical behavior.

With the repository available locally, start the dashboard and API from its root:

```bash
bash start.sh
```

The launcher installs locked dependencies, builds the frontend, and asks for missing local-model settings and optional credentials. Start the local model server separately for live inference, then open [localhost:8000](http://localhost:8000). The project reads `JEV_TOKEN`, falling back to `TYPESAFE_API_KEY`, and passes it explicitly to the classifier.

The API is intended for local use: it checks localhost/loopback Host headers and accepts browser requests only from the same origin or the explicit local Vite origins on port 5173. These checks are not authentication for a deployed service.

For a walkthrough that needs no model calls, select **Scripted fixture** in the dashboard. Or run this CLI command using a separate data directory:

```bash
JEV_LAB_DATA_DIR=artifacts/blog-cli \
  uv run jev-lab compare --scenario missing-item --execution fixture
```

Fixture mode runs scripted worker behavior and synthetic Jev values through the policy and review machinery. It teaches the control flow; its outcomes and timings are not evidence about model performance.

For a real paired comparison and repeated evaluation:

```bash
JEV_LAB_DATA_DIR=artifacts/blog-cli \
  uv run jev-lab compare --scenario missing-item

JEV_LAB_DATA_DIR=artifacts/blog-cli \
  uv run jev-lab evaluate --execution live --repeat 3 --review reject \
  --output artifacts/blog-evaluation.json
```

The explicit reviewer policy matters. Rejecting all reviews can prevent a justified \$40 refund from completing. Approving all reviews can permit an unsupported payout that a model correctly flagged. Record that choice and interpret the outcome checks accordingly. Only one runner process can own a data directory at a time.

To repeat the focused decision experiment, avoid simultaneous comparisons on the same model server and run:

```bash
uv run jev-lab benchmark --repeat 3 --output artifacts/benchmark
```

It writes `report.json`, `report.md`, and a standalone `report.html`, including strict and lenient scores and first-pass accuracy intervals. Three repetitions can reveal variability, but they reuse the same authored cases rather than supplying new independent tasks. The [evaluation guide](https://github.com/binzhango/jev_langchain/blob/7151a23e2e95b58feba3de86c51ee8e928dd656a/docs/evaluation.md) and [benchmark methodology](https://github.com/binzhango/jev_langchain/blob/7151a23e2e95b58feba3de86c51ee8e928dd656a/docs/decision-benchmark.md) explain the denominators and failure handling.

When reading a trace, follow the evidence in order: submitted state, question and rubric, returned prediction, Python rule, proposed action, review resolution, tool result, and final response. That sequence explains where the agent's behavior changed and whether the change helped. The decision model supplies a judgment; the application makes that judgment consequential, and the recorded outcome tells us whether it was useful.
