# Memory-Informed Decisions: Combining Hindsight and Jev in a LangChain Agent

Working draft · November 5, 2026

Status: architecture and experiment design. The combined integration has not been implemented or benchmarked. This file stays outside the published content collection while the example and evidence are developed.

## Opening

An agent can retrieve the right fact and still choose the wrong action. It can also make a reasonable decision from incomplete context and miss something a previous session already established.

Hindsight and Jev address different parts of this problem. **Hindsight supplies persistent evidence. Jev evaluates bounded decisions.** The application connects both to the worker model, current records, and execution rules.

My earlier projects explored these parts separately: [Hindsight Memory Agents](https://github.com/binzhango/hindsight-memory-agents) recovered project constraints across fresh processes; [Jev Decision Lab](https://github.com/binzhango/jev_langchain) added workflow guidance and proposed-action review to a support agent.

The next question is whether combining them improves complete tasks. A higher retrieval score or a well-formed decision response alone cannot answer that.

## Follow one request across sessions

Use a synthetic support scenario: **“The replacement has the same problem as yesterday.”**

The request omits the earlier item and issue. A previous session contains a confirmed description and replacement history. The current order service contains the latest fulfillment and refund state.

| Evidence | Where it comes from | What it establishes |
| --- | --- | --- |
| Earlier issue and confirmed resolution | Scoped Hindsight recall | Context for the follow-up |
| Current order, ownership, and refund state | Authoritative application tools | Whether the proposed action is eligible now |
| Workflow category and action-risk assessment | Jev | A prediction used by application policy |
| Tool execution result | Application tool | What actually happened |

Historical memory helps interpret the request. It does not replace the current order record or authorize a payment.

## Proposed architecture

The article's main figure should show two distinct paths:

1. **Read:** authenticated scope → Hindsight recall → labelled evidence supplied to Jev triage and the worker.
2. **Act:** worker proposal → deterministic checks → Jev risk assessment → execute, clarify, or human review.

A feedback arrow should connect **verified outcomes** to a deliberate retain operation. Record source IDs, event time, and outcome status. Model predictions and unverified response text must not silently become confirmed facts.

The diagram should visibly distinguish recalled history, current authoritative records, predictions, and effects. Use an editable draw.io overview with readable desktop/mobile layouts, following the existing article conventions.

## Extend the existing harness

| Existing hook | Proposed change | Evidence to record |
| --- | --- | --- |
| `before_agent` | Resolve scope, recall history, then classify the request | Query, returned fact IDs, selected workflow |
| `wrap_model_call` | Supply the same history to the worker as evidence | Exact memory payload and model input |
| `wrap_tool_call` | Assess the concrete proposal with current tool results and relevant history | Arguments, policy outcome, Jev score, review decision |
| Completion path | Retain eligible verified outcomes | Source identity, event time, operation completion |

Recall once per defined decision boundary and preserve the resulting snapshot through review. If a resumed action needs fresh evidence, record that refresh explicitly. Keep existing tool idempotency and policy revalidation intact.

Separate a missing first-time bank, no relevant recall results, and a retrieval failure. Evaluate the fallback behavior instead of treating all three as empty memory.

The current Jev setup calls a hosted service. Adding memories to its state also sends that selected history to Jev; make this data flow explicit in the example and use synthetic fixtures.

## Keep internal memory ranking as a separate experiment

Hindsight also supports a TypeSafe decision-model provider for internal operations. That enables a second integration: Jev can participate in memory processing rather than the application's workflow decisions.

For the first experiment, keep Hindsight's retrieval configuration fixed. Otherwise a result could mix improved retrieval, changed agent decisions, and changed latency without revealing which change mattered.

Verify the published project's pinned server image against the required configuration before documenting runnable settings. Current documentation alone does not prove support in that image.

## Evaluate four configurations

| Configuration | Hindsight memory | Application-level Jev |
| --- | --- | --- |
| Baseline | Off | Off |
| Memory only | On | Off |
| Jev only | Off | On |
| Combined | On | On |

Use the same worker model, tool implementations, execution rules, and current records. Both memory configurations receive equivalent preloaded history. Give each run an isolated bank and independent business state; one run must not teach another the evaluation answer.

Within a memory configuration, give Jev and the worker access to the same permitted historical evidence. Keep expected answers and evaluator labels out of all model inputs.

### Cases that can distinguish the configurations

- **Missing history:** an abbreviated follow-up depends on an earlier session.
- **Changed decisions:** old history conflicts with a newer confirmed event.
- **Distracting memories:** similar issues belong to another order or project.
- **Scope isolation:** another user's history must not be returned.
- **Insufficient evidence:** clarification is the correct outcome.
- **Unsupported action:** the worker proposes an action current facts do not justify.

Use development cases to tune questions and thresholds, then freeze them before held-out evaluation. Counterbalance execution order and record repeat-level variation. Do not treat repeated copies of a case as independent new scenarios.

### Measure the behavior readers care about

- **Action correctness:** correct tool, target, amount, and executed effect.
- **Response grounding:** the final answer accurately describes verified facts and effects.
- **Review quality:** missed reviews and unnecessary reviews, reported separately.
- **Task completion:** includes correct clarification, refusal, or escalation where appropriate.
- **Cost and latency:** recall, Jev calls, worker calls, writes, and complete task time.

Also inspect retrieval failures separately from decision failures. Report per-case changes and denominators before presenting an aggregate score.

## Results section: evidence required before publication

The existing Hindsight 8/8 factual result and Jev's historical decision benchmark are background evidence only. They do not measure this combination.

Before replacing this section with conclusions, preserve:

1. A runnable integration at a pinned GitHub revision.
2. The frozen development/evaluation split and scoring rules.
3. Raw inputs, recalled evidence, decisions, tool effects, and final responses for all four configurations.
4. Recomputed scores and timing summaries, including failures.
5. At least one complete success trace and one failure or regression trace.

The article's conclusion should follow those results. If memory plus Jev does not outperform memory alone, explain the observed tradeoff rather than describing the combination as an accuracy improvement.

## Sources and starting points

- [Hindsight introduction](/posts/2026/11-05-hindsight-agent-memory-benchmark/)
- [Jev decision-model article](/posts/2026/10-29-decision-models-jev-langchain/)
- [Hindsight example wrapper](https://github.com/binzhango/hindsight-memory-agents/blob/1f4f684f9aeb58a007c2cf2ad35506cfd3a50c9b/src/memory_agents/app.py)
- [Jev harness middleware](https://github.com/binzhango/jev_langchain/blob/7151a23e2e95b58feba3de86c51ee8e928dd656a/src/jev_harness/agent.py)
- [LangChain TypeSafe integration](https://docs.langchain.com/oss/python/integrations/providers/typesafe)
- [Hindsight recall API](https://hindsight.vectorize.io/developer/api/recall)
- [Hindsight decision models](https://hindsight.vectorize.io/developer/decision-models)
- [Hindsight decision-model configuration](https://hindsight.vectorize.io/developer/configuration#decision-model)

Presentation: short paragraphs, selective bold, distinct section accents, one main architecture figure, a worked evidence-to-action trace, and compact comparison tables. Preserve the readability improvements made in the first article.
