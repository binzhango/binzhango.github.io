---
title: "Choosing a model API"
date: 2026-11-17
author: BZ
description: "Learn what each model API returns and when to use it. Follow one booking question from a prompt to a message, a tool call, and a grounded answer."
categories:
  - AI ENGINEERING
tags:
  - llm
  - model-apis
  - tool-calling
  - embeddings
---

Your support bot receives a successful API response. It reads `choices[0].message.content` and displays an empty bubble. The traveler asked, “Does my fare include a checked bag?” The model returned a booking lookup in `message.tool_calls`. Your handler expected a sentence and missed its turn to act.

**What does one model call produce, and who acts next?** A completion gives you text to append. A chat call gives you a message to interpret. A Responses call can give you several items to process. That choice reaches into your backend. It determines what you store, what you display, and when you call the model again.

<!-- more -->

## Read an endpoint in four parts

Open a provider’s request example and its matching response together. For Chat Completions, the **input container** is `messages`. Each **content unit** is a role-tagged turn whose content can include text or supported media. The **output container** is `choices`, with a generated message in each choice. The **control plane** includes fields such as `tools` and `stream` that change how you handle that message.

Apply those four questions to the other generation APIs and you can predict the shape of the client you need.

| Generation API | Input container and unit | Output to inspect | One control that affects your code |
| --- | --- | --- | --- |
| Completions | `prompt` · text prefix | `choices[].text` | `stop` can end a continuation at a delimiter |
| Chat Completions | `messages` · role-tagged turns | `choices[].message` | With `tools`, the assistant may return `tool_calls` |
| Claude Messages | `messages` · turns containing typed blocks | One message’s `content[]` | `stop_reason` can indicate tool use |
| Gemini generateContent | `contents` · turns containing parts | `candidates[].content.parts[]` | `generationConfig` configures generation |
| OpenAI Responses | `input` · text or typed items | `output[]` | `previous_response_id` can link a stored response |

Two other familiar paths serve different operations. Use `/v1/embeddings` when you need vectors for search and `/v1/models` when you need model records. For exact methods and request bodies across all seven, [jump to the endpoint reference](#endpoint-reference).

## Prompts and message history

Consider the same support exchange in two wire formats. With the legacy `POST /v1/completions` endpoint, the application can build a single prompt ending at the point where the assistant should continue writing.

```text
User: Does my fare include a checked bag?
Assistant: What is your booking reference?
User: DEMO42
Assistant:
```

The model generates a suffix, which your code reads from `choices[0].text`. If a user types “Assistant:” into their answer, the protocol carries those characters in exactly the same string as your speaker labels. Keep this interface for a legacy caller that deliberately owns the template and needs a continuation. [OpenAI Completions reference](https://developers.openai.com/api/reference/resources/completions/methods/create)

With `POST /v1/chat/completions`, the application sends the same exchange as an array. The API can distinguish who said “DEMO42” because the role is a field on the message.

```json
{
  "model": "MODEL_ID",
  "messages": [
    { "role": "user", "content": "Does my fare include a checked bag?" },
    { "role": "assistant", "content": "What is your booking reference?" },
    { "role": "user", "content": "DEMO42" }
  ]
}
```

For a text reply, render `choices[0].message.content`. Save the whole assistant message if you will continue the conversation. Chat Completions fits an app that already stores a transcript or sends turns through a compatible gateway. The application decides which history to include on every request. [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)

## Images in Claude and Gemini

Suppose the traveler sends a photo of the booking confirmation instead of typing the reference. It is still one user turn. The question tells the model what to read, and the image supplies the evidence. The unit *inside* the message matters as much as its role.

The following excerpts encode that same turn on two native APIs. Use an image-capable model and replace `BASE64_PNG` with the encoded image bytes.

<p class="sample-label">Claude · POST /v1/messages</p>

```json
{
  "model": "MODEL_ID",
  "max_tokens": 100,
  "system": "Read the booking reference exactly as printed.",
  "messages": [{
    "role": "user",
    "content": [
      { "type": "text", "text": "What is my booking reference?" },
      { "type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": "BASE64_PNG"
      }}
    ]
  }]
}
```

<p class="sample-label">Gemini · POST /v1beta/models/{model}:generateContent</p>

```json
{
  "systemInstruction": { "parts": [{ "text": "Read the booking reference exactly as printed." }] },
  "contents": [{
    "role": "user",
    "parts": [
      { "text": "What is my booking reference?" },
      { "inlineData": { "mimeType": "image/png", "data": "BASE64_PNG" } }
    ]
  }]
}
```

**Choose the native format for the model you are integrating.** Claude puts instructions in top-level `system` and returns one assistant message. Walk its `content[]` blocks by type. Gemini names the model resource in the URL and applies the `:generateContent` method to it. Choose one response candidate, then walk its parts. Parts within a candidate belong together. Concatenating several candidates can splice alternative answers into one. [Claude Messages reference](https://platform.claude.com/docs/en/api/messages/create), [Gemini generateContent reference](https://aistudio.google.com/api-reference/generate-content/generate-content)

An image reader might return “DEMO42.” That identifies the booking. To answer the original baggage question, the backend still needs the allowance recorded against it.

## Tool calls and state

Give the model a custom function called `get_bag_allowance` with a required string argument, `booking_ref`. Both Chat Completions and Responses let the model request that function. Your backend checks the traveler’s access, validates the arguments, and runs it. The meaningful difference is how your backend recognizes the request and reconnects the result to the conversation. [OpenAI function calling guide](https://developers.openai.com/api/docs/guides/function-calling)

| At the handoff | Chat Completions | Responses with stored state |
| --- | --- | --- |
| Declare the function in `tools[]` | Put the name and parameters inside the `function` object | Put the name and parameters directly on the function tool |
| Find the requested work | Inspect `choices[0].message.tool_calls[]` | Inspect each `output[]` item with `type: "function_call"` |
| Run the function | Parse `call.function.arguments` | Parse `call.arguments` |
| Send its result | Append the assistant turn and a `tool` message to the history | Send a `function_call_output` input item and reference the previous response |

Here is the assistant message behind the empty bubble. Its only call has ID `call_bag`. The arguments arrive as a JSON-encoded string.

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [{
    "id": "call_bag",
    "type": "function",
    "function": {
      "name": "get_bag_allowance",
      "arguments": "{\"booking_ref\":\"DEMO42\"}"
    }
  }]
}
```

Suppose the booking service returns one included bag for this fictional booking. These JavaScript excerpts construct the **second request** for that single lookup. `model` is your chosen model ID and `messages` is the original chat history.

<p class="sample-label">Chat Completions · continue the transcript</p>

```js
const assistant = chat.choices[0].message;
const [call] = assistant.tool_calls;
const followUp = {
  model,
  messages: [
    ...messages,
    assistant,
    {
      role: "tool",
      tool_call_id: call.id,
      content: JSON.stringify({ included_checked_bags: 1 })
    }
  ]
};
```

<p class="sample-label">Responses · continue a stored response</p>

```js
const call = response.output.find(item => item.type === "function_call");
const followUp = {
  model,
  previous_response_id: response.id,
  input: [{
    type: "function_call_output",
    call_id: call.call_id,
    output: JSON.stringify({ included_checked_bags: 1 })
  }]
};
```

The next model call can answer, “Your fare includes one checked bag.” In Responses, the two IDs serve different purposes. `response.id` identifies the context to resume. `call.call_id` identifies which lookup produced this result. With several calls in one turn, return a result for each call ID. When chaining, resend top-level `instructions` that should still apply. [Responses migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses)

**When would you switch?** An existing Chat Completions client can handle this lookup by extending its message loop. OpenAI recommends Responses for new OpenAI projects. Its typed items are especially useful when a task mixes messages, reasoning items, and hosted tool calls. Hosted web search can execute within a response. Your custom booking function runs in your application. You can chain stored responses, use a conversation object, or replay items when you want to manage history yourself. Usage and status belong to the response object. [OpenAI migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses)

<details>
<summary>Try a text-only Responses request with curl</summary>

Set `OPENAI_API_KEY` and a Responses-capable `OPENAI_MODEL` in your environment. This shell example uses `jq` to build the request and print the text. The filter selects message items explicitly because other item types can appear first.

```bash
jq -n --arg model "$OPENAI_MODEL" \
  '{model: $model, input: "What is a checked bag? Answer in one sentence."}' |
  curl --fail-with-body -sS https://api.openai.com/v1/responses \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: application/json" --data-binary @- |
  jq -r '.output[] | select(.type == "message")
         | .content[] | select(.type == "output_text") | .text'
```

</details>

## Deliver the result to its consumer

The next decision is what consumes the answer. A traveler can read a sentence as it arrives. A queue worker needs a complete value it can dispatch. Streaming controls <strong data-emphasis="process">when</strong> the client sees output. A response schema controls <strong data-emphasis="data">which values</strong> a completed answer can contain. An application can use both.

### Streaming events

A Chat Completions text stream can deliver the booking answer in pieces. These selected server-sent events assemble into “One checked bag.”

```text
data: {"choices":[{"index":0,"delta":{"content":"One checked"}}]}

data: {"choices":[{"index":0,"delta":{"content":" bag."}}]}

data: [DONE]
```

Render text deltas as they arrive. For a streamed tool call, accumulate argument fragments for that call, wait for completion, then parse and validate the full arguments. A disconnect after half of “DEMO42” leaves an interrupted turn. Executing the lookup at that point would use an incomplete identifier. [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create), [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)

The events mirror the object under construction. Claude opens a message, starts and updates individual content blocks, then emits `message_stop`. Responses distinguishes events such as `response.output_text.delta` and `response.completed`. Gemini exposes a separate `:streamGenerateContent` method. Give errors, completion, and usage their own handling in the client. [Claude streaming](https://platform.claude.com/docs/en/build-with-claude/streaming), [Responses streaming](https://developers.openai.com/api/docs/guides/streaming-responses), [Gemini streaming](https://aistudio.google.com/api-reference/generate-content/stream-generate-content)

### Structured output

Suppose a worker routes incoming questions to `booking_lookup` or `policy_search`. Asking for JSON expresses a preference. JSON mode addresses syntax. Both `{"route":"booking_lookup"}` and `{"route":"maybe"}` parse successfully, but only the first names a queue that exists.

For a supported model, use strict Structured Outputs and require an enum. This is the OpenAI Responses `text.format` object.

```json
{
  "type": "json_schema",
  "name": "support_route",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "route": { "type": "string", "enum": ["booking_lookup", "policy_search"] }
    },
    "required": ["route"],
    "additionalProperties": false
  }
}
```

Chat Completions takes `response_format.type: "json_schema"` and nests `name`, `strict`, and `schema` under `response_format.json_schema`. Handle refusals and incomplete responses before dispatch. The schema constrains the possible queues. Whether the model picked the right queue is a separate behavior to evaluate. Use a response schema to produce data for a worker and a tool call when the model needs the worker’s result to finish its answer. [OpenAI Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs)

## Where embeddings and model discovery fit

### Retrieval with embeddings

“Does **my fare** include a checked bag?” requires the booking record. “What is the **oversized baggage policy**?” requires the relevant passage from the policy library. This second question is a good use for embeddings.

| When | Operation | What your application keeps |
| --- | --- | --- |
| When a policy changes | Send its passages to `POST /v1/embeddings` using an embedding model | Each `data[].embedding` vector and its source passage ID. For a batch, `data[].index` links each vector to its input. |
| When a question arrives | Embed the question with a compatible configuration and search the vector index | The matching passages and their links |
| When the UI needs a reply | Send the question and retrieved **passage text** to a generation endpoint | A readable answer grounded in those passages |

For example, a request body can be as small as `{"model":"EMBEDDING_MODEL_ID","input":"oversized baggage fees"}`. In a response excerpt, `{"index":0,"embedding":[0.0123,-0.0456,0.0789]}` represents the first input. These three illustrative coordinates stand in for the full vector.

The embedding endpoint encodes text as numbers. Your index compares the vectors. A search UI can stop once it has passages and links. Adding the final model call is useful when the traveler needs an explanation. If the retrieved policy is wrong, inspect search before changing the answer prompt. [OpenAI embeddings reference](https://developers.openai.com/api/reference/resources/embeddings/methods/create), [Embedding models and vector index compatibility](/posts/2026/09-05-rag-engineering-04-embedding-profile-index-contract/)

### Model discovery

Before deploying the assistant, use `GET /v1/models` to read visible IDs from `data[].id` or `GET /v1/models/{model}` to inspect one resource. If generation reports `model_not_found`, repeat the lookup using the same provider, project, and credentials. A typo or access restriction can explain a missing ID. A timeout leaves the question unresolved. [OpenAI list models](https://developers.openai.com/api/reference/resources/models/methods/list), [OpenAI retrieve model](https://developers.openai.com/api/reference/resources/models/methods/retrieve)

A visible ID answers the **resource** question. A minimal image or tool request tests the **model capability** and **endpoint protocol**. When that request works over HTTP but fails through your SDK or gateway, compare the base URL, serialized fields, SDK version, and provider’s compatibility notes. Groq, DeepSeek, Mistral, and OpenRouter document chat integrations with their own supported options. Reusing a client is practical when the fields your feature depends on work end to end. [Groq compatibility](https://console.groq.com/docs/openai), [DeepSeek first API call](https://api-docs.deepseek.com/), [Mistral Chat API](https://docs.mistral.ai/api/endpoint/chat), [OpenRouter quickstart](https://openrouter.ai/docs/quickstart)

## Takeaways

For a new integration, write down what your backend must receive to finish the feature. A text suffix points to a legacy completion. A conversation points to a message API. A workflow that mixes tools and text benefits from typed output items. A search index needs vectors, and a model picker needs metadata.

Test the first response that requires your application to **do something**. For the booking assistant, that is a tool request with an empty text field. If the handler can identify the function, run the lookup, and return its result to the right conversation, you have tested the part of the contract the traveler depends on.
