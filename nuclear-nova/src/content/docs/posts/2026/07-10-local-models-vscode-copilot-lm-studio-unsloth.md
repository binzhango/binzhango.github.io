---
title: "Use Local Models in VS Code Copilot with LM Studio and Unsloth Studio"
date: 2026-07-10
author: BZ
tutorial: true
description: "How to connect VS Code Copilot Chat to local OpenAI-compatible models served by LM Studio or Unsloth Studio, including the BYOK utility-model setting that prevents agent errors."
categories:
  - AI ENGINEERING
  - SOFTWARE ENGINEERING
tags:
  - vscode
  - copilot
  - lm studio
  - unsloth
  - local llm
  - byok
---

<!-- more -->

<div class="tutorial-prerequisites">
  <strong>Before you begin</strong>
  <span>Install VS Code with Copilot Chat, then download a model in LM Studio or Unsloth Studio.</span>
</div>

## Start a Model Server

Open LM Studio or Unsloth Studio, load your model, and start its OpenAI-compatible server. Keep the service running while you use the model in VS Code.

You only need one service running at a time. Start with one, confirm it works, and then add the other.

These are the two models and addresses from my setup. Your values may be different.

| Service | Where it runs | Model | Server address |
| --- | --- | --- | --- |
| LM Studio | Another computer on my network | `ornith-1.0-35b` | `http://192.168.0.71:1234` |
| Unsloth Studio | The same computer as VS Code | `Qwythos-9B-Claude-Mythos-5-1M` | `http://127.0.0.1:8888/v1` |

### If you use LM Studio on another computer

The LM Studio computer and the VS Code computer must be connected to the same reachable network. LM Studio must allow network connections instead of accepting requests only from itself.

My LM Studio address is:

```text
http://192.168.0.71:1234
```

### If you use Unsloth Studio on the VS Code computer

My Unsloth Studio address is:

```text
http://127.0.0.1:8888/v1
```

`127.0.0.1` means “this computer,” so this address works only when Unsloth Studio and VS Code run on the same machine.

> Copy the server address exactly as your service displays it. Some addresses include `/v1`, while others do not.

<div class="tutorial-success">
  <strong>You should now see</strong>
  <span>A running server and an address beginning with <code>http://</code>.</span>
</div>

## Open the VS Code Model Setup

In VS Code:

1. Open **Copilot Chat**.
2. Click the current model name to open the model picker.
3. Expand **Other Models**.
4. Click the gear icon beside **Other Models**.

<figure class="copilot-figure copilot-figure--compact">
  <img src="/assets/images/2026/vscode-copilot-other-models-gear.png" alt="The Other Models section in the VS Code Copilot model picker with its configuration gear highlighted" width="664" height="134" loading="lazy" />
  <figcaption><strong>Open model settings.</strong> Expand <strong>Other Models</strong>, then click the gear.</figcaption>
</figure>

The gear opens the model setup. From this screen:

5. Click **Add Models**.
6. Select **Custom Endpoint** from the menu.

<figure class="copilot-figure copilot-figure--panel">
  <img src="/assets/images/2026/vscode-copilot-add-models-custom-endpoint.png" alt="The VS Code Add Models menu with Custom Endpoint selected" width="840" height="578" loading="lazy" />
  <figcaption><strong>Add the provider.</strong> Click <strong>Add Models</strong> and choose <strong>Custom Endpoint</strong>.</figcaption>
</figure>

A custom endpoint is simply the server address that lets VS Code send messages to your local model.

<div class="tutorial-success">
  <strong>You should now see</strong>
  <span>The model management screen with <strong>Custom Endpoint</strong> selected.</span>
</div>

## Add the Model Configuration

Use the following configuration as a template. It includes both of my services, but you can keep only the block for the service you use.

```json
[
  {
    "name": "lm",
    "vendor": "customendpoint",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "ornith-1.0-35b",
        "name": "ornith-1.0-35b",
        "url": "http://192.168.0.71:1234",
        "toolCalling": true,
        "vision": true,
        "maxInputTokens": 128000,
        "maxOutputTokens": 16000
      }
    ]
  },
  {
    "name": "unsloth",
    "vendor": "customendpoint",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "Qwythos",
        "name": "Qwythos-9B-Claude-Mythos-5-1M",
        "url": "http://127.0.0.1:8888/v1",
        "toolCalling": true,
        "vision": true,
        "maxInputTokens": 128000,
        "maxOutputTokens": 16000
      }
    ],
    "apiKey": "${input:chat.lm.secret.71788f5f}"
  }
]
```

Before saving, check these values:

- Change each `id` and `name` if you use a different model.
- Change each `url` to the server address shown by your service.
- Keep `vendor` as `customendpoint`.
- Keep `apiType` as `chat-completions`.

The Unsloth `apiKey` in my configuration points to a secret saved by VS Code. Do not replace it with a real secret directly in the file. If VS Code asks for a key and your local server does not require one, use the placeholder expected by your server setup.

Save the configuration. If the new models do not appear in the model picker, reload VS Code.

<div class="tutorial-success">
  <strong>You should now see</strong>
  <span>Your LM Studio or Unsloth Studio model listed in VS Code.</span>
</div>

## Enable the BYOK Utility Model

This required setting is easy to miss. **BYOK** means “bring your own key,” which is the name VS Code uses for models supplied through custom providers.

To configure it:

1. Open VS Code **Settings**.
2. Search for:

```text
Chat: Byok Utility Model Default
```

3. Select **Main Agent Model** from the dropdown.

<figure class="copilot-figure copilot-figure--banner">
  <img src="/assets/images/2026/vscode-copilot-byok-utility-model-default.png" alt="VS Code setting Chat: BYOK Utility Model Default configured to use the Main Agent Model" width="2292" height="242" loading="lazy" />
  <figcaption><strong>Required setting.</strong> Use the main agent model for Copilot's BYOK utility tasks.</figcaption>
</figure>

Copilot performs small background tasks in addition to answering the main request. This setting tells Copilot to use your selected local model for those tasks too.

Without this setting, agent mode can fail with:

<figure class="copilot-figure copilot-figure--banner">
  <img src="/assets/images/2026/vscode-copilot-no-utility-model-error.png" alt="VS Code Copilot error stating that no utility model is configured for copilot-utility-small while the selected main agent model is BYOK" width="1880" height="120" loading="lazy" />
  <figcaption><strong>What this setting prevents.</strong> Agent mode cannot continue until a utility model is available.</figcaption>
</figure>

A separately configured `Chat: Utility Model` or `Chat: Utility Small Model` overrides this default. For this simple setup, **Main Agent Model** is enough.

<div class="tutorial-success">
  <strong>You should now see</strong>
  <span><strong>Main Agent Model</strong> selected, with no utility-model error in Copilot.</span>
</div>

## Test the Local Model

Test one service at a time:

1. Confirm that LM Studio or Unsloth Studio is still running.
2. Return to Copilot Chat in VS Code.
3. Open the model picker.
4. Select your local model.
5. Send a simple message such as: `Explain what this project does.`
6. Switch to agent mode and ask: `List the files in this workspace.`

After the configuration is saved, both custom models appear in the Copilot model picker. The provider name is shown beside each model: `lm` for LM Studio and `unsloth` for Unsloth Studio. Click the model you want to use; the check mark shows the currently selected model.

<figure class="copilot-figure copilot-figure--panel">
  <img src="/assets/images/2026/vscode-copilot-final-local-model-selection.png" alt="The VS Code Copilot model picker showing the configured LM Studio and Unsloth Studio models, with the Unsloth model selected" width="908" height="692" loading="lazy" />
  <figcaption><strong>Select a local model.</strong> Both providers are available, and the check mark identifies the active model.</figcaption>
</figure>

The first prompt checks basic chat. The second checks whether the model can work with Copilot's tools.

Here is a more realistic agent-mode test. I asked Copilot to check a Rust project for vulnerabilities. The local model read `Cargo.toml`, ran `cargo-audit`, and summarized the security findings and suggested fixes.

<figure class="copilot-figure copilot-figure--wide">
  <img src="/assets/images/2026/vscode-copilot-local-model-sample-agent-chat.png" alt="A sample VS Code Copilot agent chat using a local model to inspect a Rust project, run cargo-audit, and report dependency vulnerabilities" width="1870" height="1452" loading="lazy" />
  <figcaption><strong>End-to-end result.</strong> The local model reads the project, uses a terminal tool, and explains the security findings.</figcaption>
</figure>

This confirms that the local model can do more than answer a chat message: it can inspect workspace files, request terminal commands, and explain the results. Review each proposed command before allowing Copilot to run it.

<div class="tutorial-success">
  <strong>You should now see</strong>
  <span>A response from your local model. In agent mode, supported models can also read files and request tools.</span>
</div>

If both prompts work, the setup is complete. You can now switch between LM Studio and Unsloth Studio from the Copilot model picker.

## What the Configuration Options Mean

You do not need to change these settings when copying my configuration, but it helps to know what they do:

| Option | Plain-language meaning |
| --- | --- |
| `vendor: customendpoint` | The model comes from a server you provided. |
| `apiType: chat-completions` | VS Code and the server use the same chat-message format. |
| `url` | The address VS Code uses to reach the model server. |
| `toolCalling: true` | The model can ask Copilot to use workspace tools. |
| `vision: true` | The model can receive images. |
| `maxInputTokens` | The maximum amount of text the model can receive. |
| `maxOutputTokens` | The maximum length of the model's answer. |

These options describe what a model supports; they do not add new abilities. If your model cannot use images or tools, set `vision` or `toolCalling` to `false`. If the model server runs out of memory, lower the token limits.

## Troubleshooting

### Agent mode shows the utility-model error

Set `Chat: Byok Utility Model Default` to **Main Agent Model**, or explicitly configure `Chat: Utility Model` and `Chat: Utility Small Model`.

### The model does not appear in VS Code

Save the model configuration and reload VS Code. Then reopen Copilot Chat and look under **Other Models**.

Also check that the JSON has matching brackets and commas. A missing comma can prevent the configuration from loading.

### VS Code cannot reach LM Studio

Because my LM Studio endpoint is on another computer, I check that:

- VS Code and the LM Studio machine are on the same reachable network.
- LM Studio allows connections from other devices.
- The firewall on the LM Studio computer allows port `1234`.
- The URL and port match the values shown by LM Studio.

Do not expose an unauthenticated local inference endpoint to an untrusted network.

### The error includes `404` or “not found”

Check the server address. Some services require `/v1` at the end and others do not. Copy the address shown by the service and make sure `/v1` is not added twice.

### Regular chat works, but the agent test fails

First, confirm that the BYOK utility-model setting is enabled. If it is, the model may not support the tool-calling format Copilot expects. Set `toolCalling` to `false` and use that model for regular chat, or choose a model with tool support.

### Image prompts fail

Only models built to understand images can use image prompts. If your model is text-only, set `vision` to `false`.

### Responses stop early or the server runs out of memory

Lower `maxInputTokens` and `maxOutputTokens`. The values in my configuration may be too large for a different model or computer.

## Final Checklist

Before troubleshooting anything advanced, confirm each item:

- [ ] The model is loaded.
- [ ] LM Studio or Unsloth Studio is running its server.
- [ ] The model name and server address match the VS Code configuration.
- [ ] The model appears under **Other Models**.
- [ ] `Chat: Byok Utility Model Default` is set to **Main Agent Model**.
- [ ] A normal chat prompt works.
- [ ] The agent test works, if the model supports tools.

Once these checks pass, I can select either local model from Copilot's model picker and use it without changing my normal VS Code workflow.
