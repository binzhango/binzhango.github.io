# Databricks Model

Onboard a Databricks Model Serving endpoint into VS Code Copilot's model picker.

## Repo Structure

• dbx-proxy.py # Local proxy server (strips top_p, forwards to Databricks)
• dbx-proxy-wrapper.sh # Shell wrapper for launchd (sources ~/.zshrc for DBX_TOKEN)
• dbx-proxy-service.sh # Service management (start/stop/restart/status/log)
• com.apple.launchd.plist.template # Launchd plist template (auto-start on login)
• chatLanguageModels.sample.json # VS Code model config sample
• install.sh # One-command installer

## Why a Local Proxy? 

VS Code Copilot has two compatibility issues with Databricks-hosted Claude models:

1. VS Code always sends both temperature and top_p — Claude on Databricks rejects this combination.
2. VS Code appends /chat/completions to the URL, conflicting with the Databricks `/invocations` path.

The local proxy solves both: it receives requests from VS Code, strips `top_p`, and forwards directly to Databricks with your Bearer token.

VS Code Copilot Chat 
$\rightarrow$ POST http://localhost:19000/chat/completions 
$\rightarrow$ dbx-proxy.py $\rightarrow$ strips top_p, injects Bearer token, bypasses corporate proxy 
$\rightarrow$ POST https://<workspace>.azuredatabricks.net/serving-endpoints/<name>/invocations 
Databricks Model Serving $\rightarrow$ Claude Sonnet 4.6
---

## Background: Databricks Model Serving Endpoint 

### What does a Databricks endpoint look like? 
A Databricks Model Serving endpoint exposes a hosted model (e.g. Claude, Llama, Mistral) via a REST API. The endpoint URL follows this pattern: 
```
https://<workspace-hostname>/serving-endpoints/<endpoint-name>/invocations
```

Example: https://xyz.azuredatabricks.net/serving-endpoints/databricks-claude-sonnet-4-6/invocations

Where:

• xyz.azuredatabricks.net — your Databricks workspace hostname (found in the browser URL when logged in)
• databricks-claude-sonnet-4-6 — the name you gave your serving endpoint in the Databricks UI

The request body follows the «openai Chat Completions format»:

```json
POST /serving-endpoints/databricks-claude-sonnet-4-6/invocations
Authorization: Bearer dap_...
Content-Type: application/json

{
  "model": "databricks-claude-sonnet-4-6",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 1024,
  "temperature": 0.1
}

Databricks wraps various LLMs behind this OpenAI-compatible interface, so you can use the same client code regardless of whether the underlying model is Claude, Llama, Mistral, or something else. The only difference is the model name in the request body.

---

### Why apitype: chatcompletions'?

VS Code Copilot's Custom Endpoint provider supports three API types:

| `apiType` | Protocol | Auth header | Used for |
| --- | --- | --- | --- | --- |
| `chatcompletions` | OpenAI Chat Completions (`/POST chat(/completions`) | `Authorization: Bearer` | OpenAI, Azure OpenAI, Databricks, Ollama, any OpenAI-compatible API |
| `responses` |  OpenAI Responses API | `Authorization: Bearer` | OpenAI Responses API (newer format) |
| `messages` | Anthropic Messages (`/POST /v1/messages`) | `x-api-key` | Anthropic direct API only |

**We Use `chatcompletions` ** because:
- Databricks Model Serving uses the **OpenAI-compatible REST format**
- It uses `Authorization: Bearer <token>` which matches Databricks PAT auth. 
- The `messages` type sends auth as `x-api-key` (Anthropic native) - Databricks rejects this with a 401.

---

## Why does VS Code send top_p and why does it break?

When using `chatcompletions` type, VS Code always includes both `temperature` and `top_p` in every request, e.g.:

```json
{
  "model": "...",
  "messages": [ ... ],
  "temperature": 0.1,
  "top_p": 1
}
```

The Anthropic API (which Claude is built on) does not allow both `temperature` and `top_p` to be specified at the same time — you must pick one. Databricks enforces this constraint, returning:
```json
{"error_code": "BAD_REQUEST", "message": "'temperature' and 'top_p' cannot both be specified for this model."}
```
There is no VS Code config option to suppress `top_p`, hence the proxy.

## Prerequisites

• macOS (launched service)
• VS Code with GitHub Copilot Chat extension
• Databricks workspace with a Model Serving endpoint
• Databricks Personal Access Token (PAT) - starts with dapi
• Python 3 (python3 --version)

## Quick Install

### 1. Configure dbx-proxy.py

Edit DATABRICKS_URL and the body["model"] value in dbx-proxy.py:

DATABRICKS_URL = "https://<your-workspace>.azuredatabricks.net/serving-endpoints/<endpoint-name>/invocations"
body["model"] = "<endpoint-name>"

### 2. Set your Databricks PAT

Add to ~/.zshrc:

export DBX_TOKEN=dapi...your_token...

Get your PAT: Databricks workspace $\rightarrow$ avatar $\rightarrow$ Settings $\rightarrow$ Developer $\rightarrow$ Access tokens $\rightarrow$ Generate new token.

### 3. Run the installer

```bash
source ~/.zshrc  # load DBX_TOKEN into current shell
cd path/to/vscode_dbx_model


source ~/.zshrc # load DBX_TOKEN into current shell session cd /path/to/uvscope_dbx_model 

# Interactive (will prompt for workspace and endpoint): 
./install.sh

# Or non-interactive (pass as arguments): 
./install.sh add-2008766158291623.3.azuredatabricks.net databricks-claude-sonnet-4-6
```

The installer will: 
- Validate DBX_TOKEN and proxy configuration 
- Generate the launchd plist from the template 
- Load the service (auto-starts on every login, self-restarts on crash) 
- Verify the proxy is listening on port 19000

### 4. Test the proxy

```bash
curl -s http://localhost:19000/chat/completions \ 
-H "Content-Type:application/json" \ 
-d '{"model":"databricks-claude-name","messages":[{"role":"user","content":"say hi"}],"max_tokens":20}' \ 
| python3 -m json.tool
```

Expected: JSON with 'choices[0].message.content'.
---

## VS Code Configuration

### `chatLanguageModels.json`
Edit `~/Library/Application Support/Code/User/chatLanguageModels.json` - add the entry from `chatLanguageModels.sample.json`:

```json
{
"name": "dbx", "vendor": "customendpoint", "apiKey": "proxy", "apiType": "chatcompletions", "modelId": [ { "id": "endpoint-name", "name": "display-name", "url": "http://localhost:19000", "toolCalling": true, "vision": true, "maxInputTokens": 128000, "maxOutputTokens": 16000, "temperature":0.1 } ] }
```


> `url` must point to localhost:19000 (the proxy), not Databricks directly. 
> `apiKey` is a dummy value – real auth is handled by the proxy.

### `settings.json` (corporate proxy users) 

Add to ~/Library/Application Support/Code/User/settings.json:

```json
{
"http.proxy": "http://<your-corporate-proxy>:<port>",
"http.proxySupport": "on",
"http.proxyStrictSSL": false,
"http.noProxy": ["localhost", "127.0.0.1", "your-databricks-hostname"]
}
```

- `http.proxy` - needed for VS Code to reach GitHub for Copilot auth through the corporate proxy
- `http.noProxy` - ensures localhost:19000 and the Databricks hostname bypass the corporate proxy

Then reload: Command Palette $\rightarrow$ ++Developer: Reload Window++.

## Service Management

```bash
./dbx-proxy-service.sh start    # start
./dbx-proxy-service.sh stop     # stop
./dbx-proxy-service.sh restart  # restart
./dbx-proxy-service.sh status   # check if running
./dbx-proxy-service.sh log      # tail live log (/tmp/dbx-proxy.log)
```

Optional alias in ~/.zshrc:
```bash
alias dbxproxy='/path/to/vscode_dbx_model/dbx-proxy-service.sh'
```

## GitHub Copilot CLI Integration 

The same local proxy works with **GitHub Copilot CLI** (`copilot`). Copilot CLI supports BYOK (Bring Your Own Key) via environment variables.

### Requirements

• Copilot CLI installed (copilot --version)
• The model must support **tool calling** and **streaming** - Databricks Claude Sonnet 4.6 satisfies both

## How it works 

Copilot CLI reads three env vars to redirect a custom model:

| Variable | Value | Description |
| --- | --- | --- |
| `COPILOT_PROVIDER_TYPE` | `openai` | Use OpenAI-compatible Chat Completions format |
| `COPILOT_PROVIDER_BASE_URL` | `http://localhost:19000` |  Points to the local proxy |
| `COPILOT_PROVIDER_API_KEY` | `proxy` |  Dummy value – real auth handled by the proxy |
| `COPILOT_MODEL` | `databricks-claude-sonnet-4-6` |  Model ID forwarded in the request body|

### Usage 

**Option A – use the included helper script:**



```bash
# Launch Copilot CLI with Databricks model:
./dbx-copilot-cli.sh

# Or source it to set env vars in your current shell, then use copilot normally:
source ./dbx-copilot-cli.sh
copilot chat "explain this error"
copilot run "list all docker containers"
```

**Option B – set env vars manually: **

```bash
export COPILOT_PROVIDER_TYPE=openai
export COPILOT_PROVIDER_BASE_URL=http://localhost:19000 #export COPILOT_PROVIDER_BASE_URL=http://localhost:19000
export COPILOT_PROVIDER_API_KEY=Proxy
export COPILOT_MODEL=databricks-claude-sonnet-4-6
copilot
```

**Option C – permanent alias in "~/.zshrc": **
```bash
alias dbxcopilot="source /path/to/vscode_dbx_model/dbx-copilot-cli.sh && copilot"
```

Then just run `dbxcopilot` from any terminal.

### Notes

• "You'll see a warning Model 'databricks-claude-sonnet-4-6' is not in the built-in catalog — this is expected and harmless. Set `COPILOT_PROVIDER_MAX_PROMPT_TOKENS` and `COPILOT_PROVIDER_MAX_OUTPUT_TOKENS` to suppress it:

```bash
export COPILOT_PROVIDER_MAX_PROMPT_TOKENS=128000
export COPILOT_PROVIDER_MAX_OUTPUT_TOKENS=16000
```

• The proxy must be running before launching Copilot CLI. The ./dbx-copilot-cli.sh script checks this automatically.
• Do NOT use Transfer-Encoding: chunked — the Copilot CLI requires a standard HTTP response with Content-Length.

++ Troubleshooting ++

|Error | Cause | Fix |
| --- | --- | --- |
ECONNREFUSED 127.0.0.1:19000 Proxy not running ./dbx-proxy-service.sh start
401 Unauthorized: API key not sent Wrong or missing PAT Check DBX_TOKEN in ~/.zshrc $\rightarrow$ export PAT
400 Bad Request: temperature and top_p cannot both be specified Proxy not stripping top_p Ensure URL in config is localhost:19000, not Databricks directly
400 Bad Request: "Code starting... Proxy not updating models.json The models.json must be http://localhost:19000
Chat took too long to get ready Copilot can't reach GitHub Set http.proxy in VS Code settings.json
Address already in use Proxy already taken Change port in dbx-proxy-service.py or update ~/.zshrc and chatLanguageModels.json
DBX_TOKEN is not set Token not in env Add export DBX_TOKEN=dapi... to `~/.zshrc
