---
title: MCP Transports
authors:
  - BZ
date: 2025-06-23
categories:
  - AI ENGINEERING
tags:
  - mcp
---

<!-- more -->
# MCP Transports Overview

| **Feature**              | **`stdio`**                              | **`sse`** (Server-Sent Events)                        | **`streamable-http`**                             |
|--------------------------|------------------------------------------|-------------------------------------------------------|---------------------------------------------------|
| **Bidirectional**        | ❌ No                                     | ❌ No                                                  | ✅ Yes                                            |
| **Protocol**             | OS-level `stdin/stdout` over **TCP**     | HTTP (one-way stream over HTTP/1.1)                   | HTTP (usually chunked transfer or HTTP/2/3)       |
| **Data Format**          | Text (line-delimited), JSON              | JSON/Text (EventStream)                              | Arbitrary bytes, JSON, binary, I/O streams        |
| **Latency**              | 🟢 Low (no network)                       | 🟡 Medium (HTTP overhead, server push model)          | 🟢 Low-to-medium (depends on server impl)         |
| **Streaming Support**    | ⚠️ Simulated (by polling stdout)          | ✅ Yes (server pushes tokens as events)               | ✅ Yes (true duplex stream)                       |
| **Transport Overhead**   | 🟢 Minimal                                | 🟡 Medium (HTTP headers, reconnection)                | 🟡 Higher (duplex control, buffers)               |
| **Concurrency**          | ❌ Usually single process/thread          | ✅ Supported with multiple HTTP connections           | ✅ Natively concurrent                             |
| **Error Handling**       | ❌ Basic (process exit codes)             | 🟡 Limited (need to parse events)                     | ✅ Custom error/status codes possible             |
| **Tooling Complexity**   | 🟢 Simple subprocess                     | 🟡 Moderate (SSE client lib required)                 | 🔴 High (custom protocol or server code needed)   |
| **Server Implementation**| CLI app or local executable              | Web server with SSE support                          | Custom backend (e.g., FastAPI with async streams) |
| **Ideal Use Case**       | Local models or CLI tools                | Hosted LLMs with streaming (e.g., OpenAI)             | Agent/toolchain with rich, stateful interaction   |
| **Transport Type**       | IPC/OS process-based                     | Unidirectional HTTP stream                            | Bidirectional over HTTP                           |
| **Compression Support**  | ❌ None                                   | ⚠️ Limited (gzip encoding possible)                   | ✅ Full control over compression                   |
| **Backpressure Handling**| ❌ Minimal (buffer overflow risk)         | ⚠️ Poor (SSE lacks flow control)                      | ✅ Good (can implement windowing/chunking)         |


# Server Code Snippet

```python linenums="1"

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Server1")

# start the server with default stdio transport
mcp.run(transport="stdio")

# start the server with streamable HTTP transport
mcp.run(transport="streamable-http")

# start the server with Server-Sent Events (SSE) transport
mcp.run(transport="sse") # http://127.0.0.1:8000/sse

```

# Client Code Snippet (langchain adapter)

```python linenums="1"
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "server1": {
            "command": "python",
            "args": ["server1.py"],
            "transport": "stdio",
        },
        "server2": {
            "url": "http://localhost:8000/mcp",
            "transprot": "streamable_http"
        },
        "server3":{
            "url": "http://localhost:8000/sse",
            "transport": "sse"
        }
    }
)

```