---
title: How LLM Tools work
authors:
  - BZ
date: 2025-07-11
categories: 
  - LLM
---

<!-- more -->
# Tools in Large Language Models (LLMs)
> Tools enable large language models (LLMs) to interact with external systems, APIs, or data sources,
>
> extending their capabilities beyond text generation.

**Two aspects of tools are crucial:**

1. How to create tools
2. How LLM finds and uses these tools

## Create Tool
{==Tool system is a form of **metaprogramming**==}

Tools are defined with metadata, including

- <span style="color: green">**Name**</span>: A unique identifier (e.g., get_current_weather).
- <span style="color: green">**Description**</span>: A natural language explanation of what the tool does (e.g., "Retrieve the current weather for a given city").
- <span style="color: green">**Schema**</span>: A JSON schema or similar structure specifying the input parameters (e.g., {"city": {"type": "string"}}).

```python
# langchain tool
@tool
def get_weather(city: str) -> str:
    """Return current weather in a city."""
    ...
```
>	name = "get_weather"
>
>	description = "Return current weather in a city."
>
>	args = {"city": str}
>
>  LangChain reads the metadata (function name, docstring, type hints)
>
>  :exclamation: **Note**: ==More comprehensive descriptions and schemas help LLMs understand and use tools effectively.==

## Tool Detection

- How LLMs Detect the Required Tool?
    - ==Query Parsing==:
        - The LLM analyzes the user’s query using its natural language processing capabilities.
        - It matches the query’s intent and content to the tool descriptions or keywords. For example, a query like “What’s the weather in New York?” aligns with a tool described as “Retrieve the current weather.”
        - Modern LLMs, especially those fine-tuned for tool calling (e.g., OpenAI’s GPT-4o), use semantic understanding to infer intent rather than relying solely on keywords.
    - ==Tool Selection==:
       - **Prompt-Based (LangChain)**: The LLM is given a prompt that includes tool descriptions and instructions to select the appropriate tool. The LLM reasons about the query (often using a framework like ReAct) and outputs a decision to call a specific tool with arguments.
       - **Fine-Tuned Tool Calling (OpenAI)**: The LLM is trained to output a structured JSON object specifying the tool name and arguments directly, based on the query and tool schemas provided in the API call.


# Mock Tool Implementation

- [x] Step 1: Define a Tool Function
```python
def add_numbers(x: int, y: int) -> int:
    """Add two numbers and return the result."""
    return x + y
```
- [x] Step 2: Use inspect to Introspect
```python
import inspect

sig = inspect.signature(add_numbers)

# Print parameter names and types
for name, param in sig.parameters.items():
    print(f"{name}: {param.annotation} (default={param.default})")

# Print return type
print(f"Returns: {sig.return_annotation}")
```
- [x] Step 3: Dynamically Call the Function
```python
# Assume this comes from LLM tool calling output
llm_output = {
    "x": 5,
    "y": 7
}

# Dynamically call it
result = add_numbers(**llm_output)
print(result)  # ➜ 12
```

## Summary

1. Uses `inspect.signature`(func) to introspect argument names and types.
2. Formats this into metadata for LLM prompt.
3. Parses LLM output ({tool_name, tool_args}).
4. Validates the arguments.
5. Calls the function like: tool.func(**tool_args).