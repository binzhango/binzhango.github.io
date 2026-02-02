---
title: LangChain Retry Logic
authors:
  - BZ
date: 2025-07-01
categories: 
  - LLM
---

<!-- more -->
# LangChain Invoke Retry Logic
LLM call is not stable and may fail due to network issues or other reasons,
therefore, retry logic is necessary. 

Below is an example of how to implement retry logic in LangChain.

Before implementing the retry logic, you need to install `tenacity` package which provides a flexible retry mechanism.

- [tenacity](https://github.com/jd/tenacity)

## `httpx` Retry Logic

```python linenums="1"
from httpx import ConnectTimeout
from tenacity import retry, stop_after_attempt, retry_if_exception_type

@retry(retry=retry_if_exception_type(ConnectTimeout), stop=stop_after_attempt(3))
async def send_address_match_request(requests_client, payload):
    response = await requests_client.post(
        url=f"<endpoint>",
        data=payload,
    )
    response.raise_for_status()
    resp_data = response.json()
    return resp_data 
```

> Note: The above code snippet is an basic retry logic implementation.
>
> - `stop_after_attempt(3)` : will retry 3 times
> - `retry_if_exception_type(ConnectTimeout)` : only `ConnectTimeout` exception will trigger the retry

## One Advanced Retry Logic
```python linenums="1"
import logging

logger = logging.getLogger(__name__)

@tenacity.retry(
    reraise=True,
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_random(
        min=15, max=45
    ),  # this is mainly included to allow calming RateLimitErrors
    before_sleep=before_sleep_log(logger, logging.INFO),
    after=after_log(logger, logging.INFO),
    retry=(
        tenacity.retry_if_exception_type(APITimeoutError)
        | tenacity.retry_if_exception_type(RateLimitError)
        | tenacity.retry_if_exception_type(InternalServerError)
    ),
)
async def my_async_function():
  pass
```

> Note: The above code snippet is an another retry logic implementation.
>
> - `stop_after_attempt(3)` : will retry 3 times
> - `tenacity.wait_random(min=15, max=45)` : wait for a random time between 15 and 45 seconds before retrying
> - `before_sleep_log(logger, logging.INFO)` : log before retrying
> - `after_log(logger, logging.INFO)` : log after retrying
> - `retry()` : only the specified exceptions will trigger the retry

## LangChain Retry

```python linenums="1"
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain_core.utils import RetryConfig
from tenacity import wait_fixed, stop_after_attempt, wait_random_exponential

model = ChatOpenAI()

# Custom retry configuration
retry_config = RetryConfig(
    stop=stop_after_attempt(3),  # Retry up to 3 times
    wait=wait_fixed(2),          # Wait 2 seconds between attempts
)

# Or a more complex configuration with exponential backoff
retry_config = RetryConfig(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(min=1, max=10),  # Waits 1–10s, increasing randomly
)


# Apply retry config
model_with_retry = model.with_retry(retry_config=retry_config)

# OR: You can now use this in your chain
template = PromptTemplate.from_template("Tell me a joke about {topic}.")
chain = template | model.with_retry(retry_config=retry_config)
result = chain.invoke({"topic": "AI"})

```