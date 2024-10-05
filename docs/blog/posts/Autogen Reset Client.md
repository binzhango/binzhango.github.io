---
title: AutoGen HttpClient
authors:
  - BZ
date: 2024-09-08 00:00:00
categories: 
  - LLM
  - AutoGen
pin: true
---

## HttpClient

<!-- more -->

```python linenums="1" title="my_client.py"

import httpx

class MyHttpClient(httpx.Client):
    def __deepcopy__(self, dummy):
        return self
    
```

## AutoGen LLM Config

Now we can add our proxy config in our client, which will resolve connection issue.
```json
"http_client": MyHttpClient(verify=False, proxy="http://<>")
```