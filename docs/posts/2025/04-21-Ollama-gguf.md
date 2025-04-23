---
title: Ollama Import GGUF Models
authors:
  - BZ
date: 2025-04-21
categories: 
  - LLM
---

<!-- more -->

## Ollama Models
If you're looking to experiment with various models easily, importing GGUF might be your go-to method. 
Here's how it works: 

- You start by creating a `Modelfile`, which acts as a key to unlock any GGUF model you want to use.

```shell linenums="1" title="Modelfile"
FROM /agentica-org_DeepCoder-14B-Preview-Q8_0.gguf
FROM /agentica-org_DeepCoder-14B-Preview.imatrix

TEMPLATE """
{{- if .System }}{{ .System }}{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1}}
{{- if eq .Role "user" }}<｜User｜>{{ .Content }}
{{- else if eq .Role "assistant" }}<｜Assistant｜>{{ .Content }}{{- if not $last }}<｜end▁of▁sentence｜>{{- end }}
{{- end }}
{{- if and $last (ne .Role "assistant") }}<｜Assistant｜>{{- end }}
{{- end }}
"""
```

- Once that file is ready, all you need to do is run a simple command, and voilà—your model is imported and ready for action!

```shell
ollama create deepcoder -f ./Modelfile

# gathering model components 
# copying file sha256:8add5abcbf3c0413496f039275119f6d74555a16e410c32ada75d69815d904cb 100% 
# parsing GGUF 
# using existing layer sha256:8add5abcbf3c0413496f039275119f6d74555a16e410c32ada75d69815d904cb 
# creating new layer sha256:6b62346ff5a355054f0cf583cb76c4cb0841c729114aae34d40e75c7170a8c59 
# writing manifest 
# success 

```




