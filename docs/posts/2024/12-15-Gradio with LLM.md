---
title: Gradio with Ollama
authors:
  - BZ
date: 2024-12-15
categories: 
  - python
---

# Simple Unstructured file processing
We have a lot of pdf files that contain import information, however, the information
are unstructured (text, table, image, etc...).
To extract and utilize them in our downstream job, an open source [unstructured](https://github.com/Unstructured-IO/unstructured) is helpful to implement what we want

![image](https://mintlify.s3.us-west-1.amazonaws.com/unstructured-53/img/open-source/strategy.png)
<!-- more -->

# Demo App

```mermaid

%%{init: { 'look':'handDrawn' } }%%

flowchart LR
  A[Gradio UI] --> B(Ollama Server)
  B --> C["
  #bull; Gemma2
  #bull; Llama3
  #bull; Phi3
  #bull; Mistral
  "]
  style C color:#FFFFFF,text-align:left,fill:#D2691E
  style B fill:#FFE4C4

```