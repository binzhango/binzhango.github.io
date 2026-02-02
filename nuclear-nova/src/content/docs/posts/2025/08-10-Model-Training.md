---
title: Training LLM From Zero
authors:
  - BZ
date: 2025-08-10
categories: 
  - LLM
---

1. [Objective](#objective)
2. [Environment Setup](#environment-setup)


<!-- more -->
# Objective

The goal of this project is to design, implement, and train a small-scale Large Language Model (LLM) from scratch, 
progressing through the full training lifecycle:

1.  Pre-training on large-scale unlabeled text.
2.	Supervised Fine-Tuning (SFT) on high-quality instruction-following datasets.
3.	Parameter-Efficient Fine-Tuning (LoRA) for resource-efficient adaptation.
4.	Direct Preference Optimization (DPO) for aligning the model with human preferences.

The project aims to serve as a practical, hands-on implementation of LLM training concepts from recent research.


# Environment Setup
- `macOS` with `M Series` chip

    {==‼️ MPS is not optimized for training ==}

```bash
    Testing on: macOS MPS device (M4, 64GB RAM)
    PyTorch version: 2.3.0
    MPS available: True
    
    Matrix 1024x1024: 10.40 TFLOPS | Time: 20.65ms
    Matrix 2048x2048: 13.45 TFLOPS | Time: 127.76ms
    Matrix 4096x4096: 13.49 TFLOPS | Time: 1018.53ms
    Matrix 8192x8192: 12.82 TFLOPS | Time: 8573.45ms
    Matrix 16384x16384: 9.37 TFLOPS | Time: 93871.68ms
```

- `windows` with `CUDA` (**recommended**)

```bash
    Testing on:CUDA device (2080Ti, Memory 11G)
    PyTorch version: 2.8.0+cu129
    CUDA available: True
    
    Matrix 1024x1024: 65.62 TFLOPS | Time: 3.27ms
    Matrix 2048x2048: 634.46 TFLOPS | Time: 2.71ms
    Matrix 4096x4096: 4447.00 TFLOPS | Time: 3.09ms
    Matrix 8192x8192: 34163.30 TFLOPS | Time: 3.22ms
    Matrix 16384x16384: 199933.93 TFLOPS | Time: 4.40ms
```

- **python package** for `CUDA` support
    - `torch` [PyTorch](https://pytorch.org/get-started/locally/) ‼️{== Be careful: CUDA version must match the PyTorch (12.6, 12.8, 12.9) ==}

         `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129`

    - `transformers` [Install](https://huggingface.co/docs/transformers/en/installation)

        - `pip install transformers`
        - or `pip3 install --no-build-isolation transformer_engine[pytorch]` [CUDA Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html)

    - `peft`

        - `pip install peft`

- **cuda toolkit**
    - choose the right version for your system [link](https://developer.nvidia.com/cuda-toolkit-archive)
    - install `cudnn` from [link](https://developer.nvidia.com/cudnn)
    - check your Nvidia driver and cuda version: `nvidia-smi`


# Dataset
- tokenizer dataset
- pre-training dataset
- sft (Supervised Fine-Tuning) dataset
- dpo (Direct Preference Optimization) dataset