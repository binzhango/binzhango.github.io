---
title: LLM Interview Questions
authors:
  - BZ
date: 2025-10-30
categories: 
  - LLM
---

# LLM Questions
<!-- more -->
## Machine Learning Fundamentals


??? question "Explain bias-variance tradeoff. How does it manifest in LLMs?"

    - <span class="def-mono-red">Bias</span>: Error from incorrect assumption in the model
        - High bias leads to underfitting, where the model fails to capture patterns in the training data

    - <span class="def-mono-red">Variance</span>: Error from sensitivity to small fluctuations in the training data
        - High variance leads to overfitting, where the model memorizes noise instead of learning generalization patterns


    The bias-variance tradeoff in ML describe the tension between **ability to fit training data** and **ability to generalize the new data**

    <span class="def-mono-blue">bias-variance in LLM</span>:

    - <span class="def-mono-gold">Model Parameters: Capacity vs. Overfitting</span>
        - **Too few parameters**: A model with insufficient (e.g. small transformer) cannot capture complex patterns in the data, leading to high bias.
        > A small LLM might fail to understand language or generate coherent long texts.
        - **Too many parameters**: A model with excessive capacity risks overfitting to training data, memorizing noise and details instead of learning generalizable patterns
        > A large LLM fine-tuned on a small dataset may generate text that is statistically similar to the training data but lack coherence and factual accuracy. (e.g. <span class="def-mono-red">hallucinations</span>)
        - **Balancing Act**:
        > More parameters reduce bias by enabling the model to capture complex patterns but increase variance if not regularized.
        > Regularization techniques: (e.g dropout, weight decay) help mitigate overfitting in high-parameter models

    - <span class="def-mono-gold">Training Epochs: Learning Duration vs. Overfitting</span>
        - **Too few epochs**: The model hasn't learned enough from the data, leading to high bias.
        > A transformer trained for only 1 epoch may fail to capture meaningful relationships in the text.
        - **Too many epochs**: The model starts memorizing training data, increasing variance. This is common in transformer with high capacity and small datasets
        > A transformer fine-tuned on a medical dataset for 100 epochs may overfit to rare cases, leading to poor generalization.
        - **Tradeoff in Transformers**
        > Training loss decreases with epochs (low bias), but validation loss eventually increase (high variance).
        >
        > Early stopping is critical for transformers to avoid overfitting, especially when training on small or noisy datasets.
    - <span class="def-mono-gold">Noise vs Representativeness </span>
        - **Low-quality data**: Noisy, biased, or incomplete data prevents the model from learning accurate patterns, increasing biase.
        > A transformer trained on a dataset with limited examples of rare diseases may fail to diagnose them accurately
        - **Noisy/unrepresentative data**: The model learns inconsistent patterns, increasing variance.
        > A dataset with duplicate or corrupted text may cause the model to overfit. A transformer trained on a dataset with biased political content 
        > may generate polarized outputs.
        > Data augmentation (e.g. paraphrasing, back-translation) increases diversity, mitigating overfitting
    
??? question "What is the difference between L1 and L2 regularization? When would you use elastic net in an LLM fine-tune?"

    <span class="def-mono-gold">Regularization</span> adds a penalty term to the loss function so that the optimizer favours simpler or smoother solutions.
    In practice it is usually added to a model‑level loss (cross‑entropy, MSE, …) as a separate scalar that scales with the weights.

    \[  \text{Loss}_{\text{regularized}} = \text{Loss}_{\text{original}} + \lambda \cdot \text{Penalty}(w) \]

    |Feature | L1 (Lasso) | L2 (Ridge)|
    |-:|:-:|:-:|
    |Weight Behavior| Many → 0 (sparse)|"All → small, non-zero"|
    |Feature Selection| Yes| No|
    |Solution|Not always unique|Always unique|
    |Robust to Outliers|Less|More|

    <span class="def-mono-red"> Key Insight:</span>

      - <span class="def-mono-blue">L1 regularization</span> is more robust to outliers in the **data (target outliers)**
      - <span class="def-mono-blue">L2 regularization</span> is more robust to outliers in the **features (collinearity)**

    <span class="def-mono-red">L1/L2 in LLM</span>:

      - Use L2 by default. Use L1 if you want sparse, interpretable updates.
      - L2 keeps updates smooth. L1 keeps updates minimal — and that’s often better for deployment.
      - Use L2 to win benchmarks. Use L1 to ship to users.
      > 1. Sparse LoRA = Tiny Adapters
      > 2. Faster Inference (Real Speedup!)
      > 3. Better Generalization (Less Overfitting)
      > 4. Interpretable Fine-Tuning
      > 5. Clean Model Merging

      ```sh
        ┌──────────────────────┐
        │ Fine-tuning an LLM?  │
        └────────┬─────────────┘
                │
                ▼
        ┌──────────────────────┐     YES → Use L2 (weight_decay=0.01)
        │ Large, clean data?   │───NO──►┐
        └────────┬─────────────┘        │
                │                    │
                ▼                    ▼
        ┌──────────────────────┐ ┌──────────────────────────┐
        │ Need max accuracy?   │ │ Want small/fast model?   │
        └────────┬─────────────┘ └────────────┬─────────────┘
                │                          │
                YES                        YES
                │                          │
                ▼                          ▼
            Use L2                     Use L1 (+ pruning)
      ```





??? question "Prove that dropout is equivalent to an ensemble during inference (hint: geometric distribution)."
    - Where dropout appears in a Transformer
        - Attention dropout
        - Feedforward dropout
        - Residual dropout
    - The ensemble view of dropout in a Transformer
        - Each layer (and even each neuron) may be dropped independently.
        - A particular dropout mask $m = (m^{(1)}, m^{(2)}, \dots, m^{(L)})$ defines one specific subnetwork (one “member” of the ensemble).

    **During training:** Randomly turn off some neurons (like flipping a coin for each one). This forces the network to learn many different "sub-networks" — each time you train, a different combination of neurons is active.

    **During testing (inference):** Instead of picking one sub-network, we use all neurons, but scale down their strength (usually by half if dropout rate is 50%). This is the "mean network."

    <span class="def-mono-gold">Why this is like an ensemble:</span>
    Imagine you could run the model 1,000 times (or $2^{(N)}$ times for N neurons), each time with a different random set of neurons turned off, and then average all their predictions. **That would be a huge ensemble of sub-networks** — very accurate, but way too slow.
    Dropout’s trick: Using the scaled "mean network" at test time gives exactly the same prediction as if you had averaged the geometric mean of all those possible sub-networks.

    <span class="def-mono-blue">Dropout = training lots of sub-networks, inference = using their collective average — fast and smart.</span>



??? question "What is the curse of dimensionality? How do positional encodings mitigate it in Transformers?"
    <span class="def-mono-blue">Higher dimensions → sparser data → harder to learn meaningful relationships.</span>

    The curse of dimensionality refers to the set of problems that arise when data or model representations exist in high-dimensional spaces.

    - **Data sparsity:** Points become exponentially sparse — distances between points tend to concentrate, making similarity less meaningful.
    - **Combinatorial explosion:** The volume of the space grows exponentially $O(k^{(d)})$, so covering it requires exponentially more data.
    - **Poor generalization:** Models struggle to learn smooth mappings because there’s too little data to constrain the high-dimensional space.
    
    
    **Token in Transformers**
    Transformers process tokens as vectors in a **high-dimensional** embedding space (e.g., 768 or 4096 dimensions).
    However — *self-attention* treats each token as a set element rather than a sequence element. The attention mechanism itself has no built-in sense of order.
    *The model only knows “content similarity,” not which token came first or last.*

    Without order, the model would need to **learn positional relationships implicitly** across high-dimensional embeddings.
    That’s hard — and it exacerbates the curse of dimensionality because:

	 - There’s no geometric bias for position.
	 - Each token embedding can drift freely in a massive space.
	 - The model must infer ordering purely from statistical co-occurrence — *requiring more data and more parameters.*

     **How Positional Encodings Help**

     **Positional encodings (PEs)** inject structured, low-dimensional information about sequence order directly into the embeddings.
        - Adds a geometric bias to embeddings — nearby positions have nearby encodings.
        - Reduces the effective search space — positions are no longer independent random vectors.
        - Enables extrapolation: the sinusoidal pattern generalizes beyond training positions.
        - The model can compute relative positions via linear operations (e.g., dot products of PEs reflect distance).



??? question "Explain maximum likelihood estimation for language modeling."
    Training a neural LM (like a Transformer) by minimizing the negative log-likelihood (NLL) is the same as maximizing the likelihood:

    $$\boxed{
    \text{Maximizing likelihood} 
    \;\; \Leftrightarrow \;\; 
    \text{Maximizing log-likelihood} 
    \;\; \Leftrightarrow \;\; 
    \text{Minimizing negative log-likelihood}
    }$$

    **Example**
    > Sentence: "The cat sat on the mat."
    >
    > The MLE objective trains the model to maximize:
    > $P(\text{The}) \cdot P(\text{cat}|\text{The}) \cdot P(\text{sat}|\text{The cat}) \cdot P(\text{on}|\text{The cat sat}) \cdot \dots$




??? question "What is negative log-likelihood? Write the per-token loss for GPT."

    $$\boxed{
        \ell(\theta) = \sum_{t=1}^T \log P(x_t \mid x_{<t}; \theta)
    }
    $$

    $$\boxed{
        \text{NLL}(\theta) = -\ell(\theta)
    }
    $$

    $$\boxed{
        \text{NLL}(\theta) = -\sum_{t=1}^T \log P(x_t \mid x_{<t}; \theta)
    }
    $$

    **where** $x_{<t}$ means **All tokens** before $t$: $x_1, \dots, x_{t-1}$
    <!-- $$\boxed{
    x_{<t} = \text{the past context used to predict } x_t
    }$$ -->

    This is the heart of autoregressive language modeling — like GPT!


??? question "Compare cross-entropy, perplexity, and BLEU. When is perplexity misleading?"

    1. **Cross-Entropy:** Cross-entropy measures how well a probabilistic model predicts a target distribution — in LM, how well the model assigns high probability to the correct next tokens.
    2. **Perplexity:** Perplexity (PPL) is simply the exponentiation of the cross-entropy
    3. **BLEU (Bilingual Evaluation Understudy):** BLEU is an n-gram overlap metric for evaluating machine translation or text generation quality against reference texts
    
    <span class="def-mono-blue">Perplexity is rephrasing cross-entropy in a more intuitive, more human-readable.</span>
    
    > **Perplexity = "How predictable is the language?"**
    >
    > **BLEU = "How much does the output match a reference?"**
    >
    > Example:
    >
    > Reference: "The cat is on the mat."
    >
    > Model output: "The dog is on the mat."
    >
    > → Low perplexity (grammatical, fluent)
    >
    > → Low BLEU (wrong content)
    > **BLEU is non-probabilistic and reference-based — unlike cross-entropy and perplexity.**


    ⚠️ <span class="def-mono-red">When Perplexity Is Misleading???</span>

    **Perplexity only measures how well the model predicts tokens probabilistically — not how meaningful or correct the generated text is.**

    - Different tokenizations or vocabularies
        - A model with smaller tokens or subwords might have lower perplexity just because predictions are more granular, not actually better linguistically.
    - Domain mismatch
        - A model trained on Wikipedia might have low perplexity on Wikipedia text but produce incoherent answers to questions — it knows probabilities, not task structure.
    - Human-aligned vs statistical objectives
        - A model can assign high likelihood to grammatical but dull continuations (e.g., “The cat sat on the mat”) while rejecting creative or rare but correct continuations — good perplexity, poor real-world usefulness.
    - Non-autoregressive or non-likelihood models
        - For encoder-decoder or retrieval-augmented systems, perplexity may not correlate with generation quality because these models are not optimized purely for next-token prediction.
    - Overfitting
        - A model with very low perplexity on training data may memorize text, but generalize poorly (BLEU or human eval drops).


??? question "Why is label smoothing used in LLMs? Derive its modified loss?"
    **Label smoothing is used in LLMs to prevent overconfidence and improve generalization.**

    Instead of training on a **one-hot** target (where the correct token has probability 1 and all others 0), a small portion ε of that probability is spread across all other tokens.

    So the true token gets (1 − ε) probability, and the rest share ε uniformly.

    This changes the loss from the usual “−log p(correct token)” to a mix of:

	 - `(1 − ε) × loss` for the correct token, and
	 - `ε × average loss` over all tokens.

??? question "What is the difference between hard and soft attention?"
    - Hard attention → discrete, selective, non-differentiable.
    - Soft attention → continuous, weighted, differentiable.


## Fundamentals of Large Language Models (LLMs)

??? "Question Bank: TBD"

    I. Fundamentals of Large Language Models (LLMs)

    1. LLM Basics

        •	What are the main open-source LLM families currently available?
        •	What’s the difference between prefix decoder, causal decoder, and encoder-decoder architectures?
        •	What is the training objective of large language models?
        •	What causes the emergent abilities of LLMs?
        •	Why are most modern LLMs decoder-only architectures?
        •	Give a simple introduction to large language models (LLMs).
        •	What do the numbers like 175B, 60B, or 540B mean in LLM names?
        •	What are the advantages of LLMs?
        •	What are the disadvantages of LLMs?
        •	Explain the difference between encoder-only, decoder-only, and encoder-decoder models.
        •	Compare the major LLMs such as BART, LLaMA, GPT, T5, and PaLM.
        •	What’s the difference between prefix LM and causal LM?

    ⸻

    2. Layer Normalization Variants

    (a) Formulas and Concepts

        •	What’s the computation formula for LayerNorm?
        •	What’s the computation formula for RMSNorm?
        •	What are the main characteristics of RMSNorm compared to LayerNorm?
        •	What’s the core idea of DeepNorm?
        •	Show the basic code implementation of DeepNorm.
        •	What are the advantages of DeepNorm?

    (b) Position in Model

        •	What are the differences when applying LayerNorm at different positions in LLMs?

    (c) Comparison
        •	Which normalization method is used in different LLM architectures?

    ⸻

    3. Activation Functions in LLMs
    
        •	What’s the formula for the FFN (Feed-Forward Network) block?
        •	What’s the GeLU formula?
        •	What’s the Swish formula?
        •	What’s the formula of an FFN block with GLU (Gated Linear Unit)?
        •	What’s the formula of a GLU block using GeLU?
        •	What’s the formula of a GLU block using Swish?
        •	Which activation functions do popular LLMs use?
        •	What are the differences between Adam and SGD optimizers?

    ⸻

    4. Attention Mechanisms — Advanced Topics

    (a) Attention Optimization and Variants
        •	What are the problems with traditional attention?
        •	What are the directions of improvement for attention?
        •	What are the attention variants?

    (b) Multi-Query and Grouped-Query Attention
        •	What issues exist in multi-head attention?
        •	Explain Multi-Query Attention (MQA).
        •	Compare Multi-head, Multi-Query, and Grouped-Query Attention.
        •	What are the benefits of MQA?
        •	Which models use MQA or GQA?

    (c) FlashAttention
        •	Why was FlashAttention introduced?
        •	Briefly explain its core idea.
        •	What are its advantages?
        •	Which models implement FlashAttention?

    (d) Other Improvements
        •	What is parallel transformer block?
        •	What’s the computational complexity of attention and how can it be improved?
        •	What is Paged Attention?
        •	Compare MHA, GQA, and MQA — what are their key differences?

    ⸻

    5. Cross-Attention
        •	Why do we need Cross-Attention?
        •	Explain Cross-Attention.
        •	Compare Cross-Attention and Self-Attention — similarities and differences.
        •	Compare Cross-Attention and Multi-Head Attention.
        •	Provide a code implementation of Cross-Attention.
        •	What are its application scenarios?
        •	What are the advantages and challenges of Cross-Attention?

    ⸻

    6. Transformer Operations
        •	How to load a BERT model using transformers?
        •	How to output a specific hidden_state from BERT using transformers?
        •	How to get the final or intermediate layer vector outputs of BERT?

    ⸻

    7. LLM Loss Functions
        •	What is KL divergence?
        •	Write the cross-entropy loss and explain its meaning.
        •	What’s the difference between KL divergence and cross-entropy?
        •	How to handle large loss differences in multi-task learning?
        •	Why is cross-entropy preferred over MSE for classification tasks?
        •	What is information gain?
        •	How to compute softmax and cross-entropy loss (and binary cross-entropy)?
        •	What if the exponential term in softmax overflows the float limit?

    ⸻

    8. Similarity & Contrastive Learning
        •	Besides cosine similarity, what other similarity metrics exist?
        •	What is contrastive learning?
        •	How important are negative samples in contrastive learning, and how to handle costly negative sampling?

    ⸻

    II. Advanced Topics in LLMs
        •	What is a generative large model?
        •	How do LLMs make generated text diverse and non-repetitive?
        •	What is the repetition problem (LLM echo problem)?
        •	Why does it happen?
        •	How can it be mitigated?
        •	Can LLaMA handle infinitely long inputs?
        •	When should you use BERT vs. LLaMA / ChatGLM models?
        •	Do different domains require their own domain-specific LLMs?
        •	How to enable an LLM to process longer texts?

    ⸻

    III. Fine-Tuning Large Models

    1. General Fine-Tuning
        •	Why does the loss drop suddenly in the second epoch during SFT?
        •	How much VRAM is needed for full fine-tuning?
        •	Why do models seem dumber after SFT?
        •	How to construct instruction fine-tuning datasets?
        •	How to improve prompt representativeness?
        •	How to increase prompt data volume?
        •	How to select domain data for continued pretraining?
        •	How to prevent forgetting general abilities after domain tuning?
        •	How to make the model learn more knowledge during pretraining?
        •	When performing SFT, should the base model be Chat or Base?
        •	What’s the input/output format for domain fine-tuning?
        •	How to build a domain evaluation set?
        •	Is vocabulary expansion necessary?
        •	How to train your own LLM?
        •	Experience in training Chinese LLMs?
        •	What are the benefits of instruction fine-tuning?
        •	During which stage — pretraining or fine-tuning — is knowledge injected?

    ⸻

    2. SFT Tricks
        •	What’s the typical SFT workflow?
        •	What are key aspects of training data?
        •	How to choose between large and small models?
        •	How to ensure multi-task training balance?
        •	Can SFT learn knowledge at all?
        •	How to select datasets effectively?

    ⸻

    3. Training Experience
        •	How to choose a distributed training framework?
        •	What are key LLM training tips?
        •	How to choose model size?
        •	How to select GPU accelerators?

    ⸻

    IV. LangChain and Agent-Based Systems

    1. LangChain Core
        •	What is LangChain?
        •	What are its core concepts?
        •	Components and Chains
        •	Prompt Templates and Values
        •	Example Selectors
        •	Output Parsers
        •	Indexes and Retrievers
        •	Chat Message History
        •	Agents and Toolkits

    ⸻

    2. Long-Term Memory in Multi-Turn Conversations
        •	How can Agents access conversation context?
        •	Retrieve full history
        •	Use sliding window for recent context
        •	Other strategies

    ⸻

    3. Practical RAG Q&A using LangChain
        •	(Practical implementation questions about RAG apps in LangChain)

    ⸻

    V. Retrieval-Augmented Generation (RAG)

    1. RAG Basics
        •	Why do LLMs need an external (vector) knowledge base?
        •	What’s the overall workflow of LLM+VectorDB document chat?
        •	What are the core technologies?
        •	How to build an effective prompt template?

    ⸻

    2. RAG Concepts
        •	What are the limitations of base LLMs that RAG solves?
        •	What is RAG?
        •	Retrieval module
        •	How to obtain accurate semantic representations?
        •	How to align query/document semantic spaces?
        •	How to match retrieval model output with LLM preferences?
        •	Generation module
        •	How to improve results via post-retrieval processing?
        •	How to optimize generator adaptation to inputs?
        •	What are the benefits of using RAG?

    ⸻

    3. RAG Layout Analysis

    (a) PDF Parsing
        •	Why is PDF parsing necessary?
        •	What are common methods and their differences?
        •	What problems exist in PDF parsing?

    (b) Table Recognition
        •	Why is table recognition important?
        •	What are the main methods?
        •	Traditional methods
        •	pdfplumber extraction techniques

    (c) Text Chunking
        •	Why do we need text chunking?
        •	What are common chunking strategies (regex, Spacy, LangChain, etc.)?

    ⸻

    4. RAG Retrieval Strategies
        •	Why use LLMs to assist recall?
        •	HYDE approach: idea and issues
        •	FLARE approach: idea and recall strategies
        •	Why construct hard negative samples?
        •	Random sampling vs. Top-K hard negative sampling

    ⸻

    5. RAG Evaluation
        •	Why evaluate RAG?
        •	What are the evaluation methods, metrics, and frameworks?

    ⸻

    6. RAG Optimization
        •	What are the optimization strategies for retrieval and generation modules?
        •	How to enhance context using knowledge graphs (KGs)?
        •	What are the problems with vector-based context augmentation?
        •	How can KG-based methods improve it?
        •	What are the main pain points in RAG and their solutions?
        •	Content missing
        •	Top-ranked docs missed
        •	Context loss
        •	Failure to extract answers
        •	Explain RAG-Fusion:
        •	Why it’s needed,
        •	Core technologies,
        •	Workflow, and
        •	Advantages.

    ⸻

    7. Graph RAG
        •	Why do we need Graph RAG?
        •	What is Graph RAG and how does it work?
        •	Show a code example and use case.
        •	How to improve ranking optimization in Graph RAG?

    ⸻

    VI. Parameter-Efficient Fine-Tuning (PEFT)

    1. PEFT Fundamentals
        •	What is fine-tuning, and how is it performed?
        •	Why do we need PEFT?
        •	What is PEFT and its advantages?

    ⸻

    2. Adapter Tuning
        •	Why use adapter-tuning?
        •	What’s the core idea behind adapter-tuning?
        •	How does it differ from full fine-tuning?

