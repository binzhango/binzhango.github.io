---
title: LLM Interview Questions
authors:
  - BZ
date: 2025-10-30
categories:
  - LARGE LANGUAGE MODELS
tags:
  - llm fundamentals
---

# Questions
<!-- more -->

## Machine Learning

<details>
<summary>Machine Learning Concepts</summary>

::::tip
:::note[Question: How would you describe the concept of machine learning in your own words?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
Machine learning focuses on creating systems that improve their performance on a task by learning patterns from data rather than relying on explicit programming.
:::

</details>

:::note[Question: Can you give a few examples of real-world areas where machine learning is particularly effective?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
Machine learning is especially valuable for solving complex problems without clear rule-based solutions, automating decision-making instead of hand-crafted logic, adapting to changing environments, and extracting insights from large datasets.
:::

</details>

:::note[Question: What are some typical problems addressed with unsupervised learning methods?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
Typical unsupervised learning tasks include clustering, data visualization, dimensionality reduction, and association rule mining.
:::

</details>

:::note[Question: Would detecting spam emails be treated as a supervised or unsupervised learning problem, and why?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
Spam filtering is an example of a supervised learning problem because the model learns from examples of emails labeled as "spam" or "not spam".
:::

</details>

:::note[Question: What does the term ‘out-of-core learning’ refer to in machine learning?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
Out-of-core learning enables training on datasets too large to fit in memory by processing them in smaller chunks (mini-batches) and updating the model incrementally.
:::

</details>

:::note[Question: How can you distinguish between model parameters and hyperparameters?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
- **Model parameters** define how the model behaves and are learned during training (e.g., weights in linear regression).

- **Hyperparameters** are external settings chosen before training, such as the learning rate or regularization strength.
:::

</details>

:::note[Question: What are some major difficulties or limitations commonly faced when building machine learning systems?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
	Key challenges in machine learning include 

    - insufficient or low-quality data
    - poor feature selection
    - non-representative samples
    - models that either underfit (too simple) or overfit (too complex)
:::

</details>

:::note[Question: If a model performs well on training data but poorly on unseen data, what issue is occurring, and how might you address it?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
When a model performs well on training data but poorly on unseen examples, it’s overfitting. This can be mitigated by collecting more diverse data, simplifying the model, applying regularization, or cleaning noisy data.
:::

</details>

:::note[Question: What is a test dataset used for, and why is it essential in evaluating a model’s performance?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
A test set provides an unbiased estimate of how well a model will perform on new, real-world data before deployment.
:::

</details>

:::note[Question: What role does a validation set play during the model development process?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
A validation set helps compare multiple models and tune hyperparameters, ensuring better generalization to unseen data.
:::

</details>

:::note[Question: What is a train-dev dataset, in what situations would you create one, and how is it applied during model evaluation?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
The train-dev set is a small portion of the training data set aside to identify mismatches between the training distribution and the validation/test distributions. You use it when you suspect that your production data may differ from your training data. The model is trained on most of the training data and evaluated on the train-dev set to detect overfitting or data mismatch before comparing results on the validation set.
:::

</details>

:::note[Question: Why is it problematic to adjust hyperparameters based on test set performance?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
If you tune hyperparameters using the test set, you risk overfitting to that specific test data, making your performance results misleadingly high. As a result, the model might perform worse in real-world scenarios because the test set is no longer an unbiased measure of generalization.
:::

</details>

::::

</details>

## LLM Fundamentals


:::note[Question: Explain bias-variance tradeoff. How does it manifest in LLMs?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
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
:::

</details>

:::note[Question: What is the difference between L1 and L2 regularization? When would you use elastic net in an LLM fine-tune?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
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
:::

</details>

:::note[Question: Prove that dropout is equivalent to an ensemble during inference (hint: geometric distribution).]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
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
:::

</details>

:::note[Question: What is the curse of dimensionality? How do positional encodings mitigate it in Transformers?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
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
:::

</details>

:::note[Question: Explain maximum likelihood estimation for language modeling.]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
Training a neural LM (like a Transformer) by minimizing the negative log-likelihood (NLL) is the same as maximizing the likelihood:

$$
\text{Maximizing likelihood} 
\;\; \Leftrightarrow \;\; 
\text{Maximizing log-likelihood} 
\;\; \Leftrightarrow \;\; 
\text{Minimizing negative log-likelihood}
$$

**Example**
> Sentence: "The cat sat on the mat."
>
> The MLE objective trains the model to maximize:
> $P(\text{The}) \cdot P(\text{cat}|\text{The}) \cdot P(\text{sat}|\text{The cat}) \cdot P(\text{on}|\text{The cat sat}) \cdot \dots$
:::

</details>

:::note[Question: What is negative log-likelihood? Write the per-token loss for GPT.]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
$$
\ell(\theta) = \sum_{t=1}^T \log P(x_t \mid x_{<t}; \theta)
$$

$$
\text{NLL}(\theta) = -\ell(\theta)
$$

$$
\text{NLL}(\theta) = -\sum_{t=1}^T \log P(x_t \mid x_{<t}; \theta)
$$

**where** $x_{<t}$ means **All tokens** before $t$: $x_1, \dots, x_{t-1}$
<!-- $$\boxed{
x_{<t} = \text{the past context used to predict } x_t
}$$ -->

This is the heart of autoregressive language modeling — like GPT!
:::

</details>

:::note[Question: Compare cross-entropy, perplexity, and BLEU. When is perplexity misleading?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
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
:::

</details>

:::note[Question: Why is label smoothing used in LLMs? Derive its modified loss?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
**Label smoothing is used in LLMs to prevent overconfidence and improve generalization.**

Instead of training on a **one-hot** target (where the correct token has probability 1 and all others 0), a small portion ε of that probability is spread across all other tokens.

So the true token gets (1 − ε) probability, and the rest share ε uniformly.

This changes the loss from the usual “−log p(correct token)” to a mix of:

 - `(1 − ε) × loss` for the correct token, and
 - `ε × average loss` over all tokens.
:::

</details>

:::note[Question: What is the difference between hard and soft attention?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
- Hard attention → discrete, selective, non-differentiable.
- Soft attention → continuous, weighted, differentiable.
:::

</details>

## Fundamentals of Large Language Models (LLMs)

::::note[Question Bank]
- <span class="def-mono-red">Fundamentals of Large Language Models (LLMs)</span>

<details>
<summary>LLM Basic</summary>

::::tip
:::note[Question: What are the main open-source LLM families currently available?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
- Llama: Decoder-Only
- Mistral: Decoder-Only (MoE in Mixtral)
- Gemma: Decoder-Only
- Phi: Decoder-Only
- Qwen: Decoder-Only (dense + MoE)
- DeepSeek: Decoder-Only (MoE in V2)
- Falcon: Decoder-Only
- OLMo: Decoder-Only
:::

</details>

:::note[Question: What’s the difference between prefix decoder, causal decoder, and encoder-decoder architectures?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
- **Causal Decoder (Decoder-Only)**: Autoregressive model that generates text left-to-right, attending only to previous tokens.
- **Prefix Decoder (PrefixLM)**: Causal decoder with a bidirectional prefix (input context) followed by autoregressive generation.
- **Encoder-Decoder (Seq2Seq)**: Two separate Transformer stacks(Encoder & Decoder)

<details>
<summary>Causal Decoder</summary>

- Prompt
> Translate to French: The cat is on the mat.
- Generation (autoregressive, causal mask):
> Le [only sees "Le"]
>
> Le chat [sees "Le chat"]
> 
> Le chat est [sees "Le chat est"]
>
> Le chat est sur [sees up to "sur"]
> 
> Le chat est sur le [sees up to "le"]
>
> Le chat est sur le tapis. [final]
- Summary
> **Cannot see future tokens**
>
> **Cannot see full input bidirectionally — but works via prompt engineering**

</details>

<details>
<summary>Prefix Decoder</summary>

- Input Format
> [Prefix] The cat is on the mat. [SEP] Translate to French: [Generate] Le chat est sur le tapis.
- Attention
> **Prefix** (The cat is on the mat. [SEP] Translate to French:) → bidirectional

</details>

<details>
<summary>Encoder-Decoder</summary>

</details>

:::

</details>

:::note[Question: What is the training objective of large language models?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
LLMs are trained to predict the next token in a sequence.
:::

</details>

:::note[Question: Why are most modern LLMs decoder-only architectures?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
Most modern LLMs are decoder-only because this architecture is the simplest, fastest, and most flexible for large-scale text generation.
Below is the full reasoning, broken into the fundamental, engineering, and use-case levels.

 - Decoder-only naturally matches the training objective
 - Simpler architecture → easier scaling
 - Better for long-context generation
 - Fits universal multitask learning with a single text stream
 - Aligns with inference needs
    - streaming output
    - token-by-token generation
    - low latency
    - high throughput
    - continuous prompts
:::

</details>

:::note[Question: Explain the difference between encoder-only, decoder-only, and encoder-decoder models.]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
- <span class="def-mono-blue">Encoder-only Models (BERT, RoBERTa, DeBERTa, ELECTRA)</span>
    - classification (sentiment, fraud detection)
    - named entity recognition
    - sentence similarity
    - search / embeddings
    - anomaly or pattern detection
- <span class="def-mono-blue">Decoder-only Models (GPT, Llama, Mixtral, Gemma, Qwen)</span>
    - Text generation
    - Multi-task language modeling
    - Anything that treats tasks as text → text in one stream
- <span class="def-mono-blue">Encoder–Decoder (Seq2Seq) Models (T5, FLAN-T5, BART, mT5, early Transformer models)</span>
    - Translation
    - Summarization
    - Text-to-text tasks with clear input → output mapping
:::

</details>

:::note[Question: What’s the difference between prefix LM and causal LM?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
- <span class="def-mono-red">Causal LM</span>: every token can only attend to previous tokens.
- <span class="def-mono-red">Prefix LM</span>: the prefix can be fully bidirectional, while the rest is generated causally.


|Feature|Causal LM|Prefix LM|
|---|---|---|
|Attention|Strictly left-to-right|Prefix: full; Generation: causal|
|Use case|Free-form generation|Conditional generation, prefix tuning|
|Examples|GPT, Llama, Mixtral|T5 (prefix mode), UL2, some prompt-tuning models|
|Future access?|No|Only inside prefix|
|Mask complexity|Simple|Mixed masks|
:::

</details>

::::

</details>

<details>
<summary>Layer Normalization Variants</summary>

::::tip
:::note[Question: Comparison of LayerNorm vs BatchNorm vs RMSNorm?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
|Norm|Formula|Pros|Cons|
|---|---|---|---|
|BatchNorm|Normalize across batch|Great for CNNs|Bad for variable batch / autoregressive decoding|
|LayerNorm|Normalize across hidden dim|Stable for Transformers|Slightly more compute than RMSNorm|
|RMSNorm|Normalize only scale|Faster, more stable in LLMs|No centering → sometimes slightly less expressive|
:::

</details>

:::note[Question: What’s the core idea of DeepNorm?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
**DeepNorm keeps the Transformer stable at extreme depths by scaling the residual connections proportionally to the square root of the model depth.**
:::

</details>

:::note[Question: What are the advantages of DeepNorm?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
**DeepNorm = deep models that actually train and perform well, without tricks.**

- Enables Extremely Deep Transformers (1,000+ layers)
- Superior Training Stability
- Improved Optimization Landscape
- Better Performance on Downstream Tasks
- No Architectural Overhead
- Robust Across Scales and Tasks
:::

</details>

:::note[Question: What are the differences when applying LayerNorm at different positions in LLMs?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
- <span class="def-mono-red">~~Pre-NormPost-Norm~~ (Original Transformer, 2017)</span>: Normalizes after adding the residual.
    - Pros:
        - Fairly stable for shallow models (<12 layers)
        - Works well in classic NMT models
    - Cons:
        - Fails to train deep models (vanishing/exploding gradients)
        - Poor gradient flow
        - Not used in modern LLMs
- Pre-Norm (Current Standard in GPT/LLaMA): Normalize before attention or feed-forward
    - Pros:
        - Much more stable for deep Transformers
        - Great training stability up to hundreds of layers
        - Works well with small batch sizes
        - Default in GPT-2/3, LLaMA, Mistral, Gemma, Phi-3, Qwen2
    - Cons:
        - Residual stream grows in magnitude unless controlled (→ RMSNorm or DeepNorm often added)
        - Slightly diminished expressive capacity compared to Post-Norm (but negligible in practice)
- Sandwich-Norm: LayerNorm applied before AND after sublayers.
    - Pros:
        - Extra stability & smoothness
        - Improved optimization in some NMT models

    - Cons:
        - Expensive (two norms per sublayer)
        - Rarely used in large decoder-only LLMs



🧠 Why LayerNorm position matters

    1. Training Stability
        •	Pre-Norm prevents exploding residuals
        •	Post-Norm accumulates errors → unstable for deep models
    2. Gradient Flow
        - Residuals in Pre-Norm allow gradients to bypass the sublayers directly.
:::

</details>

:::note[Question: Which normalization method is used in different LLM architectures?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
**Large decoder-only LLMs almost universally use RMSNorm + Pre-Norm.**
:::

</details>

::::

</details>

<details>
<summary>Activation Functions in LLMs</summary>

::::tip
:::note[Question: What’s the formula for the FFN (Feed-Forward Network) block?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
- <span class="def-mono-red">Standard FFN Formula</span>

    $$\text{FFN}(x) = W_2 \, \sigma(W_1 x + b_1) + b_2$$

    $$W_1 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{mode$$l}}}$$

    $$b_1 \in \mathbb{R}^{d_{\text{ff}}}$$

    $$W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$$

    $$b_2 \in \mathbb{R}^{d_{\text{model}}}$$

    $$\sigma = \text{activation} \text{ }  \text{function} \text{(ReLU in original Transformer, GELU in GPT, SwiGLU/GeLU-linear in modern LLMs)}$$

- <span class="def-mono-blue">Gated FFN in LLMs</span>

    $$\text{FFN}(x) = W_3 \left( \text{Swish}(W_1x) \odot W_2x \right)$$

    $$\text{Swish}(u) = u \cdot \sigma(u)$$

    $$W_1, W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$$

    $$W_3 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$$
:::

</details>

:::note[Question: What’s the GeLU formula?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
**Gaussian Error Linear Unit (GeLU)**

$$\text{GeLU}(x) = \frac{x}{2}\left(1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$

$$\operatorname{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \, dt$$
:::

</details>

:::note[Question: What’s the Swish formula?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
**Swish is a smooth, non-monotonic activation.**

$$\text{Swish}(x) = \frac{x}{1 + e^{-x}}$$
:::

</details>

:::note[Question: What’s the formula of an FFN block with GLU (Gated Linear Unit)?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What’s the formula of a GLU block using GeLU?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What’s the formula of a GLU block using Swish?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Which activation functions do popular LLMs use?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are the differences between Adam and SGD optimizers?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

<details>
<summary>Attention Mechanisms — Advanced Topics</summary>

::::tip
:::note[Question: What are the problems with traditional attention?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are the directions of improvement for attention?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are the attention variants?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What issues exist in multi-head attention?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Explain Multi-Query Attention (MQA).]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Compare Multi-head, Multi-Query, and Grouped-Query Attention.]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are the benefits of MQA?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Which models use MQA or GQA?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Why was FlashAttention introduced? Briefly explain its core idea.]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are FlashAttention advantages?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Which models implement FlashAttention?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What is parallel transformer block?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What’s the computational complexity of attention and how can it be improved?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Compare MHA, GQA, and MQA — what are their key differences?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

<details>
<summary>Cross-Attention</summary>

::::tip
:::note[Question: Why do we need Cross-Attention?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Explain Cross-Attention.]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Compare Cross-Attention and Self-Attention — similarities and differences.]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Provide a code implementation of Cross-Attention.]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are its application scenarios?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are the advantages and challenges of Cross-Attention?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

<details>
<summary>Transformer Operations</summary>

::::tip
:::note[Question: How to load a BERT model using transformers?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to output a specific hidden_state from BERT using transformers?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to get the final or intermediate layer vector outputs of BERT?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

<details>
<summary>LLM Loss Functions</summary>

::::tip
:::note[Question: What is KL divergence?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Write the cross-entropy loss and explain its meaning.]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What’s the difference between KL divergence and cross-entropy?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to handle large loss differences in multi-task learning?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Why is cross-entropy preferred over MSE for classification tasks?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What is information gain?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to compute softmax and cross-entropy loss (and binary cross-entropy)?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What if the exponential term in softmax overflows the float limit?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

<details>
<summary>Similarity & Contrastive Learning</summary>

::::tip
:::note[Question: Besides cosine similarity, what other similarity metrics exist?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What is contrastive learning?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How important are negative samples in contrastive learning, and how to handle costly negative sampling?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

- <span class="def-mono-red">Advanced Topics in LLMs</span>

<details>
<summary>Advanced LLM</summary>

::::tip
:::note[Question: What is a generative large model?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How do LLMs make generated text diverse and non-repetitive?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What is the repetition problem (LLM echo problem)? Why does it happen? How can it be mitigated?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Can LLaMA handle infinitely long inputs? Explain why?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: When should you use BERT vs. LLaMA / ChatGLM models?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Do different domains require their own domain-specific LLMs? Why?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to enable an LLM to process longer texts?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

- <span class="def-mono-red">Fine-Tuning Large Models</span>

<details>
<summary>General Fine-Tuning</summary>

::::tip
:::note[Question: Why does the loss drop suddenly in the second epoch during SFT?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How much VRAM is needed for full fine-tuning?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Why do models seem dumber after SFT?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to construct instruction fine-tuning datasets?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to improve prompt representativeness?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to increase prompt data volume?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to select domain data for continued pretraining?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to prevent forgetting general abilities after domain tuning?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to make the model learn more knowledge during pretraining?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: When performing SFT, should the base model be Chat or Base?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What’s the input/output format for domain fine-tuning?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to build a domain evaluation set?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Is vocabulary expansion necessary? Why?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to train your own LLM?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are the benefits of instruction fine-tuning?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: During which stage — pretraining or fine-tuning — is knowledge injected?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

<details>
<summary>SFT Tricks</summary>

::::tip
:::note[Question: What’s the typical SFT workflow?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are key aspects of training data?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to choose between large and small models?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to ensure multi-task training balance?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Can SFT learn knowledge at all?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to select datasets effectively?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

<details>
<summary>Training Experience</summary>

::::tip
:::note[Question: How to choose a distributed training framework?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are key LLM training tips?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to choose model size?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to select GPU accelerators?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

- <span class="def-mono-red">LangChain and Agent-Based Systems</span>


<details>
<summary>LangChain Core</summary>

::::tip
:::note[Question: What is LangChain?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are its core concepts?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Components and Chains]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Prompt Templates and Values]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Example Selectors]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Output Parsers]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Indexes and Retrievers]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Chat Message History]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Agents and Toolkits]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

<details>
<summary>Long-Term Memory in Multi-Turn Conversations</summary>

::::tip
:::note[Question: How can Agents access conversation context?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Retrieve full history]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Use sliding window for recent context]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

<details>
<summary>Practical RAG Q&A using LangChain</summary>

::::tip
:::note[Question: (Practical implementation questions about RAG apps in LangChain)]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

- <span class="def-mono-red">Retrieval-Augmented Generation (RAG)</span>


<details>
<summary>RAG Basics</summary>

::::tip
:::note[Question: Why do LLMs need an external (vector) knowledge base?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What’s the overall workflow of LLM+VectorDB document chat?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are the core technologies?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to build an effective prompt template?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

<details>
<summary>RAG Concepts</summary>

::::tip
:::note[Question: What are the limitations of base LLMs that RAG solves?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What is RAG?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to obtain accurate semantic representations?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to align query/document semantic spaces?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to match retrieval model output with LLM preferences?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to improve results via post-retrieval processing?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to optimize generator adaptation to inputs?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are the benefits of using RAG?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

<details>
<summary>RAG Layout Analysis</summary>

::::tip
:::note[Question: Why is PDF parsing necessary?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are common methods and their differences?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What problems exist in PDF parsing?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Why is table recognition important?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are the main methods?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Traditional methods]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: pdfplumber extraction techniques]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Why do we need text chunking?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are common chunking strategies (regex, Spacy, LangChain, etc.)?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

<details>
<summary>RAG Retrieval Strategies</summary>

::::tip
:::note[Question: Why use LLMs to assist recall?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: HYDE approach: idea and issues]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: FLARE approach: idea and recall strategies]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Why construct hard negative samples?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Random sampling vs. Top-K hard negative sampling]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

<details>
<summary>RAG Evaluation</summary>

::::tip
:::note[Question: Why evaluate RAG?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are the evaluation methods, metrics, and frameworks?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

<details>
<summary>RAG Optimization</summary>

::::tip
:::note[Question: What are the optimization strategies for retrieval and generation modules?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to enhance context using knowledge graphs (KGs)?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are the problems with vector-based context augmentation?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How can KG-based methods improve it?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What are the main pain points in RAG and their solutions?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Content missing]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Top-ranked docs missed]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Context loss]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Failure to extract answers]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Explain RAG-Fusion. Why it’s needed,Core technologies,Workflow, and Advantages]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

<details>
<summary>Graph RAG</summary>

::::tip
:::note[Question: Why do we need Graph RAG?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What is Graph RAG and how does it work? Show a code example and use case.]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How to improve ranking optimization in Graph RAG?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

- <span class="def-mono-red">Parameter-Efficient Fine-Tuning (PEFT)</span>

<details>
<summary>PEFT Fundamentals</summary>

::::tip
:::note[Question: What is fine-tuning, and how is it performed?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: Why do we need PEFT?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What is PEFT and its advantages?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

<details>
<summary>Adapter Tuning</summary>

::::tip
:::note[Question: Why use adapter-tuning?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: What’s the core idea behind adapter-tuning?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

:::note[Question: How does it differ from full fine-tuning?]{.question-note}
:::

<details>
<summary>Answer</summary>

:::tip
:::

</details>

::::

</details>

::::
