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

Language models learn distributions over tokens; useful applications also depend on sound evaluation, efficient training and inference, and reliable access to evidence. These questions cover those foundations through fine-tuning, agents, and retrieval-augmented generation.

<!-- more -->

Each question has a **brief answer** for interview recall and an expandable **detailed answer** with mechanisms, assumptions, examples, and tradeoffs. Open a topic to study its questions. Technical review updated September 19, 2026; named model examples refer to specific releases, and linked library documentation can evolve.

## Machine Learning

<details>
<summary>Machine Learning Concepts</summary>

:::note[Question: How would you describe the concept of machine learning in your own words?]{.question-note}
:::

**Brief answer.** Machine learning fits models to data so they can make predictions, discover structure, or choose actions on new inputs.

<details>
<summary>Detailed answer</summary>

A developer specifies a model, objective, and training procedure; an optimizer learns parameters from examples. For instance, a spam classifier learns correlations between message features and labels rather than relying entirely on handwritten rules. Learning does not guarantee improvement on future data: success requires an appropriate objective, representative data, and evaluation on held-out examples.

</details>

:::note[Question: Can you give a few examples of real-world areas where machine learning is particularly effective?]{.question-note}
:::

**Brief answer.** Useful applications include spam detection, recommendation, speech recognition, demand forecasting, and visual defect detection.

<details>
<summary>Detailed answer</summary>

These tasks contain patterns that are difficult to enumerate as rules. A recommender ranks items from interaction history; a speech model maps audio to text; a forecasting model uses past demand and known covariates. ML works best when the data contains predictive signal and mistakes can be measured. Compare against simple rules or statistical baselines before accepting the added operational complexity.

</details>

:::note[Question: What are some typical problems addressed with unsupervised learning methods?]{.question-note}
:::

**Brief answer.** Unsupervised learning commonly addresses clustering, dimensionality reduction, density estimation, and anomaly discovery without task labels.

<details>
<summary>Detailed answer</summary>

Clustering can reveal customer segments, while principal component analysis compresses correlated measurements. Anomaly detection can flag unusual observations, but unusual does not automatically mean fraudulent or defective. Association-rule mining discovers co-occurring items. Because there are no target labels, validation often combines structural metrics, stability checks, and usefulness on a downstream task.

</details>

:::note[Question: Would detecting spam emails be treated as a supervised or unsupervised learning problem, and why?]{.question-note}
:::

**Brief answer.** Spam detection is usually supervised binary classification because training messages have spam/not-spam labels.

<details>
<summary>Detailed answer</summary>

The model learns a decision function from those examples and predicts labels or probabilities for new messages. In practice, semi-supervised learning can use unlabeled email, and anomaly detection can help find new campaigns. Evaluate precision and recall under realistic class imbalance; a high overall accuracy can hide costly false positives.

</details>

:::note[Question: What does the term ‘out-of-core learning’ refer to in machine learning?]{.question-note}
:::

**Brief answer.** Out-of-core learning processes data in chunks because the full dataset does not fit in RAM.

<details>
<summary>Detailed answer</summary>

A typical pipeline streams records, computes features with bounded memory, and updates an incremental estimator. The feature representation must remain consistent across chunks; fitting a new vocabulary independently on every chunk would break this contract. Mini-batch training enables out-of-core processing only if loading, preprocessing, and caching also avoid materializing the full dataset. See [scikit-learn's scaling guide](https://scikit-learn.org/stable/computing/scaling_strategies.html).

</details>

:::note[Question: How can you distinguish between model parameters and hyperparameters?]{.question-note}
:::

**Brief answer.** Parameters are learned model values; hyperparameters configure the model or learning process.

<details>
<summary>Detailed answer</summary>

Weights, biases, and learned embeddings are parameters. Learning rate, weight decay, layer count, and batch size are hyperparameters. Hyperparameters can be selected by a search algorithm and can change during training, as with a learning-rate schedule; they are not necessarily fixed manually before every run. Validation data guides their selection, while test data remains reserved for final evaluation.

</details>

:::note[Question: What are some major difficulties or limitations commonly faced when building machine learning systems?]{.question-note}
:::

**Brief answer.** Common difficulties include poor data quality, distribution shift, leakage, inadequate evaluation, underfitting, overfitting, and deployment constraints.

<details>
<summary>Detailed answer</summary>

A model can fail even with low training loss: labels may be inconsistent, the target may poorly represent the product goal, or production inputs may differ from training inputs. Latency, memory, privacy, and feedback loops also matter. Diagnose errors by relevant slices, inspect the data pipeline, and compare with a simple baseline before changing model capacity.

</details>

:::note[Question: If a model performs well on training data but poorly on unseen data, what issue is occurring, and how might you address it?]{.question-note}
:::

**Brief answer.** A large training-to-test gap often indicates overfitting, but distribution shift or a pipeline mismatch can produce the same symptom.

<details>
<summary>Detailed answer</summary>

First evaluate untouched data drawn from the training distribution. If performance is already poor there, consider regularization, earlier stopping, deduplication, or more representative examples. If that holdout works but production-like data fails, investigate domain shift. Also check preprocessing consistency, label definitions, and group or time leakage. Simplifying the model is one possible remedy, not a universal fix.

</details>

:::note[Question: What is a test dataset used for, and why is it essential in evaluating a model’s performance?]{.question-note}
:::

**Brief answer.** A test set estimates generalization after model development, provided it is independent and representative of the intended deployment distribution.

<details>
<summary>Detailed answer</summary>

Keep it out of training, hyperparameter tuning, prompt selection, and repeated model selection. Use group-based or chronological splits when examples share users, documents, or time dependencies. Report uncertainty and important slices alongside the aggregate score. A test set does not automatically give an unbiased estimate of real-world performance if its sampling or labels are flawed.

</details>

:::note[Question: What role does a validation set play during the model development process?]{.question-note}
:::

**Brief answer.** A validation set guides model selection, hyperparameter tuning, and decisions such as early stopping.

<details>
<summary>Detailed answer</summary>

Repeatedly comparing candidates on validation data can itself overfit that set. Keep a separate final test set, and use cross-validation or nested cross-validation when data is limited and the selection procedure warrants it. Fit preprocessing only on the training portion of each split. Good validation design improves decision quality; it does not guarantee generalization.

</details>

:::note[Question: What is a train-dev dataset, in what situations would you create one, and how is it applied during model evaluation?]{.question-note}
:::

**Brief answer.** A train-dev set is an unused holdout from the training distribution that helps distinguish overfitting from distribution mismatch.

<details>
<summary>Detailed answer</summary>

Suppose training error is 2%, train-dev error is 3%, and production-like validation error is 15%. The small first gap and large second gap suggest distribution mismatch is the main problem. A 2% versus 14% train-dev gap instead suggests poor generalization within the training distribution. Use consistent metrics and examine sampling uncertainty; the comparison is diagnostic evidence rather than a perfect decomposition of error.

</details>

:::note[Question: Why is it problematic to adjust hyperparameters based on test set performance?]{.question-note}
:::

**Brief answer.** Tuning on the test set leaks evaluation information into model selection and makes the reported result optimistically biased.

<details>
<summary>Detailed answer</summary>

Even without taking gradients on test examples, choosing the best of many candidates using test scores adapts the system to that set. Tune on validation data and use the test set only after choices are settled. If extensive test-driven iteration has already happened, obtain a new independent holdout or explicitly report that the old test set became development data.

</details>

</details>

## LLM Fundamentals

:::note[Question: Explain bias-variance tradeoff. How does it manifest in LLMs?]{.question-note}
:::

**Brief answer.** Bias measures systematic error across possible training sets; variance measures how much predictions change with the training sample. Both help explain generalization, but LLM scaling does not follow a simple “larger means more overfitting” rule.

<details>
<summary>Detailed answer</summary>

For squared-error regression, expected prediction error decomposes into squared bias, variance, and irreducible noise. Cross-entropy language modeling does not use that identical decomposition. In LLM fine-tuning, limited or repetitive data can encourage memorization, while insufficient capacity or optimization can cause underfitting. More parameters can improve generalization when accompanied by adequate data and training. Hallucination is not a synonym for high variance or overfitting, and a single pretraining epoch can still contain trillions of tokens.

</details>

:::note[Question: What is the difference between L1 and L2 regularization? When would you use elastic net in an LLM fine-tune?]{.question-note}
:::

**Brief answer.** L1 penalizes absolute weight magnitude and can encourage sparsity; L2 penalizes squared magnitude and encourages shrinkage. Elastic net combines them when both effects serve a measured objective.

<details>
<summary>Detailed answer</summary>

With trainable parameters or updates $w$:

$$
L=L_{\mathrm{task}}+\lambda_1\lVert w\rVert_1+\lambda_2\lVert w\rVert_2^2.
$$

Elastic net can be useful for a sparse task head or adapter, but it is not a standard requirement for LLM fine-tuning. State whether the penalty targets adapter weights, full weights, or the update from the pretrained checkpoint. L1 can yield exact zeros with an appropriate optimization method; ordinary gradient steps do not guarantee them.

Neither penalty inherently makes the data-fitting loss robust to target outliers. L2's unique-solution property for strictly convex ridge regression does not extend to nonconvex neural networks. Sparse weights do not automatically accelerate dense GPU kernels. Also, AdamW's decoupled weight decay differs from adding an L2 penalty to Adam's loss. See [AdamW](https://arxiv.org/abs/1711.05101).

</details>

:::note[Question: How does dropout approximate an ensemble, and when is the geometric-mean equivalence exact?]{.question-note}
:::

**Brief answer.** Dropout trains shared-weight subnetworks and approximates their ensemble at inference. Exact equivalence is limited to special cases; the relevant concept is a geometric mean, not a geometric distribution.

<details>
<summary>Detailed answer</summary>

Let $m_i\sim\mathrm{Bernoulli}(q)$ and let a softmax linear model have logits $z_m=W(m\odot x)+b$. Define its normalized weighted geometric ensemble by:

$$
\widetilde p_k=\frac{\exp(\mathbb E_m[\log p_m(k)])}{\sum_j\exp(\mathbb E_m[\log p_m(j)])}
=\operatorname{softmax}(\mathbb E_m[z_m])_k
=\operatorname{softmax}(W(qx)+b)_k.
$$

The expected log-normalizer cancels across classes. This is not the arithmetic average $\mathbb E_m[p_m]$. In a nonlinear deep network, replacing every random mask by its mean generally does not reproduce the expected logits, so the exact argument fails. Modern inverted dropout scales retained activations by $1/q$ during training and makes dropout an identity at evaluation. See the [dropout paper](https://jmlr.org/papers/v15/srivastava14a.html).

</details>

:::note[Question: What is the curse of dimensionality, and what problem do positional encodings actually solve?]{.question-note}
:::

**Brief answer.** The curse of dimensionality concerns the data and computation needed to learn in high-dimensional spaces. Positional encodings supply order information; they do not generally solve that curse.

<details>
<summary>Detailed answer</summary>

Covering a $d$-dimensional grid with $k$ values per coordinate requires $k^d$ points. Distance concentration can also weaken nearest-neighbor intuition, depending on the distribution and metric. Real data may occupy a much lower-dimensional manifold.

Unmasked self-attention without positional information is permutation-equivariant: permuting inputs permutes outputs. Positional embeddings, relative biases, or rotary embeddings make position available. A causal mask already introduces directional structure, but does not make explicit positional methods a dimensionality-reduction technique. Sinusoidal or rotary formulas can be evaluated at new positions; reliable length extrapolation still requires validation. See the [Transformer paper](https://arxiv.org/abs/1706.03762).

</details>

:::note[Question: Explain maximum likelihood estimation for language modeling.]{.question-note}
:::

**Brief answer.** Maximum likelihood estimation chooses parameters that assign high probability to observed training sequences.

<details>
<summary>Detailed answer</summary>

For an autoregressive model, factor the probability using the chain rule and maximize its log:

$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}),\qquad
\theta^*=\arg\max_\theta\sum_{x\in D}\sum_t\log p_\theta(x_t\mid x_{<t}).
$$

A beginning-of-sequence convention defines the initial context. Teacher forcing supplies observed prefixes during training. Taking logs turns products into sums; minimizing their negative gives the usual next-token cross-entropy objective. This trains a distribution over text, not a direct truth-verification objective.

</details>

:::note[Question: What is negative log-likelihood? Write the per-token loss for GPT.]{.question-note}
:::

**Brief answer.** The per-token negative log-likelihood is $\ell_t=-\log p_\theta(x_t\mid x_{<t})$; GPT-style training averages it over supervised tokens.

<details>
<summary>Detailed answer</summary>

If $z_{t,v}$ are vocabulary logits predicting token $x_t$, then:

$$
\ell_t=-z_{t,x_t}+\log\sum_{v=1}^{V}e^{z_{t,v}},\qquad
L=\frac{\sum_t m_t\ell_t}{\sum_t m_t}.
$$

Here $m_t$ is 1 for supervised targets and 0 for padding or excluded prompt tokens. The model's output at a sequence position usually predicts the following token, so labels must be shifted correctly. A probability of 0.5 gives about 0.693 nats of loss; 0.01 gives about 4.605 nats. A sequence sum and a per-token mean have different scales.

</details>

:::note[Question: Compare cross-entropy, perplexity, and BLEU. When is perplexity misleading?]{.question-note}
:::

**Brief answer.** Cross-entropy measures token prediction loss; perplexity exponentiates that loss; BLEU measures clipped n-gram overlap with reference translations.

<details>
<summary>Detailed answer</summary>

For mean NLL $H$ measured with natural logs, $\mathrm{PPL}=e^H$. BLEU combines modified n-gram precision with a brevity penalty and does not need model probabilities. It can reward high overlap despite a crucial factual error and penalize valid paraphrases.

Compare perplexities only with compatible tokenization, evaluation text, context, and masking. Fluency alone does not determine a sentence's perplexity, and changing “cat” to “dog” does not establish either a perplexity or BLEU score. Autoregressive encoder-decoder models also have well-defined conditional perplexity. Evaluate factuality and task success separately. See [perplexity evaluation](https://huggingface.co/docs/transformers/perplexity) and the [BLEU paper](https://aclanthology.org/P02-1040/).

</details>

:::note[Question: Why use label smoothing, and how is its modified loss derived?]{.question-note}
:::

**Brief answer.** Label smoothing replaces one-hot targets with a mixture of the target and a prior, discouraging overconfidence. It is useful in some settings but is not universal in LLM training.

<details>
<summary>Detailed answer</summary>

For vocabulary size $V$, one-hot target $y$, and a uniform prior:

$$
q_k=(1-\epsilon)y_k+\frac{\epsilon}{V},\qquad
L=-\sum_kq_k\log p_k
=-(1-\epsilon)\log p_y-\frac{\epsilon}{V}\sum_k\log p_k.
$$

The correct class receives $1-\epsilon+\epsilon/V$. An alternative convention assigns $1-\epsilon$ to the correct class and $\epsilon/(V-1)$ to each other class; do not mix the two definitions. Benefits depend on the task and smoothing strength, and the smoothed training loss is not ordinary hard-target perplexity. See [PyTorch cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

</details>

:::note[Question: What is the difference between hard and soft attention?]{.question-note}
:::

**Brief answer.** Soft attention forms a differentiable weighted sum of values; hard attention makes discrete selections whose selection step is generally not differentiable.

<details>
<summary>Detailed answer</summary>

Standard attention computes weights with softmax and combines all allowed values. Hard attention might sample one location or select a subset; training the selection may require a score-function estimator, relaxation, or another surrogate. A fixed causal or sparse mask does not by itself make attention “hard”: the permitted positions can still receive continuous softmax weights.

</details>

## Fundamentals of Large Language Models (LLMs)

<details>
<summary>LLM Basic</summary>

:::note[Question: What are representative open-weight LLM families, and how do they differ from fully open-source releases?]{.question-note}
:::

**Brief answer.** Well-known released-weight families include Llama, Mistral/Mixtral, Gemma, Phi, Qwen, DeepSeek, Falcon, and OLMo. “Open weights” and “open source” are not interchangeable.

<details>
<summary>Detailed answer</summary>

This is a representative family list, not a ranking or exhaustive catalog of the newest releases. Check each checkpoint's license, model card, architecture, training-data disclosure, and permitted uses. A family can contain dense, mixture-of-experts, and multimodal variants; do not assign every release the properties of one model. Start with the official [Llama](https://github.com/meta-llama/llama-models), [Qwen](https://github.com/QwenLM), [DeepSeek](https://github.com/deepseek-ai), and [OLMo](https://allenai.org/olmo) resources.

</details>

:::note[Question: What’s the difference between prefix decoder, causal decoder, and encoder-decoder architectures?]{.question-note}
:::

**Brief answer.** A causal decoder uses causal self-attention; a prefix decoder adds bidirectional attention within a designated prefix; an encoder-decoder uses a separate encoder and decoder connected by cross-attention.

<details>
<summary>Detailed answer</summary>

For translation, a causal decoder generates French tokens while attending to all preceding prompt and output tokens. It does not lose access to the English prompt. A prefix LM lets the English prefix tokens contextualize one another bidirectionally, while output tokens attend to the prefix and preceding outputs. An encoder-decoder encodes the source bidirectionally, then a causal decoder cross-attends to those representations. The distinction is the attention connectivity, not merely the wording of the prompt.

</details>

:::note[Question: What is the training objective of large language models?]{.question-note}
:::

**Brief answer.** Autoregressive LLMs usually pretrain with next-token prediction, but language models can use masked-token, denoising, or mixed objectives; post-training adds other objectives.

<details>
<summary>Detailed answer</summary>

BERT predicts selected masked tokens, while T5 reconstructs corrupted text. Decoder-only causal models maximize sequence likelihood. Supervised fine-tuning trains on desired responses, often masking prompt tokens from the loss. Preference learning and reinforcement learning optimize preference or reward objectives. Always specify the model and training stage before saying “the LLM objective.”

</details>

:::note[Question: Why are most modern LLMs decoder-only architectures?]{.question-note}
:::

**Brief answer.** Decoder-only models pair a simple single-stream architecture with scalable next-token training and flexible prompted generation.

<details>
<summary>Detailed answer</summary>

Many tasks can be serialized as a prefix followed by a completion. A single stack simplifies training infrastructure, and cached keys and values support incremental decoding. Their prevalence also reflects successful scaling recipes and ecosystem investment. They are not inherently the fastest or best for every task: encoder-decoder models can be competitive for conditional generation, and encoder-only models can be efficient for classification or retrieval. Long-context quality depends on more than the architecture label.

</details>

:::note[Question: Explain the difference between encoder-only, decoder-only, and encoder-decoder models.]{.question-note}
:::

**Brief answer.** Encoder-only models build contextual representations, decoder-only models generate autoregressively, and encoder-decoder models map a separately encoded input to an output sequence.

<details>
<summary>Detailed answer</summary>

BERT-style encoders commonly use bidirectional attention for classification, tagging, and reranking. A suitable pooling method and retrieval training are still needed for high-quality sentence embeddings. GPT/Llama-style decoders predict the next token using a causal mask. T5/BART-style encoder-decoders use bidirectional source attention plus causal output attention and cross-attention. These are architectural tendencies rather than absolute restrictions on possible tasks.

</details>

:::note[Question: What’s the difference between prefix LM and causal LM?]{.question-note}
:::

**Brief answer.** A causal LM masks future positions everywhere; a prefix LM allows bidirectional attention inside its input prefix and causal attention in its generated suffix.

<details>
<summary>Detailed answer</summary>

For prefix length $P$, prefix queries can attend only to prefix positions, while a suffix query at position $i$ can attend to every position up to $i$. Both usually include the position itself; the representation predicts the next token. Prefix language modeling is not prefix tuning: prefix tuning learns small task-specific continuous prefixes and need not change a causal model's mask. The [T5 study](https://jmlr.org/papers/v21/20-074.html) compares architectural and objective variants, including prefix language modeling.

</details>

</details>

<details>
<summary>Layer Normalization Variants</summary>

:::note[Question: How do LayerNorm, BatchNorm, and RMSNorm compare?]{.question-note}
:::

**Brief answer.** LayerNorm normalizes each token across hidden features; RMSNorm rescales those features without subtracting their mean; BatchNorm uses batch-dependent statistics during training.

<details>
<summary>Detailed answer</summary>

For token vector $x\in\mathbb R^d$:

$$
\operatorname{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,\qquad
\operatorname{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{d^{-1}\sum_i x_i^2+\epsilon}}.
$$

BatchNorm's reduction axes depend on the layer and data layout; it commonly uses running statistics at inference. Batch dependence, padding, and train/inference differences make it awkward for autoregressive language models. RMSNorm saves centering work, but speed and training stability depend on kernels and the full architecture. See the [RMSNorm paper](https://arxiv.org/abs/1910.07467).

</details>

:::note[Question: What’s the core idea of DeepNorm?]{.question-note}
:::

**Brief answer.** DeepNorm stabilizes deep Transformers by combining a scaled residual connection with depth-dependent initialization.

<details>
<summary>Detailed answer</summary>

Its sublayer has the form $x_{l+1}=\operatorname{LN}(\alpha x_l+F_l(x_l))$. For a decoder-only stack of $L$ blocks, the paper gives $\alpha=(2L)^{1/4}$ and initialization gain $\beta=(8L)^{-1/4}$ for selected projections. Encoder-decoder stacks use different constants. This is a fourth-root rule in that case, not a generic square-root scaling rule, and it is not simply an alternative normalization statistic. See [DeepNet, Figure 2](https://arxiv.org/html/2203.00555v1).

</details>

:::note[Question: What are the advantages of DeepNorm?]{.question-note}
:::

**Brief answer.** DeepNorm offers a recipe for stabilizing very deep models while retaining a post-normalization architecture.

<details>
<summary>Detailed answer</summary>

The DeepNet study demonstrated Transformers up to 1,000 layers and reported gains on its translation benchmarks. Residual scaling introduces little computation, but it must accompany the prescribed initialization. The results do not establish that any 1,000-layer LLM will train well, outperform a shallower model, or be efficient to serve. Depth still adds sequential computation. See the [DeepNet experiments](https://arxiv.org/abs/2203.00555).

</details>

:::note[Question: What are the differences when applying LayerNorm at different positions in LLMs?]{.question-note}
:::

**Brief answer.** Pre-norm normalizes before a sublayer; post-norm normalizes after the residual addition; sandwich variants add normalization at both locations around a sublayer.

<details>
<summary>Detailed answer</summary>

Two common forms are:

$$
\text{Pre-norm: } y=x+F(\operatorname{LN}(x)),\qquad
\text{Post-norm: } y=\operatorname{LN}(x+F(x)).
$$

Pre-norm gives gradients an identity path through the residual stream and often simplifies optimization at depth. It does not by itself bound residual magnitudes. Post-norm can work well with suitable initialization and learning-rate schedules, including deep variants such as DeepNorm. “Sandwich norm” is not a single universally identical implementation; specify the exact equations. See [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745).

</details>

:::note[Question: Which normalization method is used in different LLM architectures?]{.question-note}
:::

**Brief answer.** Normalization varies by checkpoint: BERT uses post-LayerNorm, GPT-2 uses pre-LayerNorm, and Llama uses pre-RMSNorm.

<details>
<summary>Detailed answer</summary>

The original Transformer uses post-LayerNorm. T5 uses a mean-free normalization closely related to RMSNorm, and Gemma 2 adds normalization around sublayers beyond a simple pre-norm layout. Therefore, “all modern LLMs use pre-RMSNorm” is too broad. Confirm the normalization class, placement, epsilon, and learned scale/bias in the model configuration and implementation. Examples are documented in [BERT](https://huggingface.co/docs/transformers/model_doc/bert), [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2), and [Gemma 2](https://arxiv.org/abs/2408.00118).

</details>

</details>

<details>
<summary>Activation Functions in LLMs</summary>

:::note[Question: What’s the formula for the FFN (Feed-Forward Network) block?]{.question-note}
:::

**Brief answer.** A standard position-wise FFN applies an input projection, nonlinearity, and output projection independently at each token.

<details>
<summary>Detailed answer</summary>

Using column vectors:

$$
\operatorname{FFN}(x)=W_2\phi(W_1x+b_1)+b_2,
$$

where $x\in\mathbb R^d$, $W_1\in\mathbb R^{h\times d}$, and $W_2\in\mathbb R^{d\times h}$. The hidden width $h$ usually exceeds $d$. Gated variants use two input projections and multiply their outputs before projecting back; applying SiLU to a single projection is not the same as SwiGLU. Some architectures omit biases. Attention mixes tokens; the FFN mixes features within each token.

</details>

:::note[Question: What’s the GeLU formula?]{.question-note}
:::

**Brief answer.** GELU is $\operatorname{GELU}(x)=x\Phi(x)$, where $\Phi$ is the standard normal cumulative distribution function.

<details>
<summary>Detailed answer</summary>

Its exact and common approximate forms are:

$$
\operatorname{GELU}(x)=\frac{x}{2}\left[1+\operatorname{erf}\left(\frac{x}{\sqrt2}\right)\right],
$$

$$
\operatorname{GELU}(x)\approx\frac{x}{2}\left[1+\tanh\left(\sqrt{2/\pi}(x+0.044715x^3)\right)\right].
$$

Unlike ReLU, GELU smoothly attenuates negative values rather than setting all of them to zero. Exact and approximate implementations have slightly different numerics, so preserve the checkpoint's activation convention. See the [GELU paper](https://arxiv.org/abs/1606.08415).

</details>

:::note[Question: What’s the Swish formula?]{.question-note}
:::

**Brief answer.** Swish is $\operatorname{Swish}_\beta(x)=x\,\sigma(\beta x)$; the common $\beta=1$ case is SiLU.

<details>
<summary>Detailed answer</summary>

Here $\sigma(u)=1/(1+e^{-u})$ is the logistic sigmoid. Swish is smooth and non-monotonic: small negative inputs can produce negative outputs, while large positive inputs approach the identity. Some definitions allow a learned or fixed $\beta$; many Transformer implementations simply use SiLU with $\beta=1$. A SwiGLU FFN combines that activation with a separate multiplicative projection. See [Searching for Activation Functions](https://arxiv.org/abs/1710.05941).

</details>

:::note[Question: What’s the formula of an FFN block with GLU (Gated Linear Unit)?]{.question-note}
:::

**Brief answer.** A GLU multiplies a linear value branch by a sigmoid gate, and an FFN adds an output projection.

<details>
<summary>Detailed answer</summary>

With column vectors and hidden width $h$:

$$
\operatorname{FFN}_{\mathrm{GLU}}(x)=W_o\left[(W_vx+b_v)\odot\sigma(W_gx+b_g)\right]+b_o.
$$

$W_v,W_g\in\mathbb R^{h\times d}$ and $W_o\in\mathbb R^{d\times h}$. The gate and value vectors must have the same shape. The elementwise product controls each hidden feature separately; it is not an attention distribution over tokens. See [GLU variants](https://arxiv.org/abs/2002.05202).

</details>

:::note[Question: What’s the formula of a GLU block using GeLU?]{.question-note}
:::

**Brief answer.** GEGLU replaces the sigmoid gate in a GLU with GELU.

<details>
<summary>Detailed answer</summary>

A bias-free column-vector form is:

$$
\operatorname{FFN}_{\mathrm{GEGLU}}(x)=W_o\left[\operatorname{GELU}(W_gx)\odot(W_vx)\right].
$$

Both input projections have shape $h\times d$, and the output projection has shape $d\times h$. Unlike a sigmoid gate, the GELU branch is not bounded between 0 and 1. This block has three projection matrices, so comparisons with an ordinary two-projection FFN should account for hidden width and parameter count. See [GLU variants](https://arxiv.org/abs/2002.05202).

</details>

:::note[Question: What’s the formula of a GLU block using Swish?]{.question-note}
:::

**Brief answer.** SwiGLU replaces the GLU gate with Swish, usually SiLU.

<details>
<summary>Detailed answer</summary>

The common bias-free form is:

$$
\operatorname{FFN}_{\mathrm{SwiGLU}}(x)=W_o\left[\operatorname{SiLU}(W_gx)\odot(W_vx)\right].
$$

The elementwise product is projected back to the model width. For comparable parameter counts, a gated FFN's hidden width is often reduced relative to a standard FFN, though exact ratios depend on the architecture and rounding. The activation alone does not guarantee an accuracy or speed improvement. See [GLU variants](https://arxiv.org/abs/2002.05202).

</details>

:::note[Question: Which activation functions do popular LLMs use?]{.question-note}
:::

**Brief answer.** Examples include GELU in BERT and GPT-2, SwiGLU in Llama and PaLM, and a GELU-gated FFN in the original Gemma models.

<details>
<summary>Detailed answer</summary>

The original Transformer used ReLU. Names such as SiLU in a configuration describe a pointwise activation; the surrounding multiplication and projections determine whether the full block is SwiGLU. Inspect a specific version rather than assuming all models in a family share the same FFN. References: [BERT](https://arxiv.org/abs/1810.04805), [Llama 2](https://arxiv.org/abs/2307.09288), [PaLM](https://arxiv.org/abs/2204.02311), and [Gemma](https://arxiv.org/abs/2403.08295).

</details>

:::note[Question: What are the differences between Adam and SGD optimizers?]{.question-note}
:::

**Brief answer.** SGD updates parameters from gradients, optionally with momentum; Adam also estimates per-coordinate gradient scales to adapt update sizes.

<details>
<summary>Detailed answer</summary>

Vanilla SGD uses $\theta\leftarrow\theta-\eta g$. Adam maintains exponential moving averages of $g$ and $g^2$, bias-corrects them, and updates with $\eta\widehat m/(\sqrt{\widehat v}+\epsilon)$. These two moment buffers increase optimizer memory. AdamW additionally decouples weight decay and is a common Transformer training choice. Neither optimizer universally generalizes better; compare tuned schedules and equal training budgets. See [Adam](https://arxiv.org/abs/1412.6980) and [AdamW](https://arxiv.org/abs/1711.05101).

</details>

</details>

<details>
<summary>Attention Mechanisms — Advanced Topics</summary>

:::note[Question: What are the problems with traditional attention?]{.question-note}
:::

**Brief answer.** Dense attention has quadratic sequence-length work during full-sequence processing, and naive implementations materialize a large attention matrix.

<details>
<summary>Detailed answer</summary>

For length $n$ and hidden width $d$, the score/value products cost $O(n^2d)$; projections add $O(nd^2)$. A naive multi-head implementation stores $O(Hn^2)$ attention entries. During cached one-token decoding, attention over an existing context is linear in that context length, but reading the growing KV cache can become a bandwidth bottleneck. Large context windows also create a modeling challenge: access to distant evidence does not guarantee its effective use.

</details>

:::note[Question: What are the directions of improvement for attention?]{.question-note}
:::

**Brief answer.** Improve attention by reducing memory traffic, restricting connectivity, sharing KV heads, compressing state, or changing the attention formulation.

<details>
<summary>Detailed answer</summary>

FlashAttention preserves dense softmax attention while improving its execution. Sliding-window or block-sparse attention reduces attended pairs; MQA/GQA reduce KV cache size; low-rank and kernel-based methods change the approximation or representation. Retrieval can reduce how much context must enter the model. Choose based on the actual bottleneck—prefill compute, decode bandwidth, memory, or quality—and measure the relevant workloads. These techniques are often complementary.

</details>

:::note[Question: What are the attention variants?]{.question-note}
:::

**Brief answer.** Attention variants differ in connectivity, head sharing, position treatment, mathematical formulation, and implementation.

<details>
<summary>Detailed answer</summary>

Self-attention versus cross-attention identifies where queries and keys/values originate. Causal, local, and block-sparse masks define connectivity. MHA, GQA, and MQA define KV sharing. Relative biases and RoPE encode position. Linear attention changes the attention formulation; FlashAttention is an efficient algorithm for supported attention operations. These categories overlap: a causal GQA layer with rotary embeddings can run using a FlashAttention kernel.

</details>

:::note[Question: What issues exist in multi-head attention?]{.question-note}
:::

**Brief answer.** Multi-head attention can require a large KV cache, repeated memory reads, and substantial compute; some heads can also learn redundant patterns.

<details>
<summary>Detailed answer</summary>

Separate KV projections for every query head increase stored state during autoregressive decoding. The cost depends on head dimensions, sequence length, batch size, and layer count. GQA/MQA reduce KV heads, while optimized kernels address memory traffic. Redundant heads are an empirical finding for some models, not proof that arbitrary heads can be removed without retraining or quality loss.

</details>

:::note[Question: Explain Multi-Query Attention (MQA).]{.question-note}
:::

**Brief answer.** Multi-query attention gives many query heads one shared key head and one shared value head.

<details>
<summary>Detailed answer</summary>

Each query head still computes its own attention weights, so shared K/V does not mean identical attention outputs. With $H$ query heads, KV cache storage can fall by roughly $H$ relative to MHA at the same head dimension. The Q projection and output projection remain, and score computation still serves every query head. This trades some representational flexibility for lower memory traffic. See the [MQA paper](https://arxiv.org/abs/1911.02150).

</details>

:::note[Question: Compare Multi-head, Multi-Query, and Grouped-Query Attention.]{.question-note}
:::

**Brief answer.** MHA has one KV head per query head; MQA has one KV head total; GQA shares each KV head across a group of query heads.

<details>
<summary>Detailed answer</summary>

For $H$ query heads and $G$ KV heads:

<div class="article-table" role="region" aria-label="Attention head sharing comparison" tabindex="0">

| Variant | KV heads | Relative KV storage |
| --- | --- | --- |
| MHA | $H$ | $1$ |
| GQA | $1<G<H$ | $G/H$ |
| MQA | $1$ | $1/H$ |

</div>

The ratios assume equal layers, batch size, context length, head dimension, and dtype. GQA offers a tunable compromise; quality and speed depend on the checkpoint and implementation. See [GQA](https://arxiv.org/abs/2305.13245).

</details>

:::note[Question: What are the benefits of MQA?]{.question-note}
:::

**Brief answer.** MQA reduces KV cache memory and decoding memory traffic, often enabling larger batches or faster token generation.

<details>
<summary>Detailed answer</summary>

Its largest benefit is commonly in memory-bandwidth-limited decoding. It does not reduce all model FLOPs by the query-head count, eliminate attention's sequence-length dependence, or guarantee equal quality. If the FFN or another subsystem dominates latency, end-to-end gains can be smaller. Compare tokens per second, time per output token, peak memory, and task quality at matched workloads.

</details>

:::note[Question: Which models use MQA or GQA?]{.question-note}
:::

**Brief answer.** PaLM uses MQA; Llama 2 70B and Mistral 7B use GQA. Llama 2 7B and 13B use MHA.

<details>
<summary>Detailed answer</summary>

These version-specific examples show why a family name alone is insufficient. Check the ratio between query-head and KV-head counts in the checkpoint configuration; equal counts indicate MHA, one KV head indicates MQA, and an intermediate count indicates GQA. Some models use other compressed attention designs that should not be classified solely by this rule. See [PaLM](https://arxiv.org/abs/2204.02311), [Llama 2](https://arxiv.org/abs/2307.09288), and [Mistral 7B](https://arxiv.org/abs/2310.06825).

</details>

:::note[Question: Why was FlashAttention introduced? Briefly explain its core idea.]{.question-note}
:::

**Brief answer.** FlashAttention was introduced to reduce the expensive memory transfers of dense attention without approximating its softmax computation.

<details>
<summary>Detailed answer</summary>

It tiles Q, K, and V into on-chip memory and uses an online softmax calculation to accumulate outputs without storing the full score/probability matrix in GPU high-bandwidth memory. Backward computation recomputes selected intermediates. Dense attention still has quadratic arithmetic in sequence length; the improvement is IO efficiency and intermediate memory use, with normal floating-point differences. See [FlashAttention](https://arxiv.org/abs/2205.14135).

</details>

:::note[Question: What are FlashAttention advantages?]{.question-note}
:::

**Brief answer.** FlashAttention lowers attention-intermediate memory use and can speed training and prefill for supported hardware and shapes.

<details>
<summary>Detailed answer</summary>

Avoiding the full attention matrix can make longer sequences feasible. Gains depend on sequence length, dtype, head dimension, GPU, masks, and kernel version; short workloads may gain little. It does not remove the autoregressive KV cache or make dense attention linear-time. Distinguish the original algorithm from later implementations when quoting benchmarks. See the [FlashAttention implementation](https://github.com/Dao-AILab/flash-attention).

</details>

:::note[Question: Which models implement FlashAttention?]{.question-note}
:::

**Brief answer.** FlashAttention is an execution backend, not a fixed model-family property; compatible implementations of models such as Llama or Mistral can use it.

<details>
<summary>Detailed answer</summary>

Actual dispatch depends on the framework version, hardware, dtype, mask, and requested features. A PyTorch scaled-dot-product-attention call may choose a fused kernel or fall back to another implementation. Requesting attention weights can also affect the path. Verify the configured backend and profiler output rather than inferring it from the checkpoint name. See [Transformers attention backends](https://huggingface.co/docs/transformers/attention_interface).

</details>

:::note[Question: What is a parallel Transformer block?]{.question-note}
:::

**Brief answer.** A parallel Transformer block computes attention and the FFN from the same incoming residual stream, then adds both results.

<details>
<summary>Detailed answer</summary>

One common schematic is:

$$
y=x+\operatorname{Attention}(\operatorname{Norm}_a(x))+\operatorname{FFN}(\operatorname{Norm}_f(x)).
$$

A sequential block instead feeds the attention-updated state into the FFN. Some parallel designs share a normalization; exact details vary. The parallel structure can improve execution scheduling but changes within-block information flow. It is not the same as tensor or pipeline parallelism across GPUs. PaLM is a documented example in the [PaLM paper](https://arxiv.org/abs/2204.02311).

</details>

:::note[Question: What’s the computational complexity of attention and how can it be improved?]{.question-note}
:::

**Brief answer.** Dense full-sequence attention costs $O(n^2d)$ for score/value products plus $O(nd^2)$ for projections; cached single-token attention costs $O(nd)$ for those products.

<details>
<summary>Detailed answer</summary>

Naive training additionally stores an $n\times n$ matrix per head. FlashAttention avoids that materialization while preserving quadratic dense arithmetic. A window of size $w$ gives approximately $O(nwd)$ attention work over a full sequence. Linear or compressed methods can reduce length scaling but change assumptions or model behavior. State whether an estimate concerns training, prefill, one decode step, or generating an entire sequence.

</details>

:::note[Question: Compare MHA, GQA, and MQA — what are their key differences?]{.question-note}
:::

**Brief answer.** The key serving difference is the number of KV heads: MHA stores the most state, MQA the least, and GQA lies between them.

<details>
<summary>Detailed answer</summary>

For ordinary cached attention, estimate KV bytes as:

$$
M_{\mathrm{KV}}=2BLnH_{\mathrm{KV}}d_hs,
$$

where $B$ is batch size, $L$ layers, $n$ cached tokens, $d_h$ head dimension, and $s$ bytes per element. The factor 2 is for keys and values. Calculated example: with $B=1$, $L=32$, $n=8192$, $d_h=128$, and $s=2$, 32 KV heads use 4 GiB, 8 use 1 GiB, and 1 uses 128 MiB. Allocator overhead and other tensors are excluded.

</details>

</details>

<details>
<summary>Cross-Attention</summary>

:::note[Question: Why do we need Cross-Attention?]{.question-note}
:::

**Brief answer.** Cross-attention lets one sequence selectively read information from a different sequence or modality.

<details>
<summary>Detailed answer</summary>

For translation, decoder states query source-language encoder states. For vision-language systems, text states can query visual features. Concatenating everything into one self-attention stream is an alternative, but cross-attention provides an explicit interface between representations with different lengths or encoders. Its usefulness depends on the architecture and task rather than being mandatory for all multimodal or retrieval systems.

</details>

:::note[Question: Explain Cross-Attention.]{.question-note}
:::

**Brief answer.** Cross-attention computes queries from a target representation and keys/values from a separate source representation.

<details>
<summary>Detailed answer</summary>

For $X\in\mathbb R^{n_q\times d_x}$ and $Y\in\mathbb R^{n_k\times d_y}$:

$$
Q=XW_Q,\quad K=YW_K,\quad V=YW_V,\qquad
O=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

The softmax runs over source positions. $M$ masks unavailable keys, such as source padding. Query and key projections must share dimension $d_k$, but the original sequences can have different feature widths. The result has one output vector per query position.

</details>

:::note[Question: Compare Cross-Attention and Self-Attention — similarities and differences.]{.question-note}
:::

**Brief answer.** Self-attention and cross-attention use the same query-key-value operation but obtain those tensors from different inputs.

<details>
<summary>Detailed answer</summary>

Self-attention uses one sequence for Q, K, and V. Cross-attention uses target states for Q and source states for K/V. Attention scores therefore have shape $n\times n$ for self-attention and $n_q\times n_k$ for cross-attention. Causality is a masking decision, not a defining property of cross-attention; a translation decoder usually reads all unpadded source positions while masking future positions in its separate self-attention layer.

</details>

:::note[Question: Provide a code implementation of Cross-Attention.]{.question-note}
:::

**Brief answer.** Use projected queries from the target and keys/values from the source; PyTorch's `MultiheadAttention` implements the projections and head combination.

<details>
<summary>Detailed answer</summary>

This CPU example uses batch-first tensors and explicitly masks padded source tokens. In this API, `True` in `key_padding_mask` means ignore that key; other attention APIs can use different boolean conventions.

```python
import torch
from torch import nn

torch.manual_seed(0)
attention = nn.MultiheadAttention(32, 4, dropout=0.0, batch_first=True)
attention.eval()
query = torch.randn(2, 3, 32)
source = torch.randn(2, 5, 32)
padding = torch.tensor([
    [False, False, False, True, True],
    [False, False, False, False, True],
])
with torch.no_grad():
    output, weights = attention(
        query, source, source,
        key_padding_mask=padding,
        need_weights=True,
        average_attn_weights=False,
    )
assert output.shape == (2, 3, 32)
assert weights.shape == (2, 4, 3, 5)
assert torch.allclose(weights.sum(-1), torch.ones(2, 4, 3))
assert torch.all(weights.masked_select(padding[:, None, None, :]) == 0)
print(tuple(output.shape))  # (2, 3, 32)
```

Every example needs at least one unmasked source token. Production code often sets `need_weights=False` to avoid materializing attention weights. See [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).

</details>

:::note[Question: What are the application scenarios for cross-attention?]{.question-note}
:::

**Brief answer.** Cross-attention is used in encoder-decoder translation, summarization, speech-to-text, multimodal fusion, and text-conditioned diffusion models.

<details>
<summary>Detailed answer</summary>

In a speech encoder-decoder, output text states query encoded audio. In a text-conditioned diffusion network, visual features can query text embeddings. Some retrieval architectures cross-attend to retrieved representations, but conventional prompt-based RAG simply places retrieved text in a decoder's input. Do not assume every RAG or multimodal implementation includes a dedicated cross-attention layer.

</details>

:::note[Question: What are the advantages and challenges of Cross-Attention?]{.question-note}
:::

**Brief answer.** Cross-attention offers flexible alignment between separate representations, at the cost of additional computation and a dependency on source quality.

<details>
<summary>Detailed answer</summary>

It supports unequal sequence lengths and reusable source K/V projections during decoding. The score/value work scales with $n_qn_kd$, so long sources can be expensive. Poor source features, incorrect masks, or noisy retrieved evidence can hurt outputs. Attention weights show mixing coefficients but are not, by themselves, a faithful explanation of the model's reasoning or evidence use.

</details>

</details>

<details>
<summary>Transformer Operations</summary>

:::note[Question: How to load a BERT model using transformers?]{.question-note}
:::

**Brief answer.** Load a matching tokenizer and BERT encoder with `AutoTokenizer` and `AutoModel`, then tokenize text into tensor inputs.

<details>
<summary>Detailed answer</summary>

This example requires PyTorch, Transformers, and access to the checkpoint on the first run. `AutoModel` returns the encoder without a task-specific classification head; use the appropriate task class when needed.

```python
import torch
from transformers import AutoModel, AutoTokenizer

checkpoint = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint).eval()
inputs = tokenizer(
    ["A short sentence.", "Another example."],
    padding=True, truncation=True, return_tensors="pt",
)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
print(outputs.last_hidden_state.shape)  # [batch, tokens, 768]
```

Pin checkpoint revisions and package versions for reproducible runs. See [BERT in Transformers](https://huggingface.co/docs/transformers/model_doc/bert).

</details>

:::note[Question: How to output a specific hidden_state from BERT using transformers?]{.question-note}
:::

**Brief answer.** Request `output_hidden_states=True`, then index the returned tuple; index 0 is the embedding output and index 1 is the first encoder layer's output.

<details>
<summary>Detailed answer</summary>

Using `model`, `inputs`, and `outputs` from the preceding example:

```python
layer_index = 8
assert 0 <= layer_index <= model.config.num_hidden_layers
layer_vectors = outputs.hidden_states[layer_index]
assert layer_vectors.shape[:2] == inputs["input_ids"].shape
assert len(outputs.hidden_states) == model.config.num_hidden_layers + 1
```

BERT-base has 12 encoder layers, so there are 13 entries including embeddings. Each has shape `[batch, sequence_length, hidden_size]`. Requesting all hidden states increases output memory; use it when needed for analysis or feature extraction.

</details>

:::note[Question: How to get the final or intermediate layer vector outputs of BERT?]{.question-note}
:::

**Brief answer.** Use `last_hidden_state` for final token vectors and `hidden_states[k]` for intermediate token vectors; pooling is a separate choice.

<details>
<summary>Detailed answer</summary>

Continuing the BERT example:

```python
final_tokens = outputs.last_hidden_state
intermediate_tokens = outputs.hidden_states[8]
cls_vector = final_tokens[:, 0, :]
mask = inputs["attention_mask"].unsqueeze(-1).to(final_tokens.dtype)
mean_vector = (final_tokens * mask).sum(1) / mask.sum(1).clamp_min(1)
assert mean_vector.shape == (inputs["input_ids"].shape[0], 768)
```

This mean excludes padding but includes special tokens. `pooler_output` applies a learned transformation to the CLS representation and is not identical to the raw CLS vector. Neither raw CLS nor mean pooling from an untuned BERT is guaranteed to provide strong semantic-search embeddings.

</details>

</details>

<details>
<summary>LLM Loss Functions</summary>

:::note[Question: What is KL divergence?]{.question-note}
:::

**Brief answer.** KL divergence measures how one probability distribution differs from another: $D_{\mathrm{KL}}(p\Vert q)=\sum_xp(x)\log[p(x)/q(x)]$.

<details>
<summary>Detailed answer</summary>

It is nonnegative and equals zero when the distributions agree almost everywhere, but it is asymmetric and is not a distance metric. Terms with $p(x)=0$ contribute zero; if $p(x)>0$ where $q(x)=0$, the divergence is infinite. With natural logs its unit is nats. In distillation, the direction and temperature specify which mismatches receive the greatest penalty.

</details>

:::note[Question: Write the cross-entropy loss and explain its meaning.]{.question-note}
:::

**Brief answer.** Cross-entropy is $H(p,q)=-\sum_kp_k\log q_k$; it penalizes assigning low predicted probability to target outcomes.

<details>
<summary>Detailed answer</summary>

For a one-hot target class $y$, the loss reduces to $-\log q_y$. Language modeling applies this over vocabulary predictions and averages over supervised token positions. For soft targets, such as a teacher distribution or label smoothing, every target probability contributes. Define the reduction and masking explicitly: averaging per example and averaging per token can weight variable-length sequences differently.

</details>

:::note[Question: What’s the difference between KL divergence and cross-entropy?]{.question-note}
:::

**Brief answer.** Cross-entropy includes the target distribution's entropy; KL divergence measures the excess beyond it: $H(p,q)=H(p)+D_{\mathrm{KL}}(p\Vert q)$.

<details>
<summary>Detailed answer</summary>

If $p$ is fixed, minimizing cross-entropy over $q$ is equivalent to minimizing forward KL because $H(p)$ is constant. If the target distribution also depends on optimized parameters, that argument no longer automatically applies. Cross-entropy need not be zero even when $p=q$; KL is zero in that case. Both depend on the chosen direction and log base.

</details>

:::note[Question: How do you handle large loss differences in multi-task learning?]{.question-note}
:::

**Brief answer.** Normalize losses sensibly, then balance task sampling and gradient contributions rather than comparing raw loss magnitudes alone.

<details>
<summary>Detailed answer</summary>

Use a weighted objective $L=\sum_i\lambda_iL_i$ and make each task's reduction explicit. One task may sum over thousands of tokens while another averages over examples, producing misleading scale differences. Inspect gradient norms and per-task validation metrics. Fixed weights are a useful baseline; adaptive methods such as uncertainty weighting or GradNorm can help, but require their own validation. Conflicting gradient directions are a separate issue from unequal scales. See [GradNorm](https://arxiv.org/abs/1711.02257).

</details>

:::note[Question: Why is cross-entropy preferred over MSE for classification tasks?]{.question-note}
:::

**Brief answer.** Cross-entropy matches categorical maximum likelihood and usually gives more effective gradients for softmax classification than MSE on class probabilities.

<details>
<summary>Detailed answer</summary>

For one-hot targets, its gradient with respect to logits is $q-y$. MSE on softmax probabilities includes the softmax Jacobian, which can shrink gradients for confident incorrect predictions. MSE is not mathematically invalid: the Brier score is a proper probabilistic scoring rule. Choose according to the training objective and calibration needs; the common preference for cross-entropy is not a claim that regression losses can never classify.

</details>

:::note[Question: What is information gain?]{.question-note}
:::

**Brief answer.** Information gain is the reduction in uncertainty about a target after observing a feature or split.

<details>
<summary>Detailed answer</summary>

For a decision-tree split $S$ and target $Y$:

$$
\operatorname{IG}(Y;S)=H(Y)-\sum_sP(S=s)H(Y\mid S=s).
$$

A split that separates classes well lowers conditional entropy and produces higher gain. For a balanced binary target, entropy is 1 bit; a perfect split reduces it to zero and gains 1 bit. Estimate probabilities from training data, and constrain tree complexity to avoid fitting accidental purity. Entropy is not the only possible tree criterion; Gini impurity is another.

</details>

:::note[Question: How to compute softmax and cross-entropy loss (and binary cross-entropy)?]{.question-note}
:::

**Brief answer.** Use softmax plus categorical cross-entropy for mutually exclusive classes, and sigmoid plus binary cross-entropy for independent binary labels; compute both directly from logits for stability.

<details>
<summary>Detailed answer</summary>

For logits $z$, softmax is $p_k=e^{z_k}/\sum_je^{z_j}$ and categorical CE is $-\log p_y$. Binary CE is $-[y\log\sigma(z)+(1-y)\log(1-\sigma(z))]$.

```python
import torch
from torch.nn import functional as F

logits = torch.tensor([[2.0, 0.0, -1.0]])
ce = F.cross_entropy(logits, torch.tensor([0]))
bce = F.binary_cross_entropy_with_logits(
    torch.tensor([2.0, -1.0]), torch.tensor([1.0, 0.0])
)
print(round(ce.item(), 4), round(bce.item(), 4))  # 0.1698 0.2201
```

Do not pass already-softmaxed probabilities to `cross_entropy` or already-sigmoided probabilities to `binary_cross_entropy_with_logits`. Multi-label tasks apply binary losses per label.

</details>

:::note[Question: What if the exponential term in softmax overflows the float limit?]{.question-note}
:::

**Brief answer.** Subtract the maximum logit before exponentiating, and compute log probabilities with a stable log-softmax or log-sum-exp operation.

<details>
<summary>Detailed answer</summary>

Let $a=\max_jz_j$. Then:

$$
\log\sum_je^{z_j}=a+\log\sum_je^{z_j-a},\qquad
p_i=\frac{e^{z_i-a}}{\sum_je^{z_j-a}}.
$$

Every shifted exponent is at most 1, preventing overflow for finite logits. This does not repair logits that already contain infinity or NaN; inspect the earlier computation. Very negative terms can underflow harmlessly in the sum, whereas taking `log(softmax(...))` can create unwanted negative infinities. Fused loss functions avoid that unstable path.

</details>

</details>

<details>
<summary>Similarity & Contrastive Learning</summary>

:::note[Question: Besides cosine similarity, what other similarity metrics exist?]{.question-note}
:::

**Brief answer.** Alternatives include dot product, Euclidean or Manhattan distance, Jaccard similarity, edit distance, BM25, and learned pairwise scores.

<details>
<summary>Detailed answer</summary>

Choose according to the representation and task: Jaccard compares sets, edit distance compares strings, and BM25 ranks lexical matches. These are not all mathematical metrics. For unit-normalized vectors, $\lVert u-v\rVert_2^2=2-2u^\top v$, so cosine, dot product, and Euclidean distance induce equivalent rankings. Without normalization, vector magnitude changes the relationship. A cross-encoder can score a query-document pair directly but costs more than comparing precomputed embeddings.

</details>

:::note[Question: What is contrastive learning?]{.question-note}
:::

**Brief answer.** Contrastive learning trains representations to score related examples above unrelated examples.

<details>
<summary>Detailed answer</summary>

A common query-positive objective is:

$$
L=-\log\frac{\exp(s(q,d^+)/\tau)}{\exp(s(q,d^+)/\tau)+\sum_{d^-}\exp(s(q,d^-)/\tau)}.
$$

$s$ is a similarity function and $\tau>0$ a temperature. Positives and negatives may come from labels, paired modalities, or augmentations. The pairing policy defines what “similar” means; poorly chosen positives teach the wrong invariances. Some self-supervised approaches avoid explicit negatives using different objectives, so not all representation learning follows this formula. See [SimCLR](https://arxiv.org/abs/2002.05709).

</details>

:::note[Question: How important are negative samples in contrastive learning, and how to handle costly negative sampling?]{.question-note}
:::

**Brief answer.** Useful negatives teach the distinctions a retriever must make; in-batch negatives, cached embeddings, and mined candidates reduce collection cost.

<details>
<summary>Detailed answer</summary>

Random negatives are cheap but can become too easy. Hard negatives resemble the query yet do not answer it, producing a more informative comparison. Filter false negatives, because another relevant document should not be pushed away merely because it lacks an annotation. Mix easy and hard examples, refresh stale mined candidates, and evaluate on held-out queries. Larger batches increase negative coverage but also memory use and false-negative risk.

</details>

</details>

## Advanced Topics in LLMs

<details>
<summary>Advanced LLM</summary>

:::note[Question: What is a generative large model?]{.question-note}
:::

**Brief answer.** A generative large model learns a distribution that can produce new content conditioned on an input or context.

<details>
<summary>Detailed answer</summary>

Autoregressive LLMs generate text by repeatedly sampling or selecting the next token from $p(x_t\mid x_{<t})$. Other generative models use mechanisms such as diffusion and can produce images, audio, or video. “Large” has no universal parameter threshold, and generative capability does not imply factual correctness, reasoning reliability, or access to current information.

</details>

:::note[Question: How do LLMs make generated text diverse and non-repetitive?]{.question-note}
:::

**Brief answer.** Sampling temperature, top-k or top-p filtering, and repetition controls change output diversity, while training quality and prompting influence coherence.

<details>
<summary>Detailed answer</summary>

Temperature rescales logits before sampling: lower values concentrate probability and higher values flatten it. Top-p retains a high-probability set whose cumulative mass reaches a threshold. Repetition penalties or n-gram constraints can break loops but can also damage names, code, or necessary repeated terms. Greater randomness is not guaranteed to improve usefulness. Tune decoding against task-specific quality and repetition metrics. See [Transformers generation settings](https://huggingface.co/docs/transformers/main_classes/text_generation).

</details>

:::note[Question: What is the repetition problem (LLM echo problem)? Why does it happen? How can it be mitigated?]{.question-note}
:::

**Brief answer.** The repetition problem is unwanted looping or excessive reuse of phrases, caused by interactions among the learned distribution, prompt, and decoding procedure.

<details>
<summary>Detailed answer</summary>

Greedy or poorly tuned decoding can repeatedly enter high-probability continuations. Repeated training data, malformed chat templates, missing stop tokens, or prompt echoing can contribute. Inspect whether repetition appears in the model output itself or in the application's streaming assembly. Try appropriate stop handling, corrected formatting, better data, and measured sampling or penalties. Raising temperature blindly can replace repetition with incoherence.

</details>

:::note[Question: Can Llama handle infinitely long inputs? Why or why not?]{.question-note}
:::

**Brief answer.** No. Llama checkpoints have finite supported context lengths, limited memory and compute, and finite training experience with positional ranges.

<details>
<summary>Detailed answer</summary>

Rotary position formulas can be evaluated beyond the training range, but that does not guarantee accurate long-range behavior. With a conventional KV cache, memory grows with stored context; full dense prefill also has quadratic attention work. Sliding windows can bound stored attention context but discard direct access to older tokens. Retrieval, external memory, or recurrent processing can handle longer streams without making a single forward pass have infinite context.

</details>

:::note[Question: When should you use BERT vs. LLaMA / ChatGLM models?]{.question-note}
:::

**Brief answer.** Use BERT-style encoders for efficient representation or classification tasks, and generative Llama/ChatGLM-style models when the output requires open-ended text or interactive instruction following.

<details>
<summary>Detailed answer</summary>

Examples for encoders include tagging, reranking, and classification with a trained task head. A contrastively trained encoder can serve dense retrieval. Generative models suit explanation, conversation, and flexible structured generation, but need output validation. Benchmark the exact checkpoint and pipeline; decoder models can also classify or embed, and a generic BERT checkpoint may not be the best specialist encoder.

</details>

:::note[Question: Do different domains require their own domain-specific LLMs? Why?]{.question-note}
:::

**Brief answer.** No. Start with a capable general model and evaluate whether prompting, retrieval, or task adaptation resolves the domain gap.

<details>
<summary>Detailed answer</summary>

RAG is useful for changing or private facts. SFT helps with terminology, response style, procedures, and task behavior. Continued pretraining may help when the model lacks broad domain-language competence and sufficient quality data exists. A dedicated model adds training, evaluation, deployment, and maintenance costs. Choose based on observed errors, privacy requirements, and serving constraints rather than the existence of a domain label.

</details>

:::note[Question: How to enable an LLM to process longer texts?]{.question-note}
:::

**Brief answer.** Combine a suitable long-context model with efficient attention, careful positional adaptation, or retrieval and hierarchical processing, depending on the task.

<details>
<summary>Detailed answer</summary>

Extending a model's native window may require positional changes and long-sequence continued training. FlashAttention and KV compression address resource use but do not teach long-context reasoning by themselves. Retrieval selects evidence; hierarchical summaries compress it but can lose details. Test questions requiring evidence from different locations and multiple documents, not only a single “needle” lookup. More accepted tokens and better use of those tokens are distinct capabilities.

</details>

</details>

## Fine-Tuning Large Models

<details>
<summary>General Fine-Tuning</summary>

:::note[Question: What can cause a sudden loss drop in the second epoch of SFT?]{.question-note}
:::

**Brief answer.** There is no universal second-epoch loss drop. Investigate repeated examples, learning-rate changes, batch composition, and logging or masking bugs before attributing it to learning.

<details>
<summary>Detailed answer</summary>

Plot loss against optimizer steps and supervised token counts, not only epoch summaries. Check scheduler boundaries, sampler resets, sequence lengths, packing, and whether an average was reset. A model may predict previously seen examples better in epoch two, but a dramatic discontinuity can also reflect a changed reduction or label mask. Compare held-out loss and task quality to distinguish useful learning from memorization.

</details>

:::note[Question: How much VRAM is needed for full fine-tuning?]{.question-note}
:::

**Brief answer.** Full fine-tuning memory includes weights, gradients, optimizer states, activations, and runtime buffers; parameter count alone is insufficient.

<details>
<summary>Detailed answer</summary>

A common mixed-precision Adam estimate is 16 bytes per parameter: 2 for BF16 weights, 2 for BF16 gradients, 4 for an FP32 master copy, and 8 for two FP32 moments. Calculated example: 7 billion parameters need about 112 GB, or 104 GiB, for those states alone. Frameworks that store gradients or master weights differently have different totals.

Add activation memory, temporary kernels, communication buffers, and allocator headroom. Activation checkpointing, FSDP/ZeRO sharding, and offload change the per-GPU requirement; GPU capacities cannot simply be summed without a suitable parallel strategy. Profile the intended sequence length and microbatch. See [Transformers training memory anatomy](https://huggingface.co/docs/transformers/v4.57.1/model_memory_anatomy).

</details>

:::note[Question: Why can a model become less capable after SFT?]{.question-note}
:::

**Brief answer.** SFT can degrade apparent capability through overfitting, forgetting, low-quality targets, overly aggressive updates, or incorrect chat formatting.

<details>
<summary>Detailed answer</summary>

Check the inference template, stop tokens, tokenizer, label masking, and generation settings before blaming the objective. Compare base and tuned checkpoints on both the target task and general regression tests. Repetitive or erroneous demonstrations can teach shortcuts; excessive steps or learning rates can overwrite useful behavior. Improve data and mixtures, reduce update strength, and choose checkpoints by held-out utility rather than training loss alone.

</details>

:::note[Question: How to construct instruction fine-tuning datasets?]{.question-note}
:::

**Brief answer.** Build instruction-response examples from real target tasks, with correct outputs, representative coverage, and leakage-safe splits.

<details>
<summary>Detailed answer</summary>

Define the expected behavior and annotation rubric first. Include realistic context, multi-turn interactions, difficult cases, tool results where appropriate, and examples where evidence is insufficient. Deduplicate by source or intent before splitting; paraphrases of the same example should not straddle training and evaluation. Record provenance and applicable data permissions. Synthetic examples need verification and should fill identified coverage gaps rather than merely inflate row counts.

</details>

:::note[Question: How to improve prompt representativeness?]{.question-note}
:::

**Brief answer.** Make prompts representative by sampling the actual deployment distribution and deliberately covering important rare cases.

<details>
<summary>Detailed answer</summary>

Build a coverage map across tasks, languages, user expertise, context lengths, ambiguity, and output formats. Use appropriately handled real usage examples when available, then add targeted cases where coverage is weak. Preserve natural imperfections such as incomplete requests instead of polishing every prompt into an ideal benchmark question. Evaluate both traffic-weighted performance and critical slices so common easy requests do not conceal consequential failures.

</details>

:::note[Question: How to increase prompt data volume?]{.question-note}
:::

**Brief answer.** Increase useful data through new source tasks, expert annotation, verified synthetic generation, and carefully chosen transformations.

<details>
<summary>Detailed answer</summary>

Paraphrases can expand wording diversity, but near-duplicates add little new supervision. Generate from independent source material or target uncovered skills, and verify answers with executable checks, source evidence, or expert review. Track source identity and keep derived examples in the same split. Measure marginal gains as data grows; a larger set of correlated or incorrect outputs can perform worse than a smaller clean dataset.

</details>

:::note[Question: How to select domain data for continued pretraining?]{.question-note}
:::

**Brief answer.** Choose domain-pretraining text that is relevant, accurate, diverse, and suitable for the intended use, then validate the mixture against downstream tasks.

<details>
<summary>Detailed answer</summary>

Inspect provenance, freshness, duplication, language, formatting, and contamination of held-out evaluations. Preserve useful structures such as code, equations, and document boundaries. Domain relevance alone does not establish quality: repeated boilerplate or unreliable claims can dominate training. Mix general text when necessary to retain broader capabilities, and run small ablations before committing a full compute budget.

</details>

:::note[Question: How to prevent forgetting general abilities after domain tuning?]{.question-note}
:::

**Brief answer.** Reduce forgetting with balanced data mixtures, conservative updates, and repeated evaluation of retained capabilities.

<details>
<summary>Detailed answer</summary>

Replay suitable general or instruction-following examples alongside domain data. Tune learning rate, training duration, and trainable parameter scope; regularization or distillation can constrain harmful drift. PEFT preserves the original checkpoint for switching back, but an enabled adapter can still degrade general behavior. Track both domain gains and regressions in language, reasoning, formatting, and safety rather than relying on one domain score.

</details>

:::note[Question: How to make the model learn more knowledge during pretraining?]{.question-note}
:::

**Brief answer.** Improve knowledge learning through reliable coverage, diverse contexts, appropriate model capacity, and a training budget that actually lets the model absorb the data.

<details>
<summary>Detailed answer</summary>

Deduplicate excessive repetition without removing every useful recurrence, balance rare topics, and keep tokenization and optimization healthy. Repeated facts in varied contexts can support learning, but adding more text does not guarantee reliable factual recall. Evaluate held-out questions and paraphrases, not just corpus likelihood. For facts that change frequently or require exact attribution, retrieval may be more maintainable than storing everything in parameters.

</details>

:::note[Question: When performing SFT, should the base model be Chat or Base?]{.question-note}
:::

**Brief answer.** Use an instruction-tuned checkpoint when you want to preserve existing assistant behavior with limited data; use a base checkpoint when you need greater control and can supply sufficient post-training data.

<details>
<summary>Detailed answer</summary>

Neither is universally best. Further tuning a chat model can be data-efficient, but can also interfere with existing capabilities or formatting. A base model generally needs broader supervision to become a reliable assistant. Compare license, context support, target-task quality, and regression behavior using a small pilot, and preserve the checkpoint's tokenizer and chat template.

</details>

:::note[Question: What’s the input/output format for domain fine-tuning?]{.question-note}
:::

**Brief answer.** The format follows the objective: continued pretraining uses text sequences, while SFT uses prompt-response or role-based conversation examples.

<details>
<summary>Detailed answer</summary>

A conceptual conversation record is:

```json
{
  "messages": [
    {"role": "user", "content": "Summarize the supplied maintenance report."},
    {"role": "assistant", "content": "The inspection found ..."}
  ]
}
```

The trainer applies the model's chat template and constructs input tokens, attention masks, and labels. Decide explicitly which tokens receive loss; assistant-only supervision is common but not automatic for every trainer and template. Preserve tool-call structure and end-of-turn tokens when present. See [TRL's SFT data formats and masking](https://huggingface.co/docs/trl/sft_trainer).

</details>

:::note[Question: How to build a domain evaluation set?]{.question-note}
:::

**Brief answer.** Build a domain evaluation set from independently held-out sources and realistic tasks, with trustworthy references and clear scoring rules.

<details>
<summary>Detailed answer</summary>

Split by document, entity, customer, or time where needed to avoid overlap. Include common requests, difficult reasoning, ambiguous inputs, unanswerable cases, and required refusals. Use executable or exact checks where possible and expert rubrics where judgment is necessary. Report uncertainty, slice results, latency, and cost. Keep development and final test sets separate, and version the dataset as domain requirements change.

</details>

:::note[Question: Is vocabulary expansion necessary? Why?]{.question-note}
:::

**Brief answer.** Vocabulary expansion is optional; consider it only when the existing tokenizer fragments important domain text inefficiently and the benefit justifies compatibility costs.

<details>
<summary>Detailed answer</summary>

Measure token counts and task quality first. New tokens require resizing and training embedding rows and, where applicable, output-head rows; tied weights must remain consistent. The tokenizer must be deployed with every dependent model and adapter. Existing subword or byte fallback can often represent the text already, so poor terminology handling is not automatically a vocabulary problem. Expansion can invalidate caches and complicate serving.

</details>

:::note[Question: How to train your own LLM?]{.question-note}
:::

**Brief answer.** Training an LLM involves defining evaluation targets, preparing data and a tokenizer, pretraining, post-training, and validating a deployable checkpoint.

<details>
<summary>Detailed answer</summary>

A practical sequence is to establish baselines, estimate compute and memory, validate a tiny training run, scale gradually, and monitor held-out loss and task metrics. Pretraining learns broad predictive structure; SFT and preference-based stages adapt behavior. Save model, optimizer, scheduler, random state, tokenizer, and data-position metadata for recovery. Training from scratch is only justified when existing checkpoints cannot meet the requirements and the data and compute budget support it.

</details>

:::note[Question: What are the benefits of instruction fine-tuning?]{.question-note}
:::

**Brief answer.** Instruction fine-tuning teaches a pretrained model how to respond to requests in the desired format, style, and task setting.

<details>
<summary>Detailed answer</summary>

It can improve instruction following, multi-turn behavior, tool-call structure, and use of supplied context. It can also teach task-specific information, but success depends on demonstration quality and coverage. SFT imitates target responses; it does not inherently verify their truth or optimize every downstream preference. Evaluate generalization to new instructions and check for regressions outside the tuning distribution.

</details>

:::note[Question: During which stage — pretraining or fine-tuning — is knowledge injected?]{.question-note}
:::

**Brief answer.** Both pretraining and fine-tuning can update factual knowledge in model parameters, while retrieval supplies information at inference without updating those parameters.

<details>
<summary>Detailed answer</summary>

Pretraining usually provides the broadest exposure to facts and language. Continued pretraining and SFT can add or modify knowledge, though reliable recall and compositional use are not guaranteed. A model answering from a retrieved document is using contextual evidence, not necessarily permanently learning it. Choose among these mechanisms based on how often information changes, how it must be attributed, and how errors will be corrected.

</details>

</details>

<details>
<summary>SFT Tricks</summary>

:::note[Question: What’s the typical SFT workflow?]{.question-note}
:::

**Brief answer.** A typical SFT workflow is baseline evaluation, dataset preparation, format and loss checks, pilot training, full training, checkpoint selection, and deployment validation.

<details>
<summary>Detailed answer</summary>

Render several tokenized examples and inspect exactly which targets contribute to loss. Run a small-batch sanity check, then tune learning rate, effective batch size, duration, and PEFT settings if used. Track validation metrics and general-capability regressions. Select on held-out task utility; preserve the tokenizer, template, training configuration, and data version. Recheck the final exported or quantized artifact because deployment transformations can alter behavior.

</details>

:::note[Question: What are key aspects of training data?]{.question-note}
:::

**Brief answer.** Training data must have correct targets, representative coverage, useful diversity, consistent formatting, and reliable provenance.

<details>
<summary>Detailed answer</summary>

Quantity is only one dimension. Duplicates change example weighting, long answers can dominate token-level objectives, and contaminated benchmarks produce misleading progress. Track task mixtures, lengths, label quality, and source overlap. Ensure each example teaches the intended behavior: a well-written answer that ignores the supplied context can train the wrong habit for a grounded assistant.

</details>

:::note[Question: How to choose between large and small models?]{.question-note}
:::

**Brief answer.** Choose the smallest model that meets measured quality and operational requirements, and compare total serving cost rather than parameter count alone.

<details>
<summary>Detailed answer</summary>

Larger models can handle more complex tasks but typically demand more memory and compute. A smaller specialist, retrieval pipeline, or routed system may meet the target more efficiently. Evaluate realistic input/output lengths, concurrency, hardware, and failure rates. Include retries, tool calls, and review costs: a cheap model that needs repeated correction may be more expensive per successful task.

</details>

:::note[Question: How to ensure multi-task training balance?]{.question-note}
:::

**Brief answer.** Balance multi-task training through both the probability of sampling each task and the weight of its loss.

<details>
<summary>Detailed answer</summary>

A large dataset can dominate updates even with equal nominal loss weights. Token-averaged training can also emphasize tasks with longer answers. Adjust sampling ratios, normalization, and weights against per-task validation metrics, and inspect gradient conflicts when one task improves at another's expense. Use a representative mixture or explicit curriculum, then verify that rare but important tasks retain enough training exposure.

</details>

:::note[Question: Can SFT learn knowledge at all?]{.question-note}
:::

**Brief answer.** Yes. SFT changes parameters and can teach new facts, but low training loss does not prove reliable knowledge acquisition.

<details>
<summary>Detailed answer</summary>

A model may memorize the wording of demonstrations while failing on paraphrases, novel combinations, or conflicting evidence. Test on questions that isolate newly introduced information and prevent source leakage. For broad domain exposure, continued pretraining may be appropriate; for changing facts, RAG may be preferable. The slogan “SFT only teaches format” is too absolute, as is the claim that a few examples reliably install arbitrary knowledge.

</details>

:::note[Question: How to select datasets effectively?]{.question-note}
:::

**Brief answer.** Select datasets according to the target task and observed failure modes, then test whether each source adds measurable value.

<details>
<summary>Detailed answer</summary>

Audit relevance, answer quality, licensing, duplication, contamination, and coverage. Sample and inspect actual records rather than trusting dataset names or leaderboard popularity. Start with a clean baseline mixture and add sources through controlled ablations. Keep selection decisions separate from the final test set, and monitor whether gains on one slice introduce regressions elsewhere.

</details>

</details>

<details>
<summary>Training Experience</summary>

:::note[Question: How to choose a distributed training framework?]{.question-note}
:::

**Brief answer.** Choose a distributed training approach from the model's memory needs, cluster interconnect, required parallelism, and operational support.

<details>
<summary>Detailed answer</summary>

Data parallelism replicates computation across batches; FSDP and ZeRO shard training states; tensor and pipeline parallelism split computation; expert parallelism distributes MoE experts. PyTorch FSDP, DeepSpeed, and Megatron-based stacks serve different combinations of those needs. Check checkpoint portability, restart behavior, kernel support, and integration with the existing codebase, then benchmark a representative run. See [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html), [ZeRO](https://www.deepspeed.ai/tutorials/zero/), and [Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html).

</details>

:::note[Question: What are key LLM training tips?]{.question-note}
:::

**Brief answer.** Validate correctness at small scale, monitor optimization and data quality, and make recovery reproducible before spending a large training budget.

<details>
<summary>Detailed answer</summary>

Check token shifting, masks, padding, chat templates, loss normalization, and packed-sequence boundaries. Track gradient norms, numerical overflow, throughput, data mixtures, and held-out quality. Verify a checkpoint can resume with optimizer and data state intact. Use appropriate mixed precision, clipping, and activation checkpointing. Change one major factor at a time so improvements and regressions remain attributable.

</details>

:::note[Question: How to choose model size?]{.question-note}
:::

**Brief answer.** Choose model size jointly with available data, training compute, and deployment requirements.

<details>
<summary>Detailed answer</summary>

For dense Transformer pretraining, $C\approx6ND$ is a rough FLOP estimate using parameter count $N$ and training tokens $D$; it omits important details such as long-context attention and exact architecture. Compute-optimal training and inference-efficient deployment can favor different choices. A smaller model trained longer may be useful when serving costs dominate. Treat scaling-law results as experimental guidance under their assumptions, not universal token-per-parameter rules. See [Chinchilla](https://arxiv.org/abs/2203.15556).

</details>

:::note[Question: How to select GPU accelerators?]{.question-note}
:::

**Brief answer.** Select accelerators by usable memory, sustained workload throughput, memory bandwidth, interconnect, supported precision, and total cost.

<details>
<summary>Detailed answer</summary>

Fit the intended sequence length and microbatch with headroom, then measure tokens per second using the actual model and kernels. Multi-GPU training depends heavily on communication, so peak FLOPs alone can be misleading. Include software compatibility, availability, reliability, power, and checkpoint storage. A faster GPU can reduce total cost if it shortens the run enough; a cheaper one can win when utilization or communication limits performance.

</details>

</details>

## LangChain and Agent-Based Systems

<details>
<summary>LangChain Core</summary>

:::note[Question: What is LangChain?]{.question-note}
:::

**Brief answer.** LangChain is a framework and integration ecosystem for building applications and agents around language models.

<details>
<summary>Detailed answer</summary>

It provides abstractions for models, messages, tools, prompts, retrieval, and structured outputs. Current LangChain agents use LangGraph for execution and persistence. The framework coordinates application logic; it does not make the underlying model more accurate automatically. Use it where its integrations and orchestration simplify the application, and check the documentation matching the installed version. See the [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview).

</details>

:::note[Question: What are its core concepts?]{.question-note}
:::

**Brief answer.** Core concepts include models, messages, prompts, tools, retrieval, structured outputs, and execution state; workflows can be deterministic or model-directed.

<details>
<summary>Detailed answer</summary>

A deterministic workflow specifies the next operation in code. An agent lets the model choose among permitted actions within a controlled loop. LangGraph supplies explicit stateful orchestration, while LangSmith supports tracing and evaluation. Older tutorials also organize concepts around “chains,” “indexes,” and “memory”; retain the concepts without assuming their historical classes remain the preferred APIs. See [LangChain concepts](https://docs.langchain.com/oss/python/langchain/overview).

</details>

:::note[Question: What are components and chains in LangChain?]{.question-note}
:::

**Brief answer.** Components perform individual operations; chains compose operations into a defined flow.

<details>
<summary>Detailed answer</summary>

For example, a workflow can format a prompt, call a model, and parse its response. Runnable-style interfaces support composition and invocation across compatible components. A chain can be conditional or parallel, but its control flow is still specified by the application rather than necessarily chosen by an agent. Handle retries and side effects explicitly: retrying a model call is different from repeating a tool that changes external state.

</details>

:::note[Question: What are prompt templates and prompt values?]{.question-note}
:::

**Brief answer.** A prompt template fills named variables into a reusable text or message structure; a prompt value is the formatted result passed to a model.

<details>
<summary>Detailed answer</summary>

A chat template can preserve system, user, and assistant roles instead of concatenating everything into one string. Validate required variables and keep user content separate from application instructions. Formatting does not enforce instruction priority or prevent prompt injection. Budget tokens after formatting, especially when inserting history and retrieved documents. The tokenizer's model-specific chat template is a separate serialization layer from an application prompt template.

</details>

:::note[Question: What are example selectors, and when are they useful?]{.question-note}
:::

**Brief answer.** An example selector chooses which few-shot demonstrations to include for a particular request.

<details>
<summary>Detailed answer</summary>

Selectors may use semantic similarity, token budget, diversity, or task metadata. Similar examples can improve relevance, but picking only near-duplicates can narrow behavior or leak answers from an evaluation set. Select demonstrations from an approved training pool, reserve room for the actual request and evidence, and evaluate whether the chosen examples improve outputs compared with a fixed small set.

</details>

:::note[Question: What do output parsers validate, and what do they not validate?]{.question-note}
:::

**Brief answer.** Output parsers convert model responses into application types and report format errors; they do not establish that the content is true.

<details>
<summary>Detailed answer</summary>

A parser might validate JSON fields, types, or an enum. Prefer a model/provider's supported structured-output mechanism when suitable, then validate the returned object and domain constraints. Decide whether malformed output should trigger a bounded retry, repair, or user-visible failure. A syntactically valid amount or citation can still be wrong. See [LangChain structured output](https://docs.langchain.com/oss/python/langchain/structured-output).

</details>

:::note[Question: How do indexes and retrievers differ?]{.question-note}
:::

**Brief answer.** An index organizes data for search; a retriever accepts a query and returns relevant documents or records.

<details>
<summary>Detailed answer</summary>

A vector store can back a retriever, but retrieval can also use BM25, a database, an API, or several sources together. Preserve stable IDs, source metadata, and authorization filters. Index construction and query-time retrieval are separate lifecycle stages: changing embedding models usually requires re-embedding compatible documents. A retriever's interface does not guarantee useful recall or proper access control; those require implementation and evaluation.

</details>

:::note[Question: How should chat message history be stored and used?]{.question-note}
:::

**Brief answer.** Chat message history stores conversation turns, while the context for a model call is the selected portion of that history and other relevant state.

<details>
<summary>Detailed answer</summary>

Store role, content, and tool-call relationships under a correctly scoped conversation identifier. Persistence can outlive a process, but storing messages does not mean every message should be sent on every call. Apply token budgeting, retention rules, and authorization. Current LangGraph-backed agents can checkpoint thread state; cross-conversation memory needs a separately designed storage and retrieval policy. See [short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory).

</details>

:::note[Question: What are agents and toolkits?]{.question-note}
:::

**Brief answer.** An agent uses a model to select actions from available tools; a toolkit groups related tools and their interfaces.

<details>
<summary>Detailed answer</summary>

A tool exposes a schema and executable behavior, while the agent loop interprets model requests, validates arguments, executes permitted calls, and returns results. Enforce permissions in code, not only in the prompt. Add timeouts, bounded iterations, error handling, and approval points for actions that require them. Use deterministic workflows when the action sequence is known; an agent loop adds flexibility and additional failure modes. See [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents).

</details>

</details>

<details>
<summary>Long-Term Memory in Multi-Turn Conversations</summary>

:::note[Question: How can Agents access conversation context?]{.question-note}
:::

**Brief answer.** Agents access conversation context through explicit state or retrieved memory that the application supplies to each model call.

<details>
<summary>Detailed answer</summary>

Combine recent messages, relevant past facts, tool results, and application state within a token budget. Scope storage and retrieval to the correct user and conversation; an embedding similarity score is not an authorization decision. Distinguish short-term thread state from long-term information shared across sessions. Store provenance and support updates or deletion, because old preferences and summaries can become stale. See [LangGraph memory](https://docs.langchain.com/oss/python/langgraph/add-memory).

</details>

:::note[Question: When should an agent retrieve the full conversation history?]{.question-note}
:::

**Brief answer.** Loading the full conversation is simple and preserves detail, but only works when the useful history fits the model's context and cost budget.

<details>
<summary>Detailed answer</summary>

Retrieve messages in order, preserve system instructions and tool-call/result pairs, and reserve space for new evidence and the answer. Long transcripts increase prefill cost and can distract the model. Never concatenate unrelated users' histories. Full-history storage can still be useful for auditing even when model calls use a smaller, selected context.

</details>

:::note[Question: How does a sliding window manage recent conversation context?]{.question-note}
:::

**Brief answer.** A sliding window retains recent conversation turns to bound context use, but can drop important older facts.

<details>
<summary>Detailed answer</summary>

Trim by token budget rather than an arbitrary number of messages, and keep complete tool interactions. Preserve essential instructions and durable task state separately; consider a summary or retrieval for older information. Summaries are lossy and should retain links to original messages when precision matters. Evaluate delayed references such as “use the constraint I gave earlier” to detect failures caused by truncation.

</details>

</details>

<details>
<summary>Practical RAG Q&A using LangChain</summary>

:::note[Question: How would you implement and validate a document Q&A application using LangChain?]{.question-note}
:::

**Brief answer.** Build a minimal RAG application with explicit ingestion, retrieval, prompt construction, generation, and evaluation before adding an agent loop.

<details>
<summary>Detailed answer</summary>

In LangChain, connect document loaders and splitters to an embedding/index backend, expose retrieval, then pass selected source text into a model prompt. Keep ingestion separate from query handling and preserve source IDs throughout. The following is conceptual pseudocode, not a version-specific runnable API example:

```text
index = ingest(parse(documents), chunker, embedding_model)
candidates = retrieve(index, question, user_permissions)
evidence = rerank_and_pack(question, candidates, token_budget)
answer = generate(question, evidence, require_source_ids=True)
return validate_citations_and_support(answer, evidence)
```

Test empty retrieval, deleted documents, conflicting sources, unauthorized evidence, and citations that do not support the answer. Framework composition alone does not enforce these properties. See [LangChain's retrieval guide](https://docs.langchain.com/oss/python/deepagents/retrieval).

</details>

</details>

## Retrieval-Augmented Generation (RAG)

<details>
<summary>RAG Basics</summary>

:::note[Question: Why do LLMs need an external (vector) knowledge base?]{.question-note}
:::

**Brief answer.** External knowledge sources provide current, private, or precisely attributable information that a model's parameters may not reliably contain.

<details>
<summary>Detailed answer</summary>

A vector database is one option, not a requirement. Lexical search, SQL, document stores, graph queries, and APIs can supply better evidence for some tasks. Retrieval can update information without retraining the generator, but introduces ingestion, authorization, ranking, and evidence-quality requirements. An LLM does not need an external knowledge base for every task, such as rewriting a user-provided paragraph.

</details>

:::note[Question: What’s the overall workflow of LLM+VectorDB document chat?]{.question-note}
:::

**Brief answer.** Document chat typically ingests and indexes documents offline, then retrieves and assembles evidence before generating a cited answer online.

<details>
<summary>Detailed answer</summary>

Offline: parse content, preserve structure and source metadata, chunk, embed, and index. Online: interpret the question, apply access filters, retrieve candidates, rerank, expand relevant context, and pack a bounded prompt. Generate an answer with source references, then check citation validity and support. Log retrieval and generation separately so an incorrect answer can be traced to the actual failing stage.

</details>

:::note[Question: What are the core technologies in a RAG system?]{.question-note}
:::

**Brief answer.** Core RAG technologies include document parsing, chunking, embeddings or lexical indexing, retrieval, reranking, context assembly, generation, and evaluation.

<details>
<summary>Detailed answer</summary>

The important interfaces are often more consequential than any individual library: source IDs must survive parsing, embedding queries must match the index, filters must respect permissions, and citations must resolve to evidence. A useful baseline may combine BM25 with dense retrieval and a reranker. Measure each stage before adding query rewriting, graph traversal, or complex agent orchestration.

</details>

:::note[Question: How to build an effective prompt template?]{.question-note}
:::

**Brief answer.** An effective RAG prompt states the task, separates instructions from evidence, specifies source attribution, and explains what to do when evidence is insufficient or conflicting.

<details>
<summary>Detailed answer</summary>

For example, supply the question and numbered excerpts, require support for factual claims, and ask the model to identify unresolved conflicts. Keep source IDs stable and include dates or titles when they affect interpretation. Avoid filling the context with redundant text. Retrieved content is untrusted data: prompt wording helps, but authorization, tool permissions, and output checks must also exist outside the model.

</details>

</details>

<details>
<summary>RAG Concepts</summary>

:::note[Question: What are the limitations of base LLMs that RAG solves?]{.question-note}
:::

**Brief answer.** RAG can reduce gaps in freshness, access to private documents, and source attribution, but does not eliminate hallucination or reasoning errors.

<details>
<summary>Detailed answer</summary>

The generator can only use evidence that exists, is accessible, is retrieved, and fits its context. Stale indexes, missing documents, misleading passages, or poor synthesis can still produce wrong answers. Retrieval is most useful when errors stem from unavailable information; it may not fix weak arithmetic, poor instruction following, or bad task definitions. Compare results with and without retrieval on representative questions.

</details>

:::note[Question: What is RAG?]{.question-note}
:::

**Brief answer.** Retrieval-augmented generation retrieves external information and conditions a generative model on it when answering a request.

<details>
<summary>Detailed answer</summary>

A retriever finds potentially relevant evidence, and a generator uses that evidence with the question. Contemporary applications often retrieve once and place excerpts in a prompt; others retrieve iteratively or use trained retrieval-generation architectures. The original RAG paper combined parametric generation with a non-parametric document index. Retrieval does not necessarily update model weights. See [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).

</details>

:::note[Question: How to obtain accurate semantic representations?]{.question-note}
:::

**Brief answer.** Obtain useful semantic representations by choosing an embedding model trained for the retrieval task and encoding well-formed, appropriately sized content.

<details>
<summary>Detailed answer</summary>

Preserve titles, relevant structure, and language information; avoid splitting an answer from its qualifier or embedding OCR noise. Follow the model's query/document prefixes, normalization, and length limits. Evaluate held-out retrieval before considering domain contrastive fine-tuning. An embedding can represent semantic relatedness without capturing answer relevance, exact identifiers, dates, or negation reliably, so lexical and metadata signals may still be necessary.

</details>

:::note[Question: How to align query/document semantic spaces?]{.question-note}
:::

**Brief answer.** Use query and document encoders trained to share a compatible retrieval space, with matching preprocessing and similarity conventions.

<details>
<summary>Detailed answer</summary>

Asymmetric retrieval can use distinct encoders or different query/document prompts; compatibility does not require identical text transformations. Match embedding version, dimensions, normalization, and metric. Training on relevant query-positive-negative triples can improve alignment. Never mix document vectors from one embedding model with arbitrary query vectors from another merely because their dimensions match; migrate or rebuild the index coherently.

</details>

:::note[Question: How to match retrieval model output with LLM preferences?]{.question-note}
:::

**Brief answer.** Select evidence that supports the answer the generator must produce, rather than optimizing semantic similarity alone.

<details>
<summary>Detailed answer</summary>

Train or evaluate the retriever with answer-supporting relevance labels, and rerank query-passage pairs for answerability. Preserve necessary definitions, qualifiers, and surrounding context when packing evidence. End-to-end feedback can improve selection, but a generator's preference for a passage is not proof that it is accurate. Keep retrieval relevance, factual correctness, and answer usefulness as distinct evaluation targets.

</details>

:::note[Question: How to improve results via post-retrieval processing?]{.question-note}
:::

**Brief answer.** Post-retrieval processing improves evidence through deduplication, reranking, context expansion, and budget-aware selection.

<details>
<summary>Detailed answer</summary>

Remove near-identical chunks, rerank against the specific question, and restore neighboring sentences or parent sections when needed. Enforce authorization before passing candidates to a model or external reranker. Diversify evidence for multi-part questions and preserve source IDs through compression. A compressor can omit a crucial exception, so compare answers against the original evidence and measure changes in recall as well as prompt size.

</details>

:::note[Question: How to optimize generator adaptation to inputs?]{.question-note}
:::

**Brief answer.** Adapt the generator to use supplied evidence through clear prompting, relevant demonstrations, or grounded SFT, and evaluate its behavior under imperfect retrieval.

<details>
<summary>Detailed answer</summary>

Train or test on sufficient evidence, missing evidence, distractors, contradictions, and unanswerable questions. Require citations where appropriate and teach calibrated abstention rather than guessing. Avoid datasets where the model can ignore retrieval and answer from memorized shortcuts. Generator adaptation cannot repair an absent source reliably; use error analysis to decide whether the bottleneck is evidence availability or evidence use.

</details>

:::note[Question: What are the benefits of using RAG?]{.question-note}
:::

**Brief answer.** RAG can improve freshness, domain coverage, source traceability, and updateability without retraining the generator for every document change.

<details>
<summary>Detailed answer</summary>

It also enables access-controlled information use and can reduce unsupported answers when retrieval is good. Costs include ingestion, indexing, retrieval latency, context tokens, and additional failure modes. Benefits are workload-dependent: irrelevant retrieval can worsen an answer. Measure grounded task success and total operating cost, not just whether the architecture contains a vector database.

</details>

</details>

<details>
<summary>RAG Layout Analysis</summary>

:::note[Question: Why is PDF parsing necessary?]{.question-note}
:::

**Brief answer.** PDF parsing converts a page-oriented format into usable text and structure for indexing, retrieval, and citation.

<details>
<summary>Detailed answer</summary>

A PDF can store positioned glyphs rather than paragraphs, contain scanned images, or mix both. Extraction must recover reading order, headings, tables, and page references. Scanned text requires OCR or an image-capable parser. Without this step, a retriever may index disconnected columns or omit the very facts needed to answer a question. Retain page and region provenance so extracted claims can be checked against the original.

</details>

:::note[Question: What are the common PDF parsing methods and their differences?]{.question-note}
:::

**Brief answer.** PDF processing methods include native text extraction, OCR, layout-aware parsing, and multimodal model-based extraction.

<details>
<summary>Detailed answer</summary>

Native extraction is often fast for digitally generated PDFs but can lose reading order. OCR recognizes text from page images but can confuse characters. Layout models locate headings, tables, and regions; multimodal models can interpret complex pages but may introduce fabricated content or inconsistent structure. Hybrid pipelines select methods by page type. Evaluate against representative documents and retain visual provenance for uncertain results.

</details>

:::note[Question: What problems exist in PDF parsing?]{.question-note}
:::

**Brief answer.** Common PDF parsing problems include broken reading order, missing text, OCR errors, split tables, repeated headers, and lost structural relationships.

<details>
<summary>Detailed answer</summary>

Two-column pages can interleave sentences; ligatures and embedded fonts can corrupt characters; scans can erase decimal points. Captions may detach from figures, and tables can lose row/column associations. Use page-level quality checks, compare suspicious regions with images, and exclude repeated boilerplate where appropriate. Parsing errors propagate into embeddings and answers, so fixing them upstream often matters more than switching generators.

</details>

:::note[Question: Why is table recognition important?]{.question-note}
:::

**Brief answer.** Table recognition preserves relationships among values, row labels, column headers, units, and footnotes.

<details>
<summary>Detailed answer</summary>

A value such as “12.4” is meaningless without its metric, time period, and units. Flattening a table into arbitrary text can associate it with the wrong row or column. Store a structured representation and a readable serialization with source coordinates. For exact aggregates or arithmetic, retrieve the relevant table and use validated computation instead of relying solely on a language model's interpretation.

</details>

:::note[Question: What are the main methods for recognizing and extracting tables?]{.question-note}
:::

**Brief answer.** Table extraction usually combines table detection, cell/structure recognition, text recognition or extraction, and conversion to a structured representation.

<details>
<summary>Detailed answer</summary>

Rule-based methods use lines, whitespace, and text alignment. Learned methods detect table regions and infer rows, columns, and spans. OCR supplies text for image-based tables, while native text coordinates can serve born-digital documents. Multimodal extraction is another option, but still needs validation. Complex merged cells, repeated headers, and multi-page tables require explicit handling beyond merely detecting a bounding box.

</details>

:::note[Question: How do traditional table-extraction methods work?]{.question-note}
:::

**Brief answer.** Traditional table extraction uses ruling lines, intersections, whitespace, and alignment heuristics to infer cells and their text.

<details>
<summary>Detailed answer</summary>

Line-based methods work well for clearly bordered tables; text-based methods infer columns and rows from aligned words in borderless layouts. They can be fast and inspectable, but depend on document conventions and tuned tolerances. Merged cells, irregular spacing, skew, and tables spanning pages break simple assumptions. Use visual debugging and structural checks rather than trusting an extraction merely because it returned rows.

</details>

:::note[Question: How can pdfplumber extract text and tables?]{.question-note}
:::

**Brief answer.** `pdfplumber` extracts native PDF text and tables using character geometry, lines, and configurable table-detection strategies; it is not an OCR engine.

<details>
<summary>Detailed answer</summary>

For a born-digital PDF, inspect `extract_words()` and `extract_text()`, crop a known table region if necessary, and use `find_tables()` or `extract_tables()`. Choose line-based, text-based, or explicit boundaries through table settings. Visual debugging with `debug_tablefinder()` helps tune intersection and alignment tolerances. Validate merged cells and repeated headers against the page image. Scanned pages need an OCR stage or another parser. See the [pdfplumber documentation](https://github.com/jsvine/pdfplumber).

</details>

:::note[Question: Why do we need text chunking?]{.question-note}
:::

**Brief answer.** Chunking creates retrievable units that fit embedding and generation budgets while preserving enough context to interpret the evidence.

<details>
<summary>Detailed answer</summary>

Whole documents can dilute relevant information or exceed model limits; tiny chunks can lose definitions and exceptions. Choose boundaries from document structure and expected questions. Preserve parent IDs, headings, and offsets so a small matched chunk can be expanded later. Overlap reduces boundary loss but increases storage and duplicates. Tune chunking with retrieval and answer-quality tests rather than applying a universal token count.

</details>

:::note[Question: What chunking strategies can you implement with regex, spaCy, or LangChain?]{.question-note}
:::

**Brief answer.** Chunking strategies include fixed token windows, sentence or paragraph boundaries, recursive separators, document structure, semantic segmentation, and parent-child retrieval.

<details>
<summary>Detailed answer</summary>

Regex can encode known boundaries; spaCy can identify sentence boundaries; LangChain splitters package token-, character-, or structure-based strategies. These are tools, not inherently different semantic objectives. A character limit is not a token limit. Preserve code blocks, tables, and headings when their structure matters, and test whether overlap or parent expansion improves evidence completeness. See [LangChain text splitters](https://docs.langchain.com/oss/python/integrations/splitters).

</details>

</details>

<details>
<summary>RAG Retrieval Strategies</summary>

:::note[Question: Why use LLMs to assist recall?]{.question-note}
:::

**Brief answer.** LLMs can improve recall by rewriting ambiguous queries, generating alternative phrasings, or decomposing a complex question into searchable subquestions.

<details>
<summary>Detailed answer</summary>

This helps bridge differences between a user's wording and the source corpus. Keep the original query among the candidates and preserve constraints such as entity names, dates, and negation. Rewrites can drift toward an incorrect interpretation, increasing noise and latency. Measure additional relevant evidence retrieved and downstream answer quality, not simply the number of search calls.

</details>

:::note[Question: How does HyDE work, and what are its limitations?]{.question-note}
:::

**Brief answer.** HyDE generates a hypothetical relevant document, embeds it, and uses that vector to retrieve real documents.

<details>
<summary>Detailed answer</summary>

The generated text supplies richer document-like semantics than a short query. It is a retrieval aid, not evidence: the final answer should rely on actual retrieved sources. A wrong hypothetical passage can steer retrieval away from the needed evidence, especially for rare terms or entities, and generation adds latency. Compare with direct and hybrid retrieval on the target corpus. See [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496).

</details>

:::note[Question: How does FLARE use active retrieval during generation?]{.question-note}
:::

**Brief answer.** FLARE retrieves during generation when an anticipated next sentence contains low-confidence tokens, then regenerates that sentence using retrieved evidence.

<details>
<summary>Detailed answer</summary>

The method predicts upcoming content, uses that content to form a retrieval query, and triggers retrieval based on token-confidence signals rather than only at the start. Query construction must avoid overcommitting to uncertain invented details. Reliable token probabilities may be unavailable through some APIs, and high confidence does not prove factual correctness. It adds retrieval and generation work, so evaluate the quality-latency tradeoff. See [Active Retrieval Augmented Generation](https://arxiv.org/abs/2305.06983).

</details>

:::note[Question: Why construct hard negative samples?]{.question-note}
:::

**Brief answer.** Hard negatives train a retriever to distinguish genuinely answer-supporting passages from plausible but irrelevant ones.

<details>
<summary>Detailed answer</summary>

For a question about a product's refund deadline, a passage about delivery deadlines may be a useful hard negative. Mine candidates from a current retriever or reranker, then verify that they are actually negative. Unlabeled relevant documents are common false negatives. Mix difficulty levels and monitor held-out recall so training does not specialize only to one miner's mistakes.

</details>

:::note[Question: How do random negatives compare with top-k hard negatives?]{.question-note}
:::

**Brief answer.** Random negatives are cheap and diverse but often easy; top-k mined negatives are more confusable but contain more false negatives and selection bias.

<details>
<summary>Detailed answer</summary>

Retrieve high-scoring candidates for each training query, remove known positives, and filter or annotate the remaining candidates before treating them as negative. Start with a mixture and refresh mining as the retriever changes. The highest-scoring candidate is not automatically a valid negative. Compare gains on independent queries and track whether the mining procedure overrepresents particular document types.

</details>

</details>

<details>
<summary>RAG Evaluation</summary>

:::note[Question: Why evaluate RAG?]{.question-note}
:::

**Brief answer.** Evaluate RAG to determine whether failures come from missing sources, retrieval, context assembly, generation, or the overall application.

<details>
<summary>Detailed answer</summary>

A good answer metric alone cannot explain why a system failed. Measure evidence coverage and rank before generation, then answer correctness, support, citation quality, and abstention. Include latency, cost, permission enforcement, and freshness. Use realistic questions and source snapshots so results are reproducible and changes to ingestion or indexing can be regression-tested.

</details>

:::note[Question: What are the evaluation methods, metrics, and frameworks?]{.question-note}
:::

**Brief answer.** Evaluate retrieval and generation separately, then measure end-to-end task success with references, human judgments, and carefully calibrated automated judges.

<details>
<summary>Detailed answer</summary>

Retrieval metrics include recall@k, precision@k, MRR, and nDCG when relevance labels support them. Generation metrics include answer correctness, faithfulness to evidence, citation precision/coverage, and appropriate abstention. A faithful answer can still be factually wrong if its source is wrong.

Ragas provides evaluation metrics, and LangSmith supports dataset-based evaluation and tracing. LLM judges need rubrics, versioning, and calibration against humans; reference-free scores are not ground truth. Include multi-hop, unanswerable, outdated, and adversarial-source cases. See [Ragas metrics](https://docs.ragas.io/en/stable/concepts/metrics/) and [LangSmith evaluation](https://docs.langchain.com/langsmith/evaluation).

</details>

</details>

<details>
<summary>RAG Optimization</summary>

:::note[Question: What are the optimization strategies for retrieval and generation modules?]{.question-note}
:::

**Brief answer.** Optimize the stage responsible for the measured error: source coverage and retrieval, evidence packing, or grounded generation.

<details>
<summary>Detailed answer</summary>

For retrieval, improve parsing, chunk boundaries, hybrid search, filters, embeddings, and candidate recall before reranking. For generation, preserve sufficient context, reduce distractors, use clear evidence instructions, and test grounded tuning when needed. Cache only with suitable freshness and access scoping. Change one stage at a time and track quality, latency, and cost on the same evaluation set; adding components everywhere obscures causality.

</details>

:::note[Question: How to enhance context using knowledge graphs (KGs)?]{.question-note}
:::

**Brief answer.** Knowledge graphs augment context with explicit entities and relationships that can connect evidence across documents.

<details>
<summary>Detailed answer</summary>

Resolve the question's entities, traverse relevant typed relationships, and retrieve the source passages supporting those edges. A product-dependency question can follow service-to-library-to-vulnerability links more directly than a single vector lookup. Retain dates, confidence, and provenance for every relationship. An extracted graph can contain errors or omissions, so treat its paths as evidence candidates and verify claims against sources.

</details>

:::note[Question: What are the problems with vector-based context augmentation?]{.question-note}
:::

**Brief answer.** Vector retrieval can miss exact identifiers, relational constraints, multi-hop connections, and facts distributed across chunks.

<details>
<summary>Detailed answer</summary>

Semantic similarity does not guarantee answer support. A document about the right topic can concern the wrong date, entity, or exception. Dense indexes also require coordinated updates when embeddings or sources change. Combine lexical search, metadata filters, structured queries, graph traversal, or parent expansion where evaluation shows a gap. These complementary methods have their own costs and should not be added without a demonstrated need.

</details>

:::note[Question: How can KG-based methods improve it?]{.question-note}
:::

**Brief answer.** Graph-based retrieval can follow explicit relationships and constraints to find evidence that a standalone similarity search misses.

<details>
<summary>Detailed answer</summary>

For example, first identify which team owns a service, then retrieve that team's current escalation policy. This depends on accurate entity resolution, edge typing, and temporal validity. Combine graph candidates with text retrieval and reranking, and pass supporting passages to the generator. Graph construction can lose information or introduce unsupported links, so it is not a universal replacement for vector or lexical search.

</details>

:::note[Question: What are the main pain points in RAG and their solutions?]{.question-note}
:::

**Brief answer.** The main RAG failure points are missing or corrupted sources, poor candidate retrieval, bad ranking or context assembly, and unsupported generation.

<details>
<summary>Detailed answer</summary>

Diagnose them in order: confirm the answer exists in the authorized corpus, inspect parsed text, inspect retrieved candidates, inspect packed context, and finally inspect the generated claims. A missing source needs ingestion work; a lost qualifier needs context repair; a model that ignores available evidence needs generation changes. Maintain examples for each failure class so a fix can be tested at the responsible stage.

</details>

:::note[Question: How do you diagnose missing content in a RAG system?]{.question-note}
:::

**Brief answer.** If required content is absent from the corpus or was lost during ingestion, retrieval tuning alone cannot recover it.

<details>
<summary>Detailed answer</summary>

Check source availability, ingestion logs, parser output, OCR, chunk exclusions, permissions, and index freshness. Compare a known source passage with its indexed representation. Repair or reingest the missing content, then run a targeted retrieval check. If no authorized evidence exists, ask for the source or return an explicit insufficient-evidence response rather than synthesizing a plausible answer.

</details>

:::note[Question: What should you check when relevant documents fail to reach the final context?]{.question-note}
:::

**Brief answer.** When relevant documents do not reach the final context, inspect candidate recall, filters, rank cutoffs, reranking, and context packing separately.

<details>
<summary>Detailed answer</summary>

If the document never appears among candidates, improve query interpretation or retrieval coverage. If it appears but is ranked poorly, assess reranking signals and relevance labels. If it ranks highly but disappears from the prompt, inspect deduplication and token-budget logic. Increasing k can improve coverage but also add noise and cost, so measure both retrieval and final answer changes.

</details>

:::note[Question: How do you prevent context loss during retrieval and compression?]{.question-note}
:::

**Brief answer.** Context loss happens when retrieval or compression separates a claim from information needed to interpret it, such as a heading, qualifier, or table header.

<details>
<summary>Detailed answer</summary>

Retrieve smaller units for matching, then expand to neighboring sentences or a parent section with stable source metadata. Preserve relationships among table cells and retain exceptions during summarization. Inspect the exact packed prompt rather than only the original document. More context is helpful only when it restores missing meaning without overwhelming the generator with irrelevant passages.

</details>

:::note[Question: What should you do when the generator fails to extract an answer from available evidence?]{.question-note}
:::

**Brief answer.** If the answer is present in the supplied evidence but the model fails to extract it, test evidence presentation, question clarity, and generator capability.

<details>
<summary>Detailed answer</summary>

Remove distractors, restore structure, identify the relevant section, or request constrained extraction with source spans. Use deterministic parsing or code for exact fields and arithmetic when appropriate. Compare a stronger model or grounded SFT only after confirming retrieval is sufficient. Validate the result against the source; a fluent answer with a citation can still misread a negation or numerical unit.

</details>

:::note[Question: How does RAG-Fusion work, and what are its benefits and tradeoffs?]{.question-note}
:::

**Brief answer.** RAG-Fusion generates multiple search queries, retrieves a ranked list for each, and fuses the lists before generation, often with reciprocal rank fusion (RRF).

<details>
<summary>Detailed answer</summary>

For document $d$:

$$
\operatorname{RRF}(d)=\sum_{r:d\in r}\frac{1}{c+\operatorname{rank}_r(d)},
$$

where ranks start at 1 and $c>0$ controls rank sensitivity. Keep stable document IDs, merge repeated results, and usually retain the original query. A document ranking 1st and 3rd in two lists with $c=60$ scores $1/61+1/63\approx0.03227$. RRF combines ranks without needing comparable raw scores; the score is not a relevance probability.

Multiple formulations can improve coverage, but add cost and query-drift risk. RRF alone is a general fusion method and does not require an LLM. See [RAG-Fusion](https://arxiv.org/abs/2402.03367).

</details>

</details>

<details>
<summary>Graph RAG</summary>

:::note[Question: Why do we need Graph RAG?]{.question-note}
:::

**Brief answer.** Graph RAG is useful when answering requires connected entities, multi-hop evidence, or corpus-level themes that independent chunk retrieval may miss.

<details>
<summary>Detailed answer</summary>

A dependency-impact question naturally involves paths, while a broad question such as “What recurring problems appear across these reports?” may benefit from community summaries. Simple fact lookup may work better with cheaper lexical or vector retrieval. Graph extraction, resolution, and maintenance add cost, so compare against a strong baseline on questions that actually require relationships or global synthesis.

</details>

:::note[Question: What is Graph RAG and how does it work? Show a code example and use case.]{.question-note}
:::

**Brief answer.** Graph RAG retrieves graph-connected evidence for generation; Microsoft's GraphRAG additionally builds entity relationships and community summaries for local and global search.

<details>
<summary>Detailed answer</summary>

A typical pipeline extracts entities and relationships with source links, resolves duplicates, builds graph structure, retrieves relevant subgraphs or summaries, and generates an answer grounded in supporting text. Microsoft's particular approach is one implementation of the broader idea. See the [GraphRAG documentation](https://microsoft.github.io/graphrag/).

This runnable Python example uses fictional, already-verified edges to demonstrate two-hop evidence retrieval. It does not perform extraction or call an LLM:

```python
edges = [
    ("Checkout", "depends_on", "Payments", "doc-1"),
    ("Payments", "owned_by", "Platform team", "doc-2"),
    ("Search", "owned_by", "Discovery team", "doc-3"),
]
sources = {
    "doc-1": "Checkout depends on Payments.",
    "doc-2": "The Platform team owns Payments.",
    "doc-3": "The Discovery team owns Search.",
}

def dependency_owners(service):
    results = []
    for src, relation, dependency, source_a in edges:
        if src != service or relation != "depends_on":
            continue
        for node, relation_b, owner, source_b in edges:
            if node == dependency and relation_b == "owned_by":
                results.append({
                    "dependency": dependency,
                    "owner": owner,
                    "evidence": [(sid, sources[sid])
                                 for sid in (source_a, source_b)],
                })
    return results

result = dependency_owners("Checkout")
assert result[0]["owner"] == "Platform team"
assert dependency_owners("Unknown") == []
print(result[0]["owner"])  # Platform team
```

An application can pass this evidence to a generator to answer “Who owns Checkout's dependency?” Production traversal must enforce authorization, entity disambiguation, edge freshness, and bounded expansion. Absence of a path means the graph lacks evidence, not that no relationship exists.

</details>

:::note[Question: How to improve ranking optimization in Graph RAG?]{.question-note}
:::

**Brief answer.** Improve Graph RAG ranking by combining entity relevance, relationship relevance, path quality, source support, and ordinary text-retrieval signals.

<details>
<summary>Detailed answer</summary>

Restrict traversal to useful relation types and hop counts, reduce domination by generic high-degree nodes, and account for dates and extraction confidence. Rerank candidate paths and their supporting passages against the question, then diversify within a token budget. Evaluate answer-supporting evidence rather than only graph connectivity. A short path or popular entity is not automatically the most relevant source.

</details>

</details>

## Parameter-Efficient Fine-Tuning (PEFT)

<details>
<summary>PEFT Fundamentals</summary>

:::note[Question: What is fine-tuning, and how is it performed?]{.question-note}
:::

**Brief answer.** Fine-tuning continues training from a pretrained checkpoint on a new objective or dataset to adapt its behavior or capabilities.

<details>
<summary>Detailed answer</summary>

Prepare and split the data, preserve tokenizer and input conventions, choose trainable parameters, and optimize with conservative settings. Full fine-tuning updates all selected model weights; parameter-efficient methods train a smaller subset or added modules. Evaluate task gains and retained capabilities, and export the exact tokenizer, configuration, and weight or adapter artifacts required for inference.

</details>

:::note[Question: Why do we need PEFT?]{.question-note}
:::

**Brief answer.** PEFT reduces the trainable parameter and optimizer-state footprint of adapting a large model, making task-specific tuning and checkpoint storage more manageable.

<details>
<summary>Detailed answer</summary>

Keeping the backbone frozen avoids gradients and optimizer moments for those weights, though forward computation and many activations still consume resources. Small adapters also let several tasks share a base checkpoint. PEFT does not make the base model disappear from memory or guarantee full-tuning quality. QLoRA further reduces backbone storage through quantization; it is a particular method, not a property of every PEFT technique. See [QLoRA](https://arxiv.org/abs/2305.14314).

</details>

:::note[Question: What is PEFT and its advantages?]{.question-note}
:::

**Brief answer.** Parameter-efficient fine-tuning trains a limited subset of parameters or small added modules while freezing most pretrained weights.

<details>
<summary>Detailed answer</summary>

Examples include bottleneck adapters, LoRA, and learned prompts or prefixes. LoRA represents a matrix update as $\Delta W=BA$ with rank $r$, using $r(d_{\mathrm{in}}+d_{\mathrm{out}})$ parameters rather than $d_{\mathrm{in}}d_{\mathrm{out}}$. Benefits include smaller task artifacts and lower optimizer memory. Quality, activation memory, and inference overhead depend on the method; LoRA updates can often be merged into a compatible dense weight matrix. See [LoRA](https://arxiv.org/abs/2106.09685).

</details>

</details>

<details>
<summary>Adapter Tuning</summary>

:::note[Question: Why use adapter-tuning?]{.question-note}
:::

**Brief answer.** Bottleneck adapter tuning adds small trainable modules to a frozen backbone, allowing multiple tasks to reuse the same pretrained model.

<details>
<summary>Detailed answer</summary>

Each task stores its own adapter parameters instead of a full copy of all model weights. This can reduce training-state memory and simplify task switching, especially when data is limited. Extra modules still require computation and can add latency. Choose adapter placement and width using held-out task quality and serving constraints. See [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751).

</details>

:::note[Question: What’s the core idea behind adapter-tuning?]{.question-note}
:::

**Brief answer.** A classic adapter projects hidden states down to a small bottleneck, applies a nonlinearity, projects back up, and adds a residual connection.

<details>
<summary>Detailed answer</summary>

A schematic form is:

$$
h'=h+W_{\mathrm{up}}\phi(W_{\mathrm{down}}h+b_{\mathrm{down}})+b_{\mathrm{up}},
$$

where the bottleneck width $r$ is much smaller than model width $d$. The projection weights contribute about $2dr$ parameters. Placement and normalization vary by method. This nonlinear bottleneck module differs from LoRA's low-rank linear weight update, even though both are often loosely called “adapters.” See the [adapter paper](https://arxiv.org/abs/1902.00751).

</details>

:::note[Question: How does it differ from full fine-tuning?]{.question-note}
:::

**Brief answer.** Full fine-tuning updates the backbone's weights, while classic adapter tuning freezes the backbone and trains added task-specific modules.

<details>
<summary>Detailed answer</summary>

Full tuning offers broader parameter freedom but usually needs more gradient and optimizer memory and a full task checkpoint. Adapter tuning stores a smaller task artifact but still loads the base model and may add inference computation. Neither guarantees better generalization. Classic nonlinear adapters generally cannot be merged into an existing linear weight the way a standard LoRA update can; deployment behavior depends on the method.

</details>

</details>
