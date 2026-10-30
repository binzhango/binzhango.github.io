# LLM Interview Questions: technical review

Reviewed the complete working-copy article at `/posts/2025/10-30-llm-interview/` on September 19, 2026, including all 35 populated answers, 132 unanswered questions, and two empty question placeholders. The original publication date remains October 30, 2025.

## Delivered revision

- All 167 substantive questions now have a brief answer and an expandable detailed answer.
- Retained the existing topic groups and original section anchors; added section navigation for the later major topics.
- Removed two entries whose entire question was “Question.” No missing question was invented for either placeholder.
- Converted vague labels such as “Content missing,” “Traditional methods,” and the practical LangChain placeholder into explicit questions within their existing topic.
- Added 61 distinct primary-paper, official-documentation, or official-project links next to relevant claims.
- Added a restricted dropout derivation, explicit loss and activation formulas, an attention comparison table, calculated memory examples, and code for cross-attention, BERT representations, stable losses, and graph evidence retrieval.
- Preserved the working-copy correction to the FFN projection dimension. Unrelated existing repository changes were not edited.

## Assessment of the original populated answers

The question numbers below refer to the original source order, before removing the two empty placeholders. “Expanded” means the original core statement was useful but needed assumptions, examples, or a fuller explanation.

| Original questions | Assessment and correction |
| --- | --- |
| 1–7: basic ML concepts | Expanded definitions, concrete application examples, and unsupervised evaluation caveats. Hyperparameters can be selected automatically or scheduled; out-of-core processing must cover the complete data pipeline. |
| 8: training versus unseen performance | Qualified: overfitting is common, but distribution shift and pipeline mismatch can produce the same symptom. |
| 9–12: evaluation splits | Corrected the unconditional claim that a test set provides an unbiased real-world estimate. Added independence, representativeness, split design, adaptive selection, and train-dev diagnostic limitations. |
| 13: bias and variance | Removed the direct identification of hallucination with overfitting and simplistic parameter/epoch rules. Distinguished the squared-error decomposition from language-model cross-entropy. |
| 14: regularization | Corrected the conflation of penalties with outlier-robust data losses; restricted uniqueness claims to applicable convex settings. Answered elastic net directly. Removed unsupported deployment, speed, and merging claims about L1. Distinguished AdamW from an L2 loss penalty. |
| 15: dropout | Replaced the false general exact-equivalence premise with a normalized geometric-mean derivation for linear softmax logits. Distinguished Bernoulli masks, geometric means, arithmetic ensembles, nonlinear networks, and inverted dropout. |
| 16: dimensionality and positions | Corrected the premise: positional information addresses ordering, not general dimensionality reduction. Qualified permutation equivariance with the unmasked setting and removed guaranteed extrapolation claims. |
| 17–18: MLE and NLL | Expanded the chain-rule objective and supplied the actual per-token loss, supervised-token mean, label shift, and mask semantics. |
| 19: metrics | Removed unsupported perplexity/BLEU conclusions from the cat/dog example. Corrected the implication that encoder-decoder models lack autoregressive likelihood. Added compatible-tokenization and evaluation-condition requirements. |
| 20: label smoothing | Supplied a derivation and separated uniform-over-all-classes from uniform-over-incorrect-classes conventions. Qualified claims of universal use or improvement. |
| 21: hard and soft attention | Expanded discrete selection versus continuous weighting; a causal or sparse mask alone does not make attention hard. |
| 22: model families | Distinguished released weights from complete openness; replaced an implied current exhaustive list with representative families and checkpoint-specific verification. |
| 23–27: architectures and objectives | Corrected causal generation's access to the prompt, filled the missing encoder-decoder explanation, distinguished prefix LM from prefix tuning, and removed universal speed/long-context superiority claims. Expanded objectives beyond next-token prediction. |
| 28: normalization | Added actual LayerNorm/RMSNorm formulas, BatchNorm reduction and inference behavior, and implementation-dependent tradeoffs. |
| 29–30: DeepNorm | Corrected square-root scaling to architecture-specific constants, including fourth-root scaling for decoder-only stacks and the paired initialization. Restricted performance claims to the reported experiments. |
| 31–32: normalization placement and examples | Corrected pre/post-norm descriptions and blanket claims about depth, residual growth, modern usage, and RMSNorm adoption. Added version-specific counterexamples. |
| 33: FFN | Preserved the corrected matrix dimension and distinguished ordinary from gated FFNs. |
| 34–35: GELU and Swish | Retained the correct core formulas, added GELU's common approximation, and distinguished generalized Swish from the SiLU special case. |

## New answers and scope limits

The previously empty sections now address attention implementation and complexity, cross-attention, BERT operations, information-theoretic losses, contrastive learning, generation, SFT, training infrastructure, LangChain, conversation state, RAG ingestion/retrieval/evaluation, graph retrieval, and PEFT.

The revision distinguishes diagnostic hypotheses from established mechanisms. A second-epoch SFT loss drop has no universal cause. RAG does not guarantee truth or citation support. SFT can learn facts, PEFT can still alter behavior adversely, and sparse parameters do not automatically accelerate dense kernels.

Model examples name particular releases instead of asserting family-wide behavior. The LangChain implementation outline is explicitly pseudocode. The graph example uses fictional, preverified edges and demonstrates evidence retrieval only; it is not a complete Microsoft GraphRAG implementation. No training, serving-performance, or model-quality benchmark was run for this editorial review.

## Verification

- Source and rendered DOM: 167 question blocks, 167 brief answers, and 167 detailed-answer disclosures; balanced disclosure tags and no empty answer blocks.
- Astro production build completed successfully, including the existing card integration check. The build emitted a duplicate content-ID warning for this route; there is only one matching source file, and the generated route contains the revised article.
- `git diff --check` passes for the changed article.
- All six Python snippets parsed and executed successfully with Python 3.12.9, PyTorch 2.14.0, and Transformers 5.17.0. Cross-attention output shapes, normalized weights, and zero attention to masked keys passed. CE/BCE produced 0.1698 and 0.2201. The graph example returned the expected owner and an empty result for an unknown entity.
- The three sequential BERT snippets ran against a locally saved random-weight fixture with 12 layers and 768 hidden dimensions (reduced vocabulary and FFN width). This exercised `from_pretrained`, tokenization, hidden-state indexing, and padding-aware pooling without downloading the public pretrained weights; the remote checkpoint download and pretrained semantic quality were not tested.
- Browser screenshots reviewed at 1365×900 and 390×844 across light and dark themes, including disclosed explanations, equations, the comparison table, and code. No page-level horizontal overflow was found in checked states; wide formulas/tables scroll inside their containers. Keyboard disclosure expansion worked.
- Rendered DOM checks confirmed each question is followed by its brief answer and detailed disclosure, zero empty detailed answers, zero MathJax error nodes, and no missing in-page anchor targets. Visual checks sampled representative content; they were not 167 individual screenshot reviews.
- Independently verified the calculated 4 GiB / 1 GiB / 128 MiB KV-cache examples, the 104.3 GiB Adam-state example, and the RRF score of approximately 0.03227.
- Enumerated all masks of a small softmax model to verify the dropout geometric-ensemble identity and its difference from the arithmetic mean.
