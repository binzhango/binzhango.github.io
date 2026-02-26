---
title: Batch Normalization

date: 
  created: 2020-02-04 08:15:15

categories: 
  - ML
---

Batch Normalization is one of important parts in our NN.

## Why need Normalization
This paper title tells me the reason
[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- accelerating traning
- reduce internal covariate shift

<!-- more -->
#### Independent and identically distributed (IID)
If our data is independent and identically distributed, training model can be simplified and its predictive ability is improved.
One important step of data preparation is **whitening** which is used to
###### Whitening
- reduce features' coralation     => Independent
- all features have zero mean and unit variances => Identically distributed


#### Internal Covariate Shift (ICS)
What is problem of ICS? Generally data is not IID
- Previous layer should update hyper-parameters to adjust new data so that reduce learning speed
- Get stuck in the saturation region as the network grows deeper and network stop learning earlier

###### Covariate Shift
> What is covariate shift? While in the process $X \rightarrow Y$
> $$P^{train}(y|x) = P^{test}(y|x)$$
> $$but\; P^{train}(x) \neq P^{test}(x)$$


# ToDo
## Normalizations
- weight scale invariance
- data scale invariance

#### Batch Normalization
#### Layer Normalization
#### Weight Normalization
#### Cosine Normalization









