---
title: Gradient Descent
date: 
  created: 2020-02-02 21:04:06
categories: 
  - ML
---


# gradient-based optimization algorithms

<!-- more -->
## Gradient Descent variants

#### Batch Gradient Descent (BGD)

Vanilla gradient descent, aka batch gradient descent, computes the gradient of the cost function w.r.t. to the parameters θ

Batch gradient descent is guaranteed to converge 
- to the global minimum for convex error surfaces
- to a local minimum for non-convex surfaces

#### Stochastic Gradient Descent (SGD)
Batch gradient descent performs redundant computations for large datasets, as it recomputes gradients for similar examples before each parameter update.
SGD does away with this redundancy by performing one update at a time. It is therefore usually much faster and can also be used to learn online.
SGD performs frequent updates with a high variance that cause the objective function to *fluctuate* heavily.
While batch gradient descent converges to the minimum of the basin the parameters are placed in, SGD's fluctuation,
- enables it to jump to new and potentially better local minima
- this ultimately complicates convergence to the exact minimum, as SGD will keep overshooting

when we slowly decrease the learning rate, SGD shows the same convergence behavior as batch gradient descent, almost certainly converging to a *local* or the *global* minimum for *non-convex* and *convex* optimization respectively.

#### Mini-batch Gradient Descent (MB-GD)
Mini-batch gradient descent finally takes the best of both worlds and performs an update for every mini-batch of n training examples

- reduces the variance of the parameter updates, which can lead to more stable convergence
- can make use of highly optimized matrix optimizations common to state-of-the-art deep learning libraries that make computing the gradient w.r.t. a mini-batch very efficient
- Mini-batch gradient descent is typically the algorithm of choice when training a neural network and the term SGD usually is employed also when mini-batches are used


#### Challenges

- **Choosing a proper learning rate can be difficult.**
> A learning rate that is too small leads to painfully slow convergence, while a learning rate that is too large can hinder convergence and cause the loss function to fluctuate around the minimum or even to diverge.

- **Learning rete schedules try to adjust the learning rate during training**
> e.g. annealing, i.e. reducing the learning rate according to a pre-defined schedule or when the change in objective between epochs falls below a threshold. These schedules and thresholds, however, have to be defined in advance and are thus unable to adapt to a dataset's characteristics

- **The same learning rate applies to all parameter updates**
> If our data is sparse and our features have very different frequencies, we might not want to update all of them to the same extent, but perform a larger update for rarely occurring features

- **Minimizing high non-convex error functions common for neural networks is avoiding getting trapped in their numerous suboptimal local minima**
> The difficulty arises in fact not from local minima but from saddle points, i.e. points where one dimension slopes up and another slopes down. These saddle points are usually surrounded by a plateau of the same error, which makes it notoriously hard for SGD to escape, as the gradient is close to zero in all dimensions.


## Gradient Descent Optimization Algorithms
We will not discuss algorithms that are infeasible to compute in practice for high-dimensional data sets, e.g. second-order methods such as Newton's method.

#### Momentum
SGD has trouble navigating ravines, i.e. areas where the surface curves much more steeply in one dimension than in another, which are common around local optima.

Some implementations exchange the signs in the equations. The momentum term γ is usually set to 0.9 or a similar value.

When using momentum, we push a ball down a hill. The ball accumulates momentum as it rolls downhill, 
becoming faster and faster on the way (until it reaches its terminal velocity if there is air resistance, i.e. γ<1). 
*The same thing happens to our parameter updates*: 
> The momentum term increases for dimensions whose gradients point in the same directions and reduces updates for dimensions whose gradients change directions. As a result, we gain *faster convergence and reduced oscillation*.


#### Nesterov Accelerated Gradient

We'd like to have a smarter ball, a ball that has a notion of where it is going so that it knows to slow down before the hill slopes up again.
Nesterov Accelerated Gradient (NAG) is a way to give our momentum term this kind of prescience. 
We know that we will use our momentum term γvθ<sub>t-1</sub> to move the parameters θ. 
Computing θ−γv<sub>t-1</sub> thus gives us an approximation of the next position of the parameters (the gradient is missing for the full update), 
a rough idea where our parameters are going to be. We can now effectively look ahead by calculating the gradient 
*not w.r.t. to our current parameters θ but w.r.t. the approximate future position of our parameters*

we are able to adapt our updates to the slope of our error function and speed up SGD in turn, 
we would also like to adapt our updates to each individual parameter to perform larger or smaller updates depending on their importance


The distinction between Momentum method and Nesterov Accelerated Gradient updates was
- Both methods are distinct only when the learning rate η is reasonably large. 
- When the learning rate η is relatively large, Nesterov Accelerated Gradients allows larger decay rate α than Momentum method, while preventing oscillations. 
- Both Momentum method and Nesterov Accelerated Gradient **become equivalent when η is small**


#### Adagrad
Adagrad is an algorithm for gradient-based optimization that does just this: 
It adapts the learning rate to the parameters, 
- performing smaller updates (i.e. low learning rates) for parameters associated with frequently occurring features, 
- and larger updates (i.e. high learning rates) for parameters associated with infrequent features.

For this reason, **it is well-suited for dealing with sparse data.**

Previously, we performed an update for all parameters θ at once as every parameter θ<sub>i</sub> used the same learning rate η. 
As Adagrad uses a different learning rate for every parameter θ<sub>i</sub> at every time step t, we first show Adagrad's per-parameter update, which we then vectorize.
For brevity, we use gt to denote the gradient at time step t. g<sub>t,i</sub> is then the partial derivative of the objective function w.r.t. to the parameter θ<sub>i</sub> at time step t

In its update rule, Adagrad modifies the general learning rate η at each time step t for every parameter θ<sub>i</sub> based on the past gradients that have been computed for θ<sub>i</sub>

θ<sub>t+1,i</sub>=θ<sub>t,i</sub>−η/√(G<sub>t,ii</sub>+ϵ)⋅g<sub>t,i</sub>

G<sub>t</sub>∈R<sup>d×d</sup> here is a diagonal matrix where each diagonal element i,i is the sum of the squares of the gradients w.r.t. θ<sub>i</sub> up to time step t,
while ϵ is a smoothing term that avoids division by zero. 
**Interestingly, without the square root operation, the algorithm performs much worse.**

- One of Adagrad's main benefits is that it eliminates the need to manually tune the learning rate
- Adagrad's main weakness is its accumulation of the squared gradients in the denominator
> Since every added term is positive, the accumulated sum keeps growing during training. This in turn causes the learning rate to shrink and eventually become infinitesimally small, at which point the algorithm is no longer able to acquire additional knowledge. The following algorithms aim to resolve this flaw.


#### Adadelta
Adadelta is an extension of Adagrad that seeks to its aggressive, monotonically decreasing learning rate.
Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to some fixed size w.

Instead of inefficiently storing w previous squared gradients, 
the sum of gradients is recursively defined as a decaying average of all past squared gradients. 




