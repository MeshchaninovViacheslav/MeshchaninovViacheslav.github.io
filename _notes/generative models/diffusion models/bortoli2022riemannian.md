---
layout: note
title: Riemannian Score-Based Generative Modelling
importance: 1
date: 2024-07-02
tags: [Riemannian manifolds]
link_pdf: "https://arxiv.org/pdf/2202.02763"
code: "https://github.com/oxcsml/riemannian-score-sde"

bibtex: |-
  @article{bortoli2022riemannian,
    title={Riemannian score-based generative modelling},
    author={Bortoli, Valentin De and Mathieu, Emile and Hutchinson, MJ and Thornton, James and Teh, Yee Whye and Doucet, Arnaud},
    journal={Advances in Neural Information Processing Systems},
    volume={35},
    year={2022}
  }
---

## Motivation
---

Score-based generative modelling (SGM) consists of a “noising” stage, whereby a diffusion is used to gradually add Gaussian noise to data, and a generative model, which entails a “denoising” process defined by approximating the time-reversal of the diffusion.
Existing SGMs assume that data is supported on a Euclidean space, i.e. a manifold with flat geometry. In many domains such as robotics, geoscience or protein modelling, data is often naturally described by distributions living on Riemannian manifolds and current SGM techniques are not appropriate. 

## Methodology
---

### Euclidean Score-based Generative Modelling

Let's recall here briefly the key concepts behind SGMs on the Euclidean space $\mathbb{R}^d$.

<br>
Consider a forward noising process $$(\mathbf{X}_t)_{t \geq 0}$$ defined by the following Stochastic Differential Equation (SDE):

\begin{equation}
    \mathrm{d} \mathbf{X}_t = -\mathbf{X}_t \mathrm{d}t + \sqrt{2} \, \mathrm{d}\mathbf{B}_t, \quad \mathbf{X}_0 \sim p_0,
    \tag{1}
\end{equation}

where $$(\mathbf{B}_t)_{t \geq 0}$$ is a $d$-dimensional Brownian motion and $p_0$ is the data distribution. 
The available data gives us an empirical approximation of $p_0$. 

<br>
The time-reversed process $$(\mathbf{Y}_t)_{t \geq 0} = (\mathbf{X}_{T - t})_{t \in [0, T]}$$ also satisfies an SDE given by

$$
\begin{equation}
    \mathrm{d} \mathbf{Y}_t = \left\{ \mathbf{Y}_t + 2 \nabla \log p_{T - t}(\mathbf{Y}_t) \right\} \mathrm{d}t + \sqrt{2} \, \mathrm{d}\mathbf{B}_t, \quad \mathbf{Y}_0 \sim p_T,
    \tag{2}
\end{equation}
$$

where $p_t$ denotes the density of $$\mathbf{X}_t$$. 
By construction, the law of $$\mathbf{Y}_{T - t}$$ is equal to the law of $$\mathbf{X}_t$$ for $t \in [0, T]$ and in particular $$\mathbf{Y}_T \sim p_0$$.
Hence, if one could sample from $$(\mathbf{Y}_t)_{t \in [0, T]}$$ then its final distribution would be the data distribution $p_0$. Unfortunately we cannot sample exactly from (2) as $p_T$ and the scores $\left( \nabla \log p_t(x) \right)_{t \in [0, T]}$ are intractable. Hence SGMs rely on a few approximations.

- $p_T$ is replaced by the reference distribution $\mathcal{N}(0, \mathbb{I}_d)$. 
- the following denoising score matching identity is exploited to estimate the scores:
$$
\nabla_{x_t} \log p_t(x_t) = \int_{\mathbb{R}^d} \nabla_{x_t} \log p_{t|0}(x_t|x_0) \, p_{t|0}(x_0|x_t) \, \mathrm{d}x_0,
$$
where $p_{t|0}(x_t|x_0)$ is the transition density of the process (1) which is available in closed form. 
It follows directly that $\nabla \log p_t$ is the minimizer of
$$
\ell_t(s) = \mathbb{E} \left[ \| s(\mathbf{X}_t) - \nabla_{x_t} \log p_{t|0}(\mathbf{X}_t | \mathbf{X}_0) \|^2 \right]
$$
over functions $s$ where the expectation is over the joint distribution of $$\mathbf{X}_0, \mathbf{X}_t$$.
This result can be leveraged by considering a neural network $$s_\theta : [0, T] \times \mathbb{R}^d \to \mathbb{R}^d$$ trained by minimizing the loss function
$$
\ell(\theta) = \int_0^T \lambda_t \ell_t(s_\theta(t, \cdot)) \mathrm{d}t
$$
for some weighting function $\lambda_t > 0$.

- Finally, an Euler–Maruyama discretization of (2) is performed using a discretization step $\gamma$ such that $T = \gamma N$ for $$N \in \mathbb{N}$$:

$$
\begin{equation}
    Y_{n+1} = Y_n + \gamma \{ Y_n + 2s_\theta(T - n\gamma, Y_n)  \} + \sqrt{2\gamma} Z_n, \quad Y_0 \sim \mathcal{N}(0, \mathbb{I}_d), \quad Z_n \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \mathbb{I}_d).
\end{equation}
$$

### Riemannian Score-based Generative Modelling

<details class="details-frame" markdown="1">
  <summary> Riemannian manifold explanation </summary>

**Simple example**

Imagine the Earth. From space it is a big 3‑D ball, but if you are an ant walking on the surface you can move only east–west and north–south, two directions. So the surface is a 2‑dimensional world living inside 3‑dimensional space. That is what mathematicians call a 2‑dimensional manifold.

Exactly the same idea works for:
- Roads or wires running through space → 1‑dimensional manifolds.
- The sheet of paper you are reading → 2‑dimensional manifold.
- Hard‑to‑picture shapes that sit inside higher‑dimensional space → still manifolds; you just need more imagination (or equations) to see them.

<br>
Saying “manifold” only tells us where we can go. It does not yet tell us
how long a step is,
what the shortest route between two towns is, or
how to measure an angle between two roads.
To add that information we give every point on the surface its own little “ruler”―a dot‑product for the arrows (vectors) that live in the tangent plane at that point. The rule for measuring must change smoothly as you walk around. Once you add those smoothly varying rulers you have a Riemannian manifold.

<br>
Euclidean space, the n-sphere, hyperbolic space, and smooth surfaces in three-dimensional space, such as ellipsoids and paraboloids, are all examples of Riemannian manifolds.

</details>

Assume that $\mathcal{M}$ is a complete, orientable connected and boundaryless Riemannian manifold, endowed with a Riemannian metric $$g: T \mathcal{M} \times T \mathcal{M} \longrightarrow \mathbb{R}$$.

Four components are required to extend SGMs to this setting: 
1. a forward noising process on $\mathcal{M}$ which converges to an easy-to-sample reference distribution, 
2. a time-reversal formula on $\mathcal{M}$ which defines a backward generative process
3. a method for approximating samples of SDEs on manifolds, 
4. a method to efficiently approximate the drift of the time-reversal process.

> Noising processes on manifolds

The first necessary component is a suitable generic noising process on manifolds that will converge to a convenient stationary distribution. A simple choice is to use Langevin dynamics described by:

\begin{equation}
    \mathrm{d} \mathbf{X}_t = - \frac{1}{2} \nabla U(\mathbf{X}_t) \mathrm{d}t + \mathrm{d}\mathbf{B}_t
    \tag{3}
\end{equation}

Two simple choices for $U(x)$:

1. $$U(x) = \frac{d^2_{\mathcal{M}}(x, \mu)}{2 \gamma^2}$$, where $d_{\mathcal(M)}$ is the geodesic distance and $\mu \in \mathcal(M)$ is an arbitrary mean location.
<br>
In this case $$\nabla U(\mathbf{X}_t) = -\frac{\exp_{\mathbf{X}_t}^{-1}(\mu)}{\gamma^2}$$. This is the potential of the ‘Riemannian normal’ distribution. 

2. Alternative is $$U(x) = \frac{d^2_{\mathcal{M}}(x, \mu)}{2 \gamma^2} + \log \mid D \exp_{\mathbf{X}_t}^{-1}(\mu) \mid $$

## TODO
