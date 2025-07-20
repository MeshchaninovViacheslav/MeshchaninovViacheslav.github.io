---
layout: note
title: Continuous Diffusion Model for Language Modeling
importance: 1
date: 2025-02-17
tags: [text-diffusion, rdlm]
link_pdf: "https://arxiv.org/pdf/2502.11564"
code: "https://github.com/harryjo97/RDLM"

bibtex: |-
  @article{jo2025continuous,
    title={Continuous Diffusion Model for Language Modeling},
    author={Jo, Jaehyeong and Hwang, Sung Ju},
    journal={arXiv preprint arXiv:2502.11564},
    year={2025}
  }
---

## Methodology
---

### Preliminaries

#### Statistical manifold of categorical distribution

1. Let $$\mathcal{X} = \{1, \cdots, d\}$$ denote the discrete data space
2. $$\Delta^{d-1} = \{(p_1, \cdots, p_d) \in \mathbb{R}^d \mid \sum_i p_i = 1, p_i \geq 0\}$$ 
denote the $(d - 1)$-dimensional probability simplex. 

A natural choice of a Riemannian metric on the simplex is the Fisher–Rao metric. 
For an interior point $p \in \Delta^{d-1}$, the Fisher–Rao metric is defined as follows:
$$
\begin{equation}
g_{FR}(p)[x, y] := \langle x, y \rangle_p := \left\langle \frac{x}{\sqrt{p}}, \frac{y}{\sqrt{p}} \right\rangle 
= \sum_{i=1}^d \frac{x_i y_i}{p_i}, \quad x, y \in \mathcal{T}_p \Delta^{d-1}, 
\end{equation}
$$

This induces a geodesic distance on the simplex defined as follows:

$$
\begin{equation}
d(p, q) = 2 \cos^{-1} \left( \sum_{i=1}^d \sqrt{p_i q_i} \right), 
\quad p, q \in \Delta^{d-1}, 
\end{equation}
$$

where $p$ and $q$ correspond to the parameters of categorical distributions. 

<br>
> The probability simplex $\Delta^{d-1}$ equipped with the Fisher–Rao metric is a Riemannian manifold  called the statistical manifold of categorical distribution, denoted as $\mathcal{P}(\mathcal{X})$ throughout the paper. 

<!-- <br>
There exists a diffeomorphism (isomorphism of differentiable manifolds) from the statistical manifold $$\mathcal{P}(\mathcal{X})$$ to the positive orthant of the $(d - 1)$-dimensional sphere $$\mathbb{S}_+^{d-1}$$:

$$
\pi : \mathcal{P}(\mathcal{X}) \rightarrow \mathbb{S}_+^{d-1} ;\quad p_i \mapsto u_i = \sqrt{p_i},
\tag{3}
$$

which induces the geodesic distance 
$d_g(u, v) = \cos^{-1} \langle u, v \rangle$ 
for $u, v \in \mathbb{S}_+^{d-1}$, where $\langle \cdot, \cdot \rangle$ denotes the Euclidean inner product. -->

#### Hypersphere

- $$\mathbb{S}^{d-1}$$ denotes the $(d-1)$-dimensional sphere $$\bigg\{ \mathbf{u} = (u_1, \cdots, u_d) : \sum_i u_i^2 = 1 \bigg\}$$ 
- $$\mathbb{S}^{d-1}_+ = \bigg\{ \mathbf{u} = (u_1, \cdots, u_d) : \sum_i u_i^2 = 1, u_i \geq 0 \bigg\}$$ denotes a positive orthant of $$\mathbb{S}^{d-1}$$. 

<!-- The hypersphere $\mathbb{S}^{d-1}$ can be embedded into the ambient Euclidean space $\mathbb{R}^d$, 
which induces a canonical inner product $\langle \mathbf{x}, \mathbf{y} \rangle := \sum_{i=1}^d x_i y_i$.  -->

For a discrete sample space $$\mathcal{X} = \{1, 2, \cdots, d\}$$, there exists a diffeomorphism (isomorphism of differentiable manifolds) from $\mathcal{P}(\mathcal{X})$ to $\mathbb{S}^{d-1}_+$ defined as follows:

$$
\begin{align*}
\pi : \mathcal{P}(\mathcal{X}) \rightarrow \mathbb{S}^{d-1}_+ ; \quad p_i \mapsto u_i = \sqrt{p_i}, \newline
\pi^{-1} : \mathbb{S}^{d-1}_+ \rightarrow \mathcal{P}(\mathcal{X}) ; \quad u_i \mapsto p_i = u_i^2.
\end{align*}
$$

The diffeomorphism induces the geodesic distance (minimal distance between two points on hypersphere) on $\mathbb{S}^{d-1}_+$:
$$
\begin{equation}
d_g(\mathbf{u}, \mathbf{v}) = \cos^{-1} \langle \mathbf{u}, \mathbf{v} \rangle, 
\quad \mathbf{u}, \mathbf{v} \in \mathbb{S}^{d-1}_+,
\end{equation}
$$
 
The corresponding exponential and logarithm maps on $\mathbb{S}^{d-1}$ can be computed as follows:

\begin{equation}
\exp_{\mathbf{u}} \mathbf{x} = \cos(\|\mathbf{x}\|) \mathbf{u} + \sin(\|\mathbf{x}\|) 
\frac{\mathbf{x}}{\|\mathbf{x}\|}, \quad \mathbf{u} \in \mathbb{S}^{d-1}, \ \mathbf{x} \in T_{\mathbf{u}}(\mathbb{S}^{d-1}),
\end{equation}

\begin{equation}
\exp_{\mathbf{u}}^{-1}(\mathbf{v}) = \frac{\cos^{-1} \langle \mathbf{u}, \mathbf{v} \rangle}
{\sqrt{1 - \langle \mathbf{u}, \mathbf{v} \rangle^2}} \left( \mathbf{v} - \langle \mathbf{u}, \mathbf{v} \rangle \mathbf{u} \right),
\quad \mathbf{u}, \mathbf{v} \in \mathbb{S}^{d-1}.
\end{equation}

Basically this formulas define the projection rule between initial space $\mathbb{S}^{d-1}$ and its tangent space $$T_{\mathbf{u}}(\mathbb{S}^{d-1})$$.

> The motivation of mapping $$\mathcal{P}(\mathcal{X})$$ to the positive orthant of a hypersphere $$\mathbb{S}_+^{d-1}$$ is that the Fisher–Rao metric is ill-defined on the boundary of the manifold where the initial distribution of the parameterized data lies. So the task of modeling the distribution of discrete data can be reformulated as modeling a distribution $p_{\text{data}}$ on the hypersphere.


<!-- #### Continuous reparameterization of discrete data

> The Fisher–Rao metric is ill-defined on the boundary of the manifold where the initial distribution of the parameterized data lies.  Because of that authors apply the diffeomorphism $\pi$ to map $$\mathcal{P}(\mathcal{X})$$ to the positive orthant of a hypersphere $\mathbb{S}_+^{d-1}$.

<br>
The reparameterized data distribution $p_{\text{data}}$ on the hypersphere can be written as
$$
p_{\text{data}}(x) = \sum_{k=1}^d p_k \delta(x - e_k),
$$ 
where $p_k$ denotes the probability of the $k$-th state, and $e_k$ are $d$-dimensional one-hot vectors. -->

### Generative Process on Hypersphere

On a general manifold $\mathcal{M}$ (complete, orientable, connected, and boundaryless) the logarithm bridge process from $x_0 \in \mathcal{M}$ to $x_1 \in \mathcal{M}$ is defined as follows:

$$
\begin{equation}
    \mathrm{d}\bar{X}_t = \gamma_t \exp^{-1}_{\bar{X}_t}(x_1)\mathrm{d}t + \sigma_t \mathrm{d}B_t^{\mathcal{M}}, 
    \quad \bar{X}_0 = x_0, \quad \gamma_t := \frac{\sigma_t^2}{\int_t^T \sigma_s^2 \mathrm{d}s}
\end{equation}
$$

where $\exp^{-1}_x(\cdot)$ denotes the logarithm map on $\mathcal{M}$ at point $x$, 
and $B_t^{\mathcal{M}}$ is the Brownian motion defined on $\mathcal{M}$. 

<details class="details-frame" markdown="1">
  <summary>Interpretation of the Equation </summary>

Even though $$\bar{X}_t \in \mathcal{M}$$, the SDE is written **in the tangent space** $$T_{\bar{X}_t} \mathcal{M}$$.

In differential geometry, this is a standard trick: 
**define dynamics in the tangent space**, 
then use **the exponential map** (or a retraction) 
to interpret these dynamics as evolving on the manifold.

This is especially necessary in **manifold-valued stochastic processes**, 
where there's no global coordinate system. So what actually happens here is:


> The SDE is computed locally in $T_{\bar{X}_t} \mathcal{M}$, and $\bar{X}_t$ evolves by lifting the update back to the manifold.


Let’s rewrite the core:

$$
\mathrm{d}\bar{X}_t = \gamma_t \exp^{-1}_{\bar{X}_t}(x_1)\, \mathrm{d}t + \sigma_t\, \mathrm{d}B_t^{\mathcal{M}}
$$

This defines a **logarithmic bridge**: a stochastic process starting at $x_0 \in \mathcal{M}$, gently "pulled" toward $x_1 \in \mathcal{M}$  using the **logarithmic map** as a direction.

- $\exp^{-1}_{\bar{X}_t}(x_1)$ gives the **direction and magnitude** in which to pull toward the target.
- $\gamma_t$ adjusts the strength of this drift over time.
- $\sigma_t\, \mathrm{d}B_t^{\mathcal{M}}$ is Brownian motion noise **on the manifold**, typically modeled via parallel transport or using local charts.


</details>

In the case of $\mathcal{M} = \mathbb{S}^{d-1}$, we can derive the logarithm bridge process from $x_0$ to $e_k$ using formula (5):
\begin{equation}
    \mathrm{d}\bar{X}_t = 
    \gamma_t \frac{\cos^{-1} \langle \bar{X}_t, e_k \rangle 
    (e_k - \langle \bar{X}_t, e_k \rangle \bar{X}_t)}
    {\sqrt{1 - \langle \bar{X}_t, e_k \rangle^2}} \mathrm{d}t 
    + \sigma_t \mathrm{d}B_t^d, \quad \bar{X}_0 = x_0
\end{equation}

### Diffusion Mixture Representation

> From the logarithm bridge processes (Eq.(7)), the authors construct a generative process $$\{X_t\}_{t=0}^T$$ on $$\mathbb{S}^{d-1}$$ using the diffusion mixture representation.

Let $p_{\text{data}}(x) = \sum_{k=1}^d p_k \delta(x - e_k)$ be a data distribution on $\mathbb{S}^{d-1}$. 
Then the following SDE defines a diffusion process that transports the initial point $x_0 \in \mathbb{S}^{d-1}$ 
to the distribution $p_{\text{data}}$:

$$
\begin{equation}
\mathrm{d}X_t = \left[ \sum_{k=1}^d p_{T|t}(e_k \mid X_t) \, \eta^k(X_t, t) \right] \mathrm{d}t 
+ \sigma_t \mathrm{d}B_t^d, \quad X_0 = x_0,
\end{equation}
$$

$$
\begin{equation}
\eta^k(z, t) := \gamma_t \frac{ \cos^{-1} \langle z, e_k \rangle (e_k - \langle z, e_k \rangle z) }
{ \sqrt{1 - \langle z, e_k \rangle^2} },
\end{equation}
$$

where $p_{T|t}(e_k \mid X_t)$ represents the conditional probability that the process will reach 
the endpoint $e_k$ at time $T$, given the current state $X_t$ at time $t$.

<br>
The authors derive a new family of generative processes by constructing a mixture over the 
time marginals of generative processes $$\{ \mathbb{Q}_t^i : 1 \leq i \leq n \}$$:

$$
\begin{equation}
    \mathbb{Q}_t^{\text{mix}} := \sum_{i=1}^n \lambda_t^i \mathbb{Q}_t^i, 
    \quad \sum_{i=1}^n \lambda_t^i = 1, \quad 0 \leq \lambda_t^i \leq 1,
\end{equation}
$$

where $\lambda_t^i$ is the time-dependent mixing schedule assigned to the $i$-th generative path. 
This construction allows the resulting process to transition between different generative behaviors over time.

In particular, the authors propose a simple mixture path built from mixing the time marginals 
of the masked diffusion and uniform diffusion, for a time-dependent schedule $\lambda_t$ as follows:

\begin{equation}
    \lambda_t \mathbb{Q}_t^{\text{mask}} + (1 - \lambda_t) \mathbb{Q}_t^{\text{unif}},
\end{equation}

with initial distribution 
$\lambda_0 \delta(e_m) + (1 - \lambda_0) \delta\left( \sum_{i=1}^d e_i / \sqrt{d} \right)$. 

### Training

#### Model parameterization

Instead of approximating the drift function directly, the authors model the probability $$p_{T \mid t}(X_T \mid X_t)$$ with a neural network $s_\theta$ as follows:

$$
\begin{equation}
p_\theta(X_t, t) := \texttt{softmax}(s_\theta(X_t, t)) = 
\left[ p_{T|t}(e_1 \mid X_t), \cdots, p_{T|t}(e_d \mid X_t) \right]^\top,
\end{equation}
$$

In the case of masked diffusion, we set the probability 
$p_{T|t}(e_m \mid X_t)$ to be zero for all $t$, 
indicating that the final state cannot be a mask token. 
From Eq.(12), the drift of the mixture process in Eq.(8) is parameterized as follows:

\begin{equation}
\eta_\theta(X_t, t) = \sum_{k=1}^d \left\langle p_\theta(X_t, t), e_k \right\rangle \eta^k(X_t, t).
\end{equation}

#### Likelihood bound

**Variational upper bound:**

Let $\mathbb{Q}^k$ be a bridge process with starting point $x_0$ and endpoint $e_k$. 
From the KL divergence between $\mathbb{Q}^\theta$ and $\mathbb{Q}^k$, a point-wise upper bound on the negative log-likelihood could be derived:

$$
\begin{equation}
- \log \hat{p}_\theta(e_k) 
= D_{\text{KL}}(\mathbb{Q}_{T}^k \| \mathbb{Q}^\theta_T) 
\leq \mathbb{E}_{X \sim \mathbb{Q}^k} \left[ 
\frac{1}{2} \int_0^T \left\| \sigma_t^{-1} \left( \eta_\theta(X_t, t) - \eta^k(X_t, t) \right) 
\right\|_2^2 \mathrm{d}t \right]
\end{equation}
$$


**Objective:**

Using Eq.(13) the following objective could be derived:

$$
\begin{equation}
\mathcal{L}(\theta) = 
\mathbb{E}_{e_k \sim p_{\text{data}} \atop \mathbf{X} \sim \mathbb{Q}^k}
\left[
\frac{1}{2} \int_0^T \sigma_t^{-2} 
\left\| 
\sum_{l=1}^d \left\langle p_\theta(\mathbf{X}_t, t), \mathbf{e}_l \right\rangle 
\eta^l(\mathbf{X}_t, t) - \eta^k(\mathbf{X}_t, t)
\right\|_2^2 \, \mathrm{d}t
\right].
\end{equation}
$$

This formula could be further simplified:

$$
\begin{equation}
\mathcal{L}(\theta) = 
\mathbb{E}_{e_k \sim p_{\text{data}} \atop \mathbf{X} \sim \mathbb{Q}^k}
\left[
\frac{1}{2} \int_0^T \sigma_t^{-2} 
\left\| 
\sum_{l=1}^d \left\langle p_\theta(\mathbf{X}_t, t) - \mathbf{e}_k, \mathbf{e}_l \right\rangle 
\eta^l(\mathbf{X}_t, t)
\right\|_2^2 \, \mathrm{d}t
\right].
\end{equation}
$$

Futhermore, in the Appendix the authors show that this objective could be upperbounded by cross-entropy objective:

$$
\begin{equation}
\mathcal{L}^{CE}(\theta) = 
\mathbb{E}_{e_k \sim p_{\text{data}} \atop \mathbf{X} \sim \mathbb{Q}^k}
\left[
\int_0^T -\log \left\langle p_\theta(\mathbf{X}_t, t), \mathbf{e}_k \right\rangle \mathrm{d}t
\right]
\end{equation}
$$

> The authors experimentally find that the cross-entropy loss yields faster convergence in training and leads to better performance than
the mean squared error loss.

#### Approximation of transition distribution

Training objective involves sampling $x_t$ at each training iteration.
This introduces a significant bottleneck during training, as it requires simulating the process.

The authors propose to approximate the manifold distribution $p(x_t \mid x_0, x_T)$ as the image of a Gaussian distribution on the tangent space via the exponential map.

### Modeling sequence of tokens 

Basically in the same manner as token process the authors model sequence process.

#### Dimension splitting

Additionally, the authors say that modeling texts with large vocabulary is challenging for neural network.
That's why they use dimmension splitting. Basically, they represent the whole probability vector of size $d$ as $m$ probablity vectors of size $b$, $b = \log_{m} d$. As I understood they just use $m$ softmax heads after neural network. This makes neural networks training significantly easier.


