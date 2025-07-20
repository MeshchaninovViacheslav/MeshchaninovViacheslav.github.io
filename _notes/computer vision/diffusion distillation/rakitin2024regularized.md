---
layout: note
title: "Regularized Distribution Matching Distillation for One-step Unpaired Image-to-Image Translation"
date: 2024-06-20
tags: [distillation, i2i]
link_pdf: "https://arxiv.org/pdf/2406.14762"
code: "https://github.com/tianweiy/DMD2"

bibtex: |-
  @article{yin2024improved,
    title={Improved distribution matching distillation for fast image synthesis},
    author={Yin, Tianwei and Gharbi, Micha{\"e}l and Park, Taesung and Zhang, Richard and Shechtman, Eli and Durand, Fredo and Freeman, Bill},
    journal={Advances in neural information processing systems},
    volume={37},
    pages={47455--47487},
    year={2024}
  }
---

## Unpaired I2I and optimal transport
---

The problem of unpaired I2I consists of learning a mapping $G$ between the source distribution $p^{\mathcal{S}}$ and the target distribution $p^{\mathcal{T}}$ given the corresponding independent data sets of samples. 
When optimized, the mapping should appropriately adapt $G(x)$ to the target distribution $p^{\mathcal{T}}$, while preserving the input’s cross-domain features. 
However, at first glance, it is unclear what the preservation of cross-domain properties should look like.

<br>
One way to formalize this is by introducing the notion of a ``transportation cost'' $c(\cdot, \cdot)$ between the generator’s input and output and stating that it should not be too large on average. 
In a practical I2I setting, we can choose $c(\cdot, \cdot)$ as any reasonable distance between images or their features that we aim to preserve, such as pixel-wise distance or the difference between embeddings.

Monge optimal transport (OT) problem follows this reasoning and aims at finding the mapping with the least average transport cost among all the mappings that fit the target $p^{\mathcal{T}}$:

$$\inf_G \left\{ \mathbb{E}_{p^{\mathcal{S}}(x)} C(x, G(x)) \mid G(x) \sim p^{\mathcal{T}} \right\} $$

which can be seen as a mathematical formalization of the I2I task.

## Methodology
---

Our goal is to obtain a generator that maps objects from one distribution to objects of another distribution. 
The main difference from DMD is that now, instead of generating from a Gaussian distribution, we generate objects from another specified distribution $p^{\mathcal{S}}$. 
Therefore, the setup of DMD does not change significantly:

$$\nabla_\theta \mathcal{L}_{\text{DMD}}(\theta) = \nabla_\theta D_{KL} = \mathbb{E}_{\substack{z \sim p^{\mathcal{S}} \\ x = G_\theta(z)}} \left[ -\left(s_{\text{real}}(F(x, t)) - s_{\text{fake}}(F(x, t))\right) \frac{dG}{d\theta} \right]$$

However, there are no guarantees that the input an bd the output will be related. Similarly to the OT problem, we fix the issue by penalizing the transport cost between them. We obtain the following objective:

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{DMD}}(\theta) + \lambda \mathbb{E}_{p^{\mathcal{S}}} C(x, G_{\theta}(x))$$



