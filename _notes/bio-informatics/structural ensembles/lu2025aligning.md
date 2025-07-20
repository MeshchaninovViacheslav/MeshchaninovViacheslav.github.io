---
layout: note
title: "Aligning Protein Conformation Ensemble Generation with Physical Feedback"
date: 2025-05-30
tags: [ensembles, APM]
link_pdf: "https://www.nature.com/articles/s41587-024-02395-w.pdf"
code: "https://github.com/bytedance/apm"

bibtex: |-
  @article{chen2025all,
    title={An All-Atom Generative Model for Designing Protein Complexes},
    author={Chen, Ruizhe and Xue, Dongyu and Zhou, Xiangxin and Zheng, Zaixiang and Zeng, Xiangxiang and Gu, Quanquan},
    journal={arXiv preprint arXiv:2504.13075},
    year={2025}
  }
---

## Motivation
---
> Crucial Role of Protein Dynamics

Understanding protein dynamics is a critical and complex challenge in studying protein functionality and regulation. Protein structures constantly transition between various conformational states across different spatial and temporal scales, which directly influences their biological functions.

> Limitations of traditional approaches

1. Traditionally, molecular dynamics (MD) simulations have been the primary computational tool for investigating the dynamic behavior of biological molecules. These simulations model the system of particles by evolving Newtonian equations of motion, with accelerations determined by predefined force fields (physical energy functions).
However, MD simulations are computationally prohibitive for capturing biologically relevant transitions, such as protein folding or unfolding, which often occur over micro- to millisecond timescales. Such simulations can require hundreds to thousands of GPU days depending on the system size.

2. While these data-driven approaches can generate structurally valid candidates, a significant challenge is that they do not explicitly model thermodynamic properties. A more accurate formulation of this problem is equilibrium sampling, which aims to sample conformation ensembles from the Boltzmann distribution over states. This is vital for accurately modeling protein ensembles and capturing their thermodynamic stability. However, this remains highly intractable because it requires generative models to not only produce plausible structures but also to match the underlying energy landscape

## Methodology
---

> Notation

Amino acid sequence with length $L$: $$ {\bf c} = (c_1, c_2, \ldots, c_L) \in \mathcal{V}^L$$ where $$\mathcal{V}$$ denotes the vocabulary of 20 standard amino acids.

Structure is described by the 3D positions of its constituent atoms: $$ { \bf x} = (x_1, x_2, \ldots, x_N) \in \mathbb{R}^{N \times 3}$$

