---
layout: note
title: Generating novel, designable, and diverse protein structures by equivariantly diffusing oriented residue clouds
importance: 1
date: 2023-01-29
tags: [3d, proteins, genie]
link_pdf: "https://arxiv.org/pdf/2301.12485"
code: "https://github.com/aqlaboratory/genie"

bibtex: |-
  @article{lin2023generating,
    title={Generating novel, designable, and diverse protein structures by equivariantly diffusing oriented residue clouds},
    author={Lin, Yeqing and AlQuraishi, Mohammed},
    journal={arXiv preprint arXiv:2301.12485},
    year={2023}
  }
---

## Receipt of GENIE
---

1. Represent protein as a sequence of $C_{\alpha}$ atomic coordinates.
2. Use standard Gaussian diffusion and SE(3)-equivariant denoizer model.

## Methodology

Genie is a DDPM that generates protein backbones as a sequence of $C_{\alpha}$ atomic coordinates.

In other words, the protein is represented by a sequence $x = [x^1, \ldots, x^N]$ of $C_{\alpha}$ coordinates, where $N$ is a number of residues.

Then standard Gaussian diffusion process is applied.

### Model

![folding]({{ "assets/notes-img/bio-informatics/protein structure generation/lin2023generating/19.png" | relative_url }}){:width="800px" .img-frame-black}

> FS frames

Each FS frame represents the position and orientation of a residue relative to the global reference frame. Once constructed, these FS frames enable downstream model components, including IPA, to reason about the relative orientations of protein residues and parts.

$$
\mathbf{t}^i = \frac{\mathbf{x}^{i+1} - \mathbf{x}^i}{\|\mathbf{x}^{i+1} - \mathbf{x}^i\|}
$$

$$
\mathbf{b}^i = \frac{\mathbf{t}^{i-1} \times \mathbf{t}^i}{\|\mathbf{t}^{i-1} \times \mathbf{t}^i\|}
$$

$$
\mathbf{n}^i = \mathbf{b}^i \times \mathbf{t}^i
$$

$$
\mathbf{R}^i = [\mathbf{t}^i, \mathbf{b}^i, \mathbf{n}^i]
$$

$$
\mathbf{F}^i = (\mathbf{R}^i, \mathbf{x}^i)
$$

$F_i$ is a discrete FS frame.

