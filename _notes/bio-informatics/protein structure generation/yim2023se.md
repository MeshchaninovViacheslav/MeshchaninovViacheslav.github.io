---
layout: note
title: SE (3) diffusion model with application to protein backbone generation
importance: 1
date: 2023-02-05
tags: [3d, proteins, framediff]
link_pdf: "https://arxiv.org/pdf/2302.02277"
code: "https://github.com/jasonkyuyim/se3_diffusion"

bibtex: |-
  @article{yim2023se,
    title={SE (3) diffusion model with application to protein backbone generation},
    author={Yim, Jason and Trippe, Brian L and De Bortoli, Valentin and Mathieu, Emile and Doucet, Arnaud and Barzilay, Regina and Jaakkola, Tommi},
    journal={arXiv preprint arXiv:2302.02277},
    year={2023}
  }
---

## Reciept of FrameDiff

---

1. Train two flow diffusion model: Riemannian diffusion for residue rotation matrices and Gaussian diffusion for residue translations.
2. Use best practices from AlphaFold2.

## Methodology

---

![folding]({{ "assets/notes-img/bio-informatics/protein structure generation/yim2023se/20.png" | relative_url }}){:width="800px" .img-frame-black}

The authors use the same parameterization as in
[AlphaFold2]({% link _notes/bio-informatics/protein generation/jumper2021highly.md %})
and
[RFDiffusion]({% link _notes/bio-informatics/protein structure generation/watson2023novo.md %}).
The protein is represented as a sequence of residue rotation and translation.

<br>
The diffusion process is essentially the same as in RFDiffusion. The main difference is the absence of folding pretraining.

I recommend the [presentation](https://www.cs.princeton.edu/courses/archive/fall23/cos597N/lectures/2023_11_16_framediff_rfdiffusion.pdf) of Jason Yim, which compares FrameDiff, RFDiffusion, and FrameFlow.
