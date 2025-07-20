---
layout: note
title: Simulating 500 million years of evolution with a language model
importance: 1
date: 2024-07-02
tags: [masked diffusion, esm3, proteins, cogeneration]
link_pdf: "https://www.biorxiv.org/content/10.1101/2024.07.01.600583v1.full.pdf"
code: "https://github.com/evolutionaryscale/esm"

bibtex: |-
  @article{hayes2025simulating,
    title={Simulating 500 million years of evolution with a language model},
    author={Hayes, Thomas and Rao, Roshan and Akin, Halil and Sofroniew, Nicholas J and Oktay, Deniz and Lin, Zeming and Verkuil, Robert and Tran, Vincent Q and Deaton, Jonathan and Wiggert, Marius and others},
    journal={Science},
    pages={eads0018},
    year={2025},
    publisher={American Association for the Advancement of Science}
  }
---

## Summary
---

ESM-3 introduces a masked diffusion approach that operates on a tokenized representation of sequence, structure, and function. Each aspect is encoded independently using specialized tokenizers.

## Technical Details
---

### Sequence Tokenizer

29 tokens total: 20 canonical + BOS, EOS, MASK, PAD, UNK + 4 non-standard (B, U, Z, O).


### Structure Tokenizer

Each residue is associated with one of 4,096 structure tokens (+4 special tokens), designed to provide a rich, learned representation of its local neighborhood. The tokens are generated with a VQ-VAE encoder, with a corresponding decoder to enable decoding of generated tokens back to 3D coordinates.


#### Encoder

The VQ-VAE encoder consists of two geometric attention blocks with an embedding width of 1024 and 128 geometric heads.

Each neighborhood is processed completely independently; for each residue, the encoder only uses the information of its 16 nearest neighbors.

It means that hidden state has dimension of $L \times 16 \times d$. This means that the encoder outputs 16 latents per residue. However, we want to learn a single token, i.e., a single latent per residue, hence we take the embedding corresponding to the query residue position $N_{:, 0, :}$.

#### Codebook Learning

We chose to learn the codebook as an exponential moving average of encoder outputs. To improve codebook utilization, unused codes are re-initialized to encoder outputs.

#### Decoder

While the encoder independently processes all local structures in parallel, the decoder attends over the entire set of L tokens to reconstruct the full structure. It is composed using a stack of bidirectional Transformer blocks with regular self-attention.

The VQ-VAE is trained in two stages. In the first stage, a smaller decoder is trained to only predict backbone coordinates. In the second stage, the encoder and codebook are frozen, the decoder weights are re-initialized and the network size is expanded to predict all atom coordinates.

#### Geometric Attention

**Translation Local Coordinates to Global Coordinates**

We look at protein backbone as a set of local residues. Each residue is described by its geometric orientation in the space. And the global view of a protein is defined by the position of each C<sub>α</sub> atom. So that the backbone is described by:

#### Self-Attention

Unlike regular self-attention, which only operates on per-residue embeddings, Geometric Attention incorporates the per-residue frames T to integrate geometric information in a rotation and translation invariant way. The process of forming the attention matrix A is as follows.

Input: `X ∈ ℝ^{L × d}`, `T = [{Rᵢ, tᵢ}] ∈ SE(3)^L`

`# TODO`

#### Losses

A bunch of losses. `# TODO`



### Function Tokenizer

`# TODO`


## Training
---

### Pretraining Tasks

In initial experimentation, we found that a fixed 15% noise schedule led to poor generation results, while a linear noise schedule where probability of each mask rate was constant led to good generation but poor representation learning results.

We find a good trade-off between representation learning and generation by sampling the noise schedule from a mixture distribution:
- 80% of the time, the mask rate is sampled from a Beta(3, 9) distribution with mean mask rate 25%.
- 20% of the time, the mask rate is sampled from a uniform distribution.

Resulting in an average overall mask rate of 30%.

For the structure coordinate track, we also modify the masking to be applied as span dropping 50% of the time. This ensures that the model sees contiguous regions of masked and provided coordinates, which better mimics the types of inputs users may provide.

Along with applying noise to each track, we want to ensure ESM-3 is able to perform well when some tracks are not provided at all.


### Structure Noise

We apply Gaussian noise with standard deviation 0.1 to all coordinates the model takes as input.


**Additional Resource**: https://chrispiech.github.io/probabilityForComputerScientists/en/part4/beta/