---
layout: note
title: "Highly accurate protein structure prediction with AlphaFold"
date: 2021-07-15
tags: [alphafold, proteins]
link_pdf: "https://www.nature.com/articles/s41586-021-03819-2"

bibtex: |-
  @article{jumper2021highly,
    title={Highly accurate protein structure prediction with AlphaFold},
    author={Jumper, John and Evans, Richard and Pritzel, Alexander and Green, Tim and Figurnov, Michael and Ronneberger, Olaf and Tunyasuvunakool, Kathryn and Bates, Russ and {\v{Z}}{\'\i}dek, Augustin and Potapenko, Anna and others},
    journal={nature},
    volume={596},
    number={7873},
    pages={583--589},
    year={2021},
    publisher={Nature Publishing Group}
  }
---

## AlphaFold Pipeline
---

The whole pipeline is aimed to generate protein backbone for protein sequence.
So the input to the pipeline is a sequence of amino acids.

![folding]({{ "assets/notes-img/bio-informatics/protein sequence generation/jumper2021highly/3.png" | relative_url }}){:width="800px"}

### Genetic database search → retrieving evolutionary relatives

Fast profile/HMM tools such as JackHMMER, HHblits or MMseqs2 scan tens of millions of sequences (UniRef 90, UniProt, BFD, MGnify, Uniclust 30) for anything that is measurably homologous to your protein. 

> What is measurably homologous? 

**Homologous = from a common ancestor.**
Two protein (or DNA) sequences are called homologous when they ultimately descended from the same ancestral gene. That fact is binary—either they are or they aren't—but we can only infer it from sequence data.

<br>

**Practical rules of thumb**

- **Percent identity.** 
Over a full-length alignment, ≥ 30 % identity is almost always enough to call two proteins homologous; below ~20 % you need profile or HMM methods to pick up the signal.
pmc.ncbi.nlm.nih.gov
- **Profile methods extend the range.** T
ools such as HHblits compare profiles (Hidden Markov Models) instead of raw sequences, letting you detect "remote" homology even when pairwise identity drops into the teens.

> Why does it matter?

The more distant—but still related—sequences you can find, the richer the evolutionary signal you get.
This operation returns a list of homologous sequences, a large pool of sequences that "know" something (through evolution) about how your protein folds.

### Multiple Sequence Alignment (MSA) → organising those relatives into columns

<details class="details-frame" markdown="1">
  <summary>1. Lines the hits up residue-by-residue so that amino acids performing the same functional/structural role sit in the same column. </summary>

  Think of a protein sequence as a long sentence where each amino-acid "word" has a job.
  When we build a multiple-sequence alignment (MSA), we try to line sentences up so that the words that play the same part of the sentence—subject, verb, object—appear in vertical columns. In the protein world, that means:

  <br>

  <table class="table">
    <thead>
      <tr>
        <th style="width: 25%;"><strong>Functional / Structural "Job"</strong></th>
        <th style="width: 50%;"><strong>What the Residue Is Doing</strong></th>
        <th style="width: 25%;"><strong>Example</strong></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Catalysis (chemical work)</strong></td>
        <td>Directly takes part in the reaction—often found in enzyme active sites.</td>
        <td>The Serine in the catalytic triad of serine proteases.</td>
      </tr>
      <tr>
        <td><strong>Ligand / metal binding</strong></td>
        <td>Holds on to a co-factor, metal ion, small molecule, DNA, another protein, etc.</td>
        <td>The two Histidines and one Glutamate that chelate zinc in carbonic anhydrase.</td>
      </tr>
      <tr>
        <td><strong>Structural scaffolding</strong></td>
        <td>Packs into the hydrophobic core, forms a disulphide bridge, or acts as a hinge/turn that lets the protein fold properly.</td>
        <td>A buried Leucine in a helix bundle; a Glycine in a β-turn.</td>
      </tr>
      <tr>
        <td><strong>Signal or 
        recognition patch</strong></td>
        <td>Forms part of a surface motif recognised by other molecules.</td>
        <td>The RGD motif (Arg-Gly-Asp) that integrins recognise.</td>
      </tr>
    </tbody>
  </table>

  <br>

  When the alignment is correct, every row in a given column traces back to the same position in the common ancestor of all those proteins, so the residues in that column:

  - Sit at the same 3-D spot in the folded structure, and
  - Usually have the same job (catalysis, binding, packing, flexibility, etc.).

  That's why you often see:

  - **Conserved columns**: the exact amino acid hardly changes because the role is intolerant to mutation (e.g., catalytic Ser).
  - **Variable but chemically similar columns**: the individual letters differ, but they share properties—say, all hydrophobic—because any residue that behaves the same is good enough for that structural slot.

  So in the sentence you quoted, "**amino acids performing the same functional/structural role**" means residues that occupy equivalent positions in homologous proteins and therefore contribute in equivalent ways to what the protein does or how it folds.
  Lining them up in the same column lets AlphaFold (or any analysis) read the evolutionary record of which jobs are critical and which can flex.

  > Example

  > TODO

</details>

<details class="details-frame" markdown="1">
  <summary>2. Encodes the MSA into a 3-D tensor called the "MSA representation" with axes (sequence, residue position, feature channels). </summary>

Each cell gets an embedding vector; AlphaFold treats the first row (your query) as a special "single representation" and keeps the rest as contextual rows.

This tensor is the main input to the Evoformer block.

</details>

### Pairing → initialization of contact representation

Build an initial L × L feature matrix that simply tells the network how far apart residues are in sequence.

Think of it as a very lightweight spatial prior (positional information) before any structural reasoning happens. 
Without this, the model would have to discover contacts from scratch, making learning far harder.


### Structure database search → templates

1. AlphaFold runs a sequence‑profile search (HHsearch/HHblits) against a curated, non‑redundant slice of the Protein Data Bank (often called "PDB 70/100" because similar chains are clustered at 70 % or 100 % identity). So the search key is still the amino‑acid sequence, not 3‑D shape similarity. The hit list may include very remote homologues whose sequences share only 15–20 % identity with yours, because profile–profile matches can detect distant evolutionary relationships.

<details class="details-frame" markdown="1">
  <summary>2. For the top ~20 alignments that pass an E‑value & coverage threshold, AlphaFold copies the template's backbone coordinates and encodes the inter‑residue distances/orientations into fixed‑size bins. </summary>

  This geometric information is what eventually feeds into the pair representation and gives the network concrete spatial hints, especially in regions where the multiple‑sequence alignment (MSA) is thin.

  > E-value (expect value)

  It measures a chance of meeting the sequence with aligment score bigger than given one in the whole database.
  <br>
  **Formal definition**

  $E = Kmne^{-\lambda S}$
  (the Karlin-Altschul equation), where S is the raw alignment score, m and n are the effective lengths of the query and database, and K, λ are statistical constants for the scoring matrix.

  <br>
  Lower is better.
  E = 1 × 10⁻³ means you'd expect one false-positive hit in a thousand equally good database searches.

  > Coverage threshold

  $ \text{Coverage} = \frac{\text{aligned residues}}{\text{query length}}$. 
  A spectacular E-value is useless if it aligns only a tiny fragment—that fragment may not be relevant to the overall fold you want to model.

  > Why both thresholds are used together

  1. E-value guards against false positives (bad statistics).
  2. Coverage guards against trivial positives (tiny true alignments that are uninformative for structure).
  
  Only hits that pass both filters are promoted to "template" status; their atomic coordinates are then converted to the distance-and-orientation bins that seed the pair representation.
</details>

> Why it matters

Many regions of a protein don't have strong co‑evolutionary signal in the MSA. A template provides direct geometric hints there (backbone distances, side‑chain orientations).
Templates help the network converge faster and are especially valuable for β‑sheet topologies that are hard to infer from sequence alone.
Inside the model these coordinate‑based features are converted to a 2‑D array (distance & orientation bins) and added into the initial pair representation (the blue square in the diagram).

### Evoformer

![folding]({{ "assets/notes-img/bio-informatics/protein sequence generation/jumper2021highly/4.png" | relative_url }}){:width="800px" .img-frame-black}

<div class="row">
  <div class="col-md-6">
    <img src='{{ "assets/notes-img/bio-informatics/protein sequence generation/jumper2021highly/5.png" | relative_url }}' alt="folding" class="img-fluid img-frame-black">
  </div>
  <div class="col-md-6">
    <img src='{{ "assets/notes-img/bio-informatics/protein sequence generation/jumper2021highly/7.png" | relative_url }}' alt="folding" class="img-fluid img-frame-black" style="width: 50%;">
  </div>
</div>


<details class="details-frame" markdown="1">
  <summary>1. Row‑wise gated self‑attention with pair bias. </summary>

  - What it does. For every single sequence (row) in the MSA, apply self‑attention along the residue axis.
  - Why "pair bias". The attention logits get an additive term derived from the current pair embedding for (i, j), so each sequence row is "aware" of the model's latest guess about how residues i and j interact.
  - Why "gated". A learned sigmoid gate scales the output, letting the network suppress noisy signals when the alignment at that row is poor.
  - Motivation. Propagate long‑range context within each homolog while conditioning on emerging structural hints.

<div class="algorithm-box" markdown="1">
<span class="caption">Algorithm: Row‑wise Gated Self‑Attention with Pair Bias (single head $h$)</span>

**Inputs:**
$$
\begin{align*}
    \mathbf{M} &\in \mathbb{R}^{S \times L \times d_m} && \text{(current MSA embeddings)}\\
    \mathbf{Z} &\in \mathbb{R}^{L \times L \times d_z} && \text{(current pair embeddings)}\\
    W^Q,\,W^K,\,W^V &\in \mathbb{R}^{d_h \times d_m} && \text{(head‑specific projections)}\\
    w^P &\in \mathbb{R}^{d_z} && \text{(projects pair features to scalar bias)}\\
    W^O &\in \mathbb{R}^{d_m \times (H d_h)} && \text{(output projection)}\\
    w^G &\in \mathbb{R}^{d_m},\; b^G \in \mathbb{R} && \text{(gate parameters)}
\end{align*}
$$

<ol class="algorithm-list">
  <li><b>for</b> $s = 1$ <b>to</b> $S$ <b>do</b> <span class="comment">% row loop (one sequence)</span>
    <ol>
      <li>$\mathbf{Q} \leftarrow W^Q \, \mathbf{M}_{s,:,:}^{\!\top}$ <span class="comment">% shape $d_h\times L$</span></li>
      <li>$\mathbf{K} \leftarrow W^K \, \mathbf{M}_{s,:,:}^{\!\top}$</li>
      <li>$\mathbf{V} \leftarrow W^V \, \mathbf{M}_{s,:,:}^{\!\top}$</li>
      <li><b>for</b> $i = 1$ <b>to</b> $L$ <b>do</b> <span class="comment">% token loop (residue $i$)</span>
        <ol>
          <li><b>for</b> $j = 1$ <b>to</b> $L$ <b>do</b>
            <ol>
              <li>$b_{ij} \leftarrow (w^P)^\top \mathbf{Z}_{ij}$ <span class="comment">% pair‑bias</span></li>
              <li>$e_{ij} \leftarrow \dfrac{\mathbf{Q}_{:,i}^{\!\top} \mathbf{K}_{:,j}}{\sqrt{d_h}} + b_{ij}$ <span class="comment">% logits</span></li>
            </ol>
          </li>
          <li><b>end for</b></li>
          <li>$\boldsymbol{\alpha}_{i} \leftarrow \operatorname{softmax}(\mathbf{e}_{i,:})$ <span class="comment">% over $j$</span></li>
          <li>$\mathbf{o}_i \leftarrow \displaystyle\sum_{j=1}^{L} \alpha_{ij}\, \mathbf{V}_{:,j}$ <span class="comment">% weighted sum</span></li>
          <li>$g_i \leftarrow \sigma\bigl((w^G)^\top\mathbf{M}_{s,i,:} + b^{G}\bigr)$ <span class="comment">% gate scalar</span></li>
          <li>$\widetilde{\mathbf{o}}_i \leftarrow g_i\,\mathbf{o}_i$ <span class="comment">% apply gate</span></li>
          <li>$\mathbf{M}_{s,i,:} \leftarrow \mathbf{M}_{s,i,:} + W^{O}\widetilde{\mathbf{o}}_i$ <span class="comment">% residual</span></li>
        </ol>
      </li>
      <li><b>end for</b></li>
    </ol>
  </li>
  <li><b>end for</b></li>
</ol>
</div>
    
</details>

<details class="details-frame" markdown="1">
  <summary>2. Column‑wise gated self‑attention. </summary>

  - What it does. Now fix a residue position r and attend down the column across the s sequences.
  - Motivation. Let each residue in the target sequence look at how that same position varies (or co‑varies) across evolution, capturing conservation and compensatory mutations.
  - Complexity trick. Row‑ and column‑wise attention factorise a huge 2‑D attention ( s r × s r ) into two linear passes, cutting time and memory by ~O(s r)
</details>

<details class="details-frame" markdown="1">
  <summary>3. Transition </summary>

  - A two‑layer feed‑forward network (ReLU / GLU) applied to every (s, r) token independently — the same role a "Transformer FFN" plays after its attention layer.
  - Purpose: mix features non‑linearly and give the model extra capacity. 

</details>

<details class="details-frame" markdown="1">
  <summary>4. Outer product mean </summary>

  - How. For each residue pair $(i, j)$, take the embedding vectors $v_i, v_j$ from every sequence, form the outer product $v_i \bigotimes v_j$ (a matrix), then average over the $s$ sequences.
  - What it produces. A rich co‑evolution feature saying "when residue $i$ mutates like this, residue $j$ mutates like that."
  - Where it goes. Added into the pair tensor, giving it evolutionary coupling signal.

</details>

<details class="details-frame" markdown="1">
  <summary>5. Triangle update using outgoing edges </summary>

  - View the pair tensor as a fully‑connected directed graph of residues; each edge $(i, j)$ stores an embedding $z_{i, j}$.
  - Update edge $(i, j)$ by multiplying / gating information that flows along the two edges $(i, k)$ and $(k, j)$ for every third residue $k$.
  - **Motivation**: Impose triangle‑inequality‑like constraints and capture three‑body interactions essential for 3‑D geometry.

  > Algorithm

  Let $\mathbf{Z} \in \mathbb{R}^{L \times L \times c}$ be the current pair tensor and let $i, j, k \in \{1, \dots, L\}$ index residues.

  **1. Project the two "arms" of each triangle**

  $$
  \begin{align*}
    & \mathbf{a}_{ik} &= W^{(a)} \mathbf{z}_{ik} \in \mathbb{R}^{c_m}, && \text{($i \rightarrow k$ arm)} \\
    & \mathbf{b}_{kj} &= W^{(b)} \mathbf{z}_{kj} \in \mathbb{R}^{c_m}, && \text{($k \rightarrow j$ arm)}
  \end{align*}
  $$

  - $W^{(a)}, W^{(b)} \in \mathbb{R}^{c_m \times c}$ are learned linear maps.
  - The projection width $c_m$ is usually $c/2$ so the outer product that follows keeps the channel count $\approx c$.

  **2. Form the triangle message for edge $(i, j)$:**
  $$
  \begin{align*}
    \quad \mathbf{t}_{ij} = \sum_{k=1}^{L} \frac{\mathbf{a}_{ik} \odot \mathbf{b}_{kj}}{\sqrt{L}} \in \mathbb{R}^{c_m}
  \end{align*}
  $$

  - Element-wise product $\odot$ mixes the two arms.
  - Division by $\sqrt{L}$ is a variance-stabilising scale (analogous to the $1/\sqrt{d}$ in self-attention).

  **3. Transform, gate and add as a residual**

  $$
  \begin{align*}
    & \mathbf{u}_{ij} = W^{(o)} \mathbf{t}_{ij} \in \mathbb{R}^{c} \\
    & g_{ij} = \sigma\left( (\mathbf{w}^{(g)})^\top \mathbf{z}_{ij} + b^{(g)} \right) \in (0, 1) \\
    & \mathbf{z}'_{ij} = \mathbf{z}_{ij} + g_{ij} \, \mathbf{u}_{ij} \\
  \end{align*}
  $$

  - $W^{(o)} \in \mathbb{R}^{c \times c_m}$ projects back to the original channel size.
  - $g_{ij}$ is a sigmoid gate computed from the pre-update edge; it lets the network down-weight noisy triangles.
  - The final line is a residual connection identical in spirit to the one in a Transformer block.

  **4. Vectorised Form**

  If we reshape $\mathbf{Z}$ to a 3-D tensor and use matrix multiplication, the whole operation is:
  
  $$
  \begin{align*}
     \mathbf{Z} \leftarrow \mathbf{Z} + \sigma\left( \langle \mathbf{Z}, \mathbf{w}^{(g)} \rangle + b^{(g)} \right) \odot W^{(o)} \left( \frac{\mathbf{A} \odot \mathbf{B}}{\sqrt{L}} \mathbf{1} \right)
  \end{align*}
  $$

  with $\mathbf{A} = W^{(a)} \mathbf{Z}, \quad \mathbf{B} = W^{(b)} \mathbf{Z}$ and the sum over $k$ is implemented by a batched tensor contraction.

  **Intuition**

  - The update $\textbf{looks along paths}$ $i \rightarrow k \rightarrow j$ ("out-going" edges from $i$).
  - It injects $\textit{multiplicative}$ three-body information, helping the network encode triangle inequalities and side-chain packing constraints—something plain pairwise couplings cannot capture.

</details>

<details class="details-frame" markdown="1">
  <summary>6. Triangle update – using incoming edges </summary>

  Same operation but centred on the opposite vertex, so the model sees both orientations of every residue triplet.

</details>

<details class="details-frame" markdown="1">
  <summary>7. Triangle self‑attention around starting node </summary>

  - Treat all edges that emanate from residue i as the "tokens" of an attention layer; use edge $(i, j)$ as the query and the set { $(i, k)$ } as keys/values.
  - Learns patterns like "edges from this residue fan out in a β‑sheet vs. coil". 
</details>

<details class="details-frame" markdown="1">
  <summary>8. Triangle self‑attention around ending node </summary>

  - Same again but for edges that end at residue j. Together, the two triangle‑attention layers let the model reason about local geometry from both directions. 
</details>

<details class="details-frame" markdown="1">
  <summary>9. Transition </summary>

  - A second feed‑forward network, now on every $(i, j)$ edge, to consolidate the information distilled by the triangle operations before looping to the next Evoformer block. 
</details>

### Structure Module

![folding]({{ "assets/notes-img/bio-informatics/protein sequence generation/jumper2021highly/6.png" | relative_url }}){:width="800px" .img-frame-black}

> Input

<table class="table">
  <thead>
    <tr>
      <th style="width: 25%;"><strong>Tensor</strong></th>
      <th style="width: 25%;"><strong>Shape</strong></th>
      <th style="width: 50%;"><strong>What it encodes</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Single representation <em>S</em></strong></td>
      <td><em>(L, c<sub>s</sub>)</em></td>
      <td>A per-residue feature vector distilled from the Evoformer.</td>
    </tr>
    <tr>
      <td><strong>Pair representation <em>Z</em></strong></td>
      <td><em>(L, L, c<sub>z</sub>)</em></td>
      <td>Rich geometry couplings (distances, orientations) between every residue pair.</td>
    </tr>
    <tr>
      <td><strong>Backbone frames <em>{R<sub>i</sub>, t<sub>i</sub>}</em></strong></td>
      <td><em>R<sub>i</sub> ∈ SO(3), t<sub>i</sub> ∈ ℝ³</em></td>
      <td>One local 3-D coordinate frame per residue. <strong>Initialised as identity + zero</strong>, i.e. every residue sits at the origin pointing the same way.</td>
    </tr>
  </tbody>
</table>
<p><em>All three are recycled from the previous passes of AlphaFold; on the very first pass the frames are just those trivial identities.</em></p>


<details class="details-frame" markdown="1">
  <summary>1. Invariant‑Point Attention (IPA) </summary>

  IPA is the SE(3)‑equivariant replacement for the dot‑product attention you know from Transformers.
  The scalars (queries Q, keys K, values V) work almost exactly as usual, but every token (here: every residue i) is also equipped with a handful of small 3‑D point clouds that move rigidly with the residue's current backbone frame.

  ![folding]({{ "assets/notes-img/bio-informatics/protein sequence generation/jumper2021highly/8.png" | relative_url }}){:width="800px" .img-frame-black}
</details>

<details class="details-frame" markdown="1">
  <summary>2. Rigid‑body update head </summary>

  A small MLP reads the new $S_i$ and predicts a **delta rotation** $\Delta R_i \in SO(3)$ (parameterised as a quaternion or axis--angle) and a **delta translation** $\Delta \mathbf{t}_i \in \mathbb{R}^3$.

$$
  \begin{align*}
    R_i \leftarrow \Delta R_i R_i, \qquad \mathbf{t}_i \leftarrow \mathbf{t}_i + \Delta \mathbf{t}_i
  \end{align*}
$$

Both deltas are passed through tanh-gated scales so that early iterations make only gentle moves; later passes can refine with larger steps.
</details>


<details class="details-frame" markdown="1">
  <summary>3. Side‑chain / χ‑angle head </summary>

  After the 8-th shared block, a final head predicts \textbf{torsion angles} $\chi_1, \chi_2, \ldots$ per residue.

  The angles are applied to idealised residue templates stored inside the network to produce \textbf{all heavy-atom coordinates} (including side chains and hydrogens).

</details>

 
> Why these ingredients?

<table class="table">
  <thead>
    <tr>
      <th style="width: 30%;"><strong>Component</strong></th>
      <th style="width: 70%;"><strong>Motivation</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Backbone frames</strong></td>
      <td>A rigid frame per residue turns geometry prediction into learning a smooth <em>field of rigid bodies</em>—exactly what protein backbones are.</td>
    </tr>
    <tr>
      <td><strong>Invariant-Point Attention</strong></td>
      <td>Extends vanilla attention so that <em>queries</em>, <em>keys</em>, <em>values</em>, and <em>pair bias</em> are all SE(3)-<em>equivariant</em>; lets the network reason directly in 3-D without breaking equivariance.</td>
    </tr>
    <tr>
      <td><strong>Rigid-body deltas</strong></td>
      <td>Predicting changes (Δ) rather than absolute coordinates keeps gradients stable and avoids large jumps.</td>
    </tr>
    <tr>
      <td><strong>Shared weights over 8 recycles</strong></td>
      <td>Memory-efficient unrolled optimisation: the same parameters are reused, but the inputs keep evolving, so the network effectively performs eight refinement steps.</td>
    </tr>
    <tr>
      <td><strong>Side-chain torsion head</strong></td>
      <td>Once backbone placement converges, side-chain packing is adjusted coherently with the learned χ-angle statistics from the PDB.</td>
    </tr>
  </tbody>
</table>

## Training Procedure

1. **"Seed" model:** Train AlphaFold only on experimental PDB structures for ≈ 300 k optimiser steps (just under half of the full run).
2. **One‑off inference sweep:** Run that seed model on all UniRef90 clusters that have no PDB hit (≈ 2 M clusters). Keep the single representative sequence for each cluster. This step gives you 3‑D coordinates, per‑residue pLDDT and a predicted TM‑score for every sequence.
Time cost: a few days on the same TPU pod that trains the network.
3. **Filtering / re‑weighting:** No hard pre‑training filter.
Instead, add the predictions to a TFRecord and store pLDDT as a training weight:
$w_i = (\frac{pLDDT_i}{100})^2$. 
High‑confidence residues (pLDDT≈90) get weight ≈ 0.8; low‑confidence ones (pLDDT≈30) get weight ≈ 0.1 and therefore contribute almost nothing to the loss.
4. **Merge & continue training:** Resume the same optimiser state. At every step sample 75 % PDB, 25 % self‑distilled crops. Continue for another ≈ 400 k steps. Because the optimiser is not re‑initialised, you can think of this as "fine‑tuning on a bigger, noisy set while still seeing clean PDB data every batch."


### On‑the‑fly feature generation 

- **MSA search:**
Uses jackhmmer (UniRef90), HHblits (BFD, Uniclust30), and MGnify metagenomes.
To avoid frozen data, 20 % of the time one of those databases is randomly dropped; another 20 % the MSA is subsampled or shuffled.

- **Template search:**
Runs HHsearch against a PDB70 library; at training time the top‑hit template is kept only with p = 0.75 and otherwise discarded so the model learns to work template‑free.

- **Random cropping:** If a protein exceeds 384 residues, a contiguous window of crop_size ∼ Uniform(256,384) is chosen; MSAs and templates are cropped consistently.

- **Masking loss (BERT‑style):** 15 % of MSA tokens are replaced by a mask character; the network must reconstruct them (auxiliary MSA‑loss).

### Recycling

```python
# --- 0. Feature construction -----------------------------------------
S0, Z0 = msa_and_template_features(query)     # shapes as above
frames0 = make_identity_frames(L)             # all residues at origin

# --- 1. Recycle loop ---------------------------------------------------
for t in range(N_RECYCLE):                    # 3 during training
    # Evoformer: updates tensor representations
    S_t, Z_t = Evoformer(S0, Z0)

    # Structure: predicts backbone + side chains
    frames_t, plddt_t, pae_t = Structure(S_t, Z_t)

    # Stop-grad on everything but keep the tensors
    S0 = stop_gradient(S_t)
    Z0 = stop_gradient(
            concat(Z_t, bins_from(frames_t), pae_t, plddt_t) )
```

Only the last pass's outputs receive gradient signals (so GPU/TPU memory is roughly that of one forward path).

<table class="table">
  <thead>
    <tr>
      <th><strong>Without recycling</strong></th>
      <th><strong>With recycling</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        Evoformer must convert raw MSA + template signal <strong>directly</strong> into an atomic model.
      </td>
      <td>
        First pass produces a <em>rough</em> fold; the next pass can treat that fold as an <em>extra template</em> and “zoom in” on clashes, mis-packed loops, β-sheet registry, etc.
      </td>
    </tr>
    <tr>
      <td>
        pLDDT/PAE confidence heads see only one sample.
      </td>
      <td>
        Confidence from pass <em>t</em> steers pass <em>t + 1</em>: the model learns to <em>trust</em> high-pLDDT regions and re-work low-pLDDT ones.
      </td>
    </tr>
    <tr>
      <td>
        CASP-level accuracy would require a deeper or larger net.
      </td>
      <td>
        Three passes of a 48-layer Evoformer + 8-block Structure module reach < 1Å GDT-TS error on many domains, with <strong>shared parameters</strong>.
      </td>
    </tr>
  </tbody>
</table>


### pLDDT Prediction

AlphaFold2 does not compute its confidence score after it has finished the 3-D model.
Instead, it trains the network to predict its own error at the same time that it predicts the
structure. The result is the predicted Local Distance Difference Test score (pLDDT)—a
per-residue estimate (0–100) of how well the Cα atoms of that residue will match the
unknown “true” structure.

After the structure module finishes its last recycling step it produces a
single representation vector $s_i$ for every residue i.
Those vectors feed a tiny multilayer perceptron (MLP):
LayerNorm → Linear → ReLU
Linear → ReLU (128 hidden channels)
Linear → Softmax → 50 logits
Each logit corresponds to an lDDT bin that is 2 units wide
(centres = 1, 3, 5 … 99).
The network therefore returns a probability distribution $p_i(b)$ over the 50 bins for every residue. 

For each residue $\text{pLDDT}_i = \sum_{b=1}^{50} p_i(b) v_b,$
where $v_b$ is the bin centre (again 1, 3, 5 … 99).

- **Ground truth:** 
For every PDB training chain with resolution 0.1–3.0 Å (the high-quality subset), the lDDT-Cα of the final AlphaFold output against the experimental structure is computed and discretised into the same 50 bins.

- **Loss:**
A simple per-residue cross-entropy between the predicted distribution $p_i$ and the one-hot target vector $y_i$.
The confidence loss is added to the main FAPE / torsion / distogram losses with its own weight.
The cross-entropy between the one-hot target and the logits does back-propagate through the MLP into the
single residue embedding and all weights below it (Structure-Module and Evoformer).
Thus the network is encouraged to encode “how right am I likely to be?” in that embedding.


> Why it works so well

- The single representation still encodes all the information that
controlled the structure prediction—template alignment quality, MSA depth,
invariant point attention activations, etc.—so the tiny head can learn a rich
error model without ever “seeing” the experimental target during inference.

### Finetuning Phase

Last 50 k steps switch to full‑length crops (no random window), enable violation loss, and freeze the MSA‑masking term to zero weight.
Fine‑tune re‑centres the model on physically plausible stereochemistry that the coarse cropping sometimes breaks.

### Losses

<table class="table">
  <thead>
    <tr>
      <th><strong>#</strong></th>
      <th><strong>Loss term</strong></th>
      <th><strong>What is actually computed</strong></th>
      <th><strong>Motivation</strong></th>
      <th><strong>When / how it is used</strong></th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td>1</td>
      <td><em>Frame-Aligned Point Error</em><br><strong>L<sub>FAPE</sub></strong></td>
      <td>Align each residue’s local frame (predicted ↔ true), square-clamped distance between corresponding atoms, then average.</td>
      <td>Primary geometry driver: forces correct local geometry while being invariant to global rigid moves; keeps gradients stable.</td>
      <td>Full 7.5 M steps, highest weight (initial ≈ 1.0, then uncertainty-learned).</td>
    </tr>

    <tr>
      <td>2</td>
      <td><em>Distogram</em><br><strong>L<sub>dist</sub></strong></td>
      <td>64-bin cross-entropy on Cβ–Cβ (Cα for Gly) distance distributions.</td>
      <td>Guides pair representation; stabilises early training and enables self-distillation.</td>
      <td>Entire training run; weight ≈ 0.3–0.5.</td>
    </tr>

    <tr>
      <td>3</td>
      <td><em>Backbone &amp; side-chain torsion</em><br><strong>L<sub>χ</sub></strong></td>
      <td>L2 loss on sine/cosine of φ, ψ, and χ<sub>1–4</sub> angles.</td>
      <td>Enforces stereochemistry; distances alone cannot lock correct rotamers or Ramachandran regions.</td>
      <td>Applied layer-wise + final; moderate weight ≈ 0.3.</td>
    </tr>

    <tr>
      <td>4</td>
      <td><em>Masked-MSA reconstruction</em><br><strong>L<sub>msa</sub></strong></td>
      <td>BERT-style token cross-entropy on 15 % masked MSA positions.</td>
      <td>Self-supervision on evolutionary data; regularises and enriches sequence embeddings.</td>
      <td>Whole training; small weight ≈ 0.1.</td>
    </tr>

    <tr>
      <td>5</td>
      <td><em>Confidence (pLDDT &amp; pTM)</em><br><strong>L<sub>conf</sub></strong></td>
      <td>Cross-entropy (pLDDT) &amp; MSE (pTM) against on-the-fly self-targets.</td>
      <td>Teaches the model to calibrate its own error without pulling atoms directly.</td>
      <td>All steps; tiny weight ≈ 0.01 (∼1 % of FAPE).</td>
    </tr>

    <tr>
      <td>6</td>
      <td><em>Violation</em><br><strong>L<sub>viol</sub></strong></td>
      <td>Penalties for bond/angle outliers, clashes, Ramachandran violations.</td>
      <td>Injects basic chemistry; catches illegal geometries not covered by FAPE.</td>
      <td>Only last ~50 k fine-tune steps; weight ≈ 0.3.</td>
    </tr>
  </tbody>
</table>

## Ablation

![folding]({{ "assets/notes-img/bio-informatics/protein sequence generation/jumper2021highly/9.png" | relative_url }}){:width="800px" .img-frame-black}

<table class="table">
  <thead>
    <tr>
      <th><strong>Ablation experiment</strong></th>
      <th><strong>Δ GDT<sub>TS</sub><br>(CASP14)</strong></th>
      <th><strong>Δ lDDT‑Cα<br>(PDB set)</strong></th>
      <th><strong>Interpretation of the drop / gain</strong></th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td>With&nbsp;<em>self‑distillation</em> data added</td>
      <td>≈ +1.5</td>
      <td>≈ +0.2</td>
      <td>Extra pseudo‑labels expose far more sequence diversity, giving a modest but consistent boost.</td>
    </tr>

    <tr>
      <td>No templates</td>
      <td>≈ ‑2</td>
      <td>≈ ‑0.3</td>
      <td>Template signal helps only occasionally; AF2 is largely <em>de&nbsp;novo</em>, so the impact is small.</td>
    </tr>

    <tr>
      <td>No auxiliary <em>distogram</em> head</td>
      <td>≈ ‑3 – 4</td>
      <td>≈ ‑0.5</td>
      <td>Pair representation loses explicit distance supervision, hurting contact inference.</td>
    </tr>

    <tr>
      <td>No <em>raw MSA</em> (use pair‑frequency matrix only)</td>
      <td>≈ ‑6</td>
      <td>≈ ‑1.0</td>
      <td>Removing higher‑order co‑evolution erases much of the contact signal encoded in the MSA.</td>
    </tr>

    <tr>
      <td>No <em>Invariant Point Attention</em> (direct projection)</td>
      <td>≈ ‑7</td>
      <td>≈ ‑1.5</td>
      <td>Loses SE(3)‑equivariance; the structure module can’t reason about rotations/translations.</td>
    </tr>

    <tr>
      <td>No auxiliary <em>masked‑MSA</em> head</td>
      <td>≈ ‑3.5</td>
      <td>≈ ‑0.7</td>
      <td>Row‑wise representation is less regularised; long or sparse MSAs suffer most.</td>
    </tr>

    <tr>
      <td>No <em>recycling</em></td>
      <td>≈ ‑6</td>
      <td>≈ ‑1.0</td>
      <td>A single pass cannot iron out clashes or register errors; iterative refinement is crucial.</td>
    </tr>

    <tr>
      <td>No <em>triangle updates, biasing or gating</em><br>(use axial attention)</td>
      <td>≈ ‑7</td>
      <td>≈ ‑1.3</td>
      <td>Weakens long‑range geometric reasoning that holds β‑sheets and domain interfaces together.</td>
    </tr>

    <tr>
      <td>No <em>end‑to‑end structure gradients</em><br>(keep only auxiliary heads)</td>
      <td>≈ ‑8</td>
      <td>≈ ‑2.0</td>
      <td>Blocking coordinate‑error back‑prop prevents Evoformer from learning geometry‑aware features.</td>
    </tr>

    <tr>
      <td>No IPA <strong>and</strong> no recycling</td>
      <td>≈ ‑18</td>
      <td>≈ ‑3.5</td>
      <td>Combining the two worst ablations collapses the model, confirming they are foundational.</td>
    </tr>
  </tbody>
</table>
