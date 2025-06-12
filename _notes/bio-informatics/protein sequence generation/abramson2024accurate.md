---
layout: note
title: "Accurate structure prediction of biomolecular interactions with AlphaFold 3"
date: 2024-04-29
tags: [alphafold, proteins, cogeneration]
link_pdf: "https://www.nature.com/articles/s41586-024-07487-w"
code: "https://github.com/google-deepmind/alphafold3"

bibtex: |-
  @article{abramson2024accurate,
    title={Accurate structure prediction of biomolecular interactions with AlphaFold 3},
    author={Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans, Richard and Green, Tim and Pritzel, Alexander and Ronneberger, Olaf and Willmore, Lindsay and Ballard, Andrew J and Bambrick, Joshua and others},
    journal={Nature},
    volume={630},
    number={8016},
    pages={493--500},
    year={2024},
    publisher={Nature Publishing Group UK London}
  }
---

## Technical Details
---

![folding]({{ "assets/notes-img/bio-informatics/protein sequence generation/abramson2024accurate/10.png" | relative_url }}){:width="800px" .img-frame-black}


<details class="details-frame" markdown="1">
  <summary>1. Pairformer </summary>

  What it is: a streamlined successor to the Evoformer. It iterates attention and triangular updates over the single and pair features but no longer keeps an MSA axis.
  Why 48 blocks? Empirically, this depth was the sweet spot in AF2 for capturing long‑range geometry; AF3 preserves it but spends the saved MSA FLOPs on more tokens (all atoms, ligands, nucleic acids). 

</details>

<details class="details-frame" markdown="1">
  <summary>2. Recycling loop </summary>

  As in AlphaFold 2 recycling loop is applied during training (with stop gradient) and inference. Typically four recycles are used.
</details>

<details class="details-frame" markdown="1">
  <summary>3. Recycling loop </summary>

  ![folding]({{ "assets/notes-img/bio-informatics/protein sequence generation/abramson2024accurate/11.png" | relative_url }}){:width="800px" .img-frame-black}


  
</details>




