### To do list in the repository (will be deleted once completed)

- Fill in missing links in the data prep docs
- Write and test download script
- Add citation to readme
- Finish readme 
- Make sure env set up is seamless
- Test all code on MacOS
- Get external lab member to review code and docs
- License the code


----
----
----

# CLASP: Contrastive Learning of Amino acid, Structure, and Protein description

**CLASP** is a tri-modal contrastive learning framework for unified representation of protein structure, sequence, and description. It enables downstream applications such as cross-modal retrieval, similarity scoring, and zero-shot classification by learning a shared embedding space across all three modalities.


## Table of contents
1. [Overview of the CLASP framework](#overview-of-the-clasp-framework) 
2. [Repository structure](#repository-structure)
3. [Environment setup](#environment-setup)
4. [Documentation and usage](#documentation-and-usage)
5. [Citation](#citation)

## Overview of the CLASP framework

CLASP (Contrastive Learning of Amino acid, Structure, and Protein description) is a tri-modal contrastive learning framework designed to unify protein structure (as PDB files), amino acid sequence, and natural language description into a shared embedding space. By learning aligned representations across these biologically complementary views, CLASP supports downstream applications such as cross-modal retrieval, similarity scoring, and zero-shot classification without requiring task-specific supervision.

Each modality is first encoded independently: protein structures are converted into residue-level graphs using [Graphein](https://graphein.ai/) and embedded via an [E(3)-invariant graph neural network (EGNN)](https://doi.org/10.48550/arXiv.2102.09844), amino acid sequences are embedded using pretrained language models like [ProtT5](https://doi.org/10.1109/TPAMI.2021.3095381), and textual descriptions are embedded using biomedical large language models such as [BioGPT](https://doi.org/10.1093/bib/bbac409). Lightweight projection heads then map all three modalities into a common vector space. CLASP is trained using a tri-modal contrastive loss that simultaneously encourages alignment between matching structure-sequence, structure-description, and sequence-description pairs. 

<p align="center">
  <img src="assets/clasp_pipeline.png" alt="CLASP Pipeline" width="720"/>
</p>



## Repository structure

```
CLASP/
├── assets/                # directory for asset files
├── docs/                 
│   ├── data_preparation.md
│   ├── inference_utilities.md
│   ├── training_clasp.md
│   └── README.md
├── src/                   
    ├── models/            # model definitions for CLASP
│   ├── utils/             # utility functions for data processing and model training
│   ├── compute_similarity_matrices.py
│   ├── get_quick_similarity_scores.py
│   ├── preprocess_pdb_graphs.py
│   ├── rank_amino_acids_by_description.py
│   ├── train_clasp.py     
├── .gitignore             
├── environment.yml       
└── README.md      
```        


## Environment setup

TBD


## Documentation and usage

| File                                                    | Description                                    |
| ------------------------------------------------------- | ---------------------------------------------- |
| [`data_preparation.md`](docs/data_preparation.md)       | Instructions for obtaining and formatting data |
| [`train_clasp.md`](docs/train_clasp.md)                 | Training setup and script usage                |
| [`inference_utilities.md`](docs/inference_utilities.md) | Inference and retrieval utilities              |


## Citation

TBD

