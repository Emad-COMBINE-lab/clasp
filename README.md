# CLASP: Contrastive Learning of Amino acid, Structure, and Protein description

**CLASP** is a tri-modal contrastive learning framework for unified representation of protein structure, sequence, and description. It enables downstream applications such as cross-modal retrieval, similarity scoring, and zero-shot classification by learning a shared embedding space across all three modalities.

<p align="center">
  <img src="assets/clasp_pipeline.pdf" alt="CLASP Pipeline" width="720"/>
</p>

---

## Repository Structure

```
CLASP/
├── src/                   # Core source code
├── data/                  # Example data files and mappings
├── final_models/          # Trained checkpoints
├── docs/                  # Documentation
│   ├── data_preparation.md
│   ├── train_clasp.md
│   ├── inference_utilities.md
│   └── assets/clasp_pipeline.png
├── requirements.txt
└── README.md
```

---

## Documentation

| File                                                    | Description                                    |
| ------------------------------------------------------- | ---------------------------------------------- |
| [`data_preparation.md`](docs/data_preparation.md)       | Instructions for obtaining and formatting data |
| [`train_clasp.md`](docs/train_clasp.md)                 | Training setup and script usage                |
| [`inference_utilities.md`](docs/inference_utilities.md) | Inference and retrieval utilities              |

---

## Citation

TBD
---








----
----
----
## Ignore everything below this line

conda env create -f environment.yml
conda activate claspenv
<!-- pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cpu.html -->
conda install pytorch=2.6.0 torchvision torchaudio pytorch-cuda=12.2 -c pytorch -c nvidia -y
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu122.html


### TO DO LIST IN REPO

[ ] Fill in missing links in the data prep docs
[ ] Write download script
[ ] Finish readme 
[ ] Make sure env set up is seamless
[ ] Test all code on MacOS
[ ] Get external lab member to review code and docs