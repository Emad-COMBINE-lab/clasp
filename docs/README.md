# CLASP Documentation

This directory contains all documentation related to preparing data, training models, evaluating performance, and performing inference using the CLASP framework for multi-modal protein representation learning.

## Contents

| Document                                           | Purpose                                                                           |
| -------------------------------------------------- | --------------------------------------------------------------------------------- |
| [`data_preparation.md`](data_preparation.md)       | Guide to downloading, generating, and formatting input data                       |
| [`training_clasp.md`](training_clasp.md)           | Instructions for training the CLASP model                                         |
| [`evaluation.md`](evaluation.md)                   | Guide for running zero-shot classification evaluation on trained CLASP models     |
| [`inference_utilities.md`](inference_utilities.md) | Tools for projecting embeddings, computing similarities, and retrieval            |

## Quick start

1. **Prepare Data**  
   Follow [`data_preparation.md`](data_preparation.md) to generate or download embeddings, PDB graphs, and train/val/test sets.

2. **Train CLASP**  
   Use [`training_clasp.md`](training_clasp.md) to train your own CLASP model from scratch, or download pre-trained weights.

3. **Evaluate Model**  
   Follow [`evaluation.md`](evaluation.md) to run zero-shot classification across PDB–AAS, PDB–DESC, and AAS–DESC tasks, selecting thresholds from validation splits.

4. **Run Inference**  
   Follow [`inference_utilities.md`](inference_utilities.md) to:
   * Compute similarity matrices and obtain projected embeddings
   * Calculate quick pairwise similarity scores
   * Retrieve amino acids based on textual queries
