# CLASP Training

This document explains how to train the CLASP tri-modal contrastive model using your preprocessed data.

---

## 1. Prerequisites

Ensure you have the following files (see [preprocessing documentation](preprocessing.md) for details on how to obtain these):

### Required Inputs

| Type                   | Format   | Description                                                                                       |
| ---------------------- | -------- | ------------------------------------------------------------------------------------------------- |
| Amino acid embeddings  | `.h5`    | HDF5 file with UniProt accessions as keys and 1024D vectors as values (e.g., from ProtT5)         |
| Description embeddings | `.h5`    | HDF5 file with UniProt accessions as keys and 1024D vectors as values (e.g., from BioGPT)         |
| PDB structure graphs   | `.pt`    | Torch dictionary with keys `<upkb_ac>-<pdb_id>` and values as `torch_geometric.data.Data` objects |
| Training pairs         | `.jsonl` | Files with `{"upkb_ac": ..., "pdb_id": ...}` format                                               |
| Validation pairs       | `.jsonl` | Same format as above                                                                              |

---

## 2. Running the Training Script

You can run the training script from the command line as follows:

```bash
python train_clasp.py \
  --aas_embeddings_file path/to/amino_acid_embeddings.h5 \
  --desc_embeddings_file path/to/desc_embeddings.h5 \
  --preprocessed_pdb_file path/to/processed_pdb_data.pt \
  --processed_data_dir path/to/pairs_directory \
  --checkpoint_dir path/to/checkpoints \
  --output_dir path/to/final_models \
  --seed 42 \
  --device cuda
```

### Arguments

| Argument                  | Required | Description                                                        |
| ------------------------- | -------- | ------------------------------------------------------------------ |
| `--aas_embeddings_file`   | ✔        | Path to `.h5` file with amino acid embeddings                      |
| `--desc_embeddings_file`  | ✔        | Path to `.h5` file with description embeddings                     |
| `--preprocessed_pdb_file` | ✔        | Path to `.pt` file with PDB graph data                             |
| `--processed_data_dir`    | ✔        | Directory containing `train_pairs_*.jsonl` and `val_pairs.jsonl`   |
| `--checkpoint_dir`        | ✖        | Directory to save best model checkpoints (default: `checkpoints/`) |
| `--output_dir`            | ✖        | Directory to save final trained models (default: `final_models/`)  |
| `--seed`                  | ✖        | Random seed (default: `42`)                                        |
| `--device`                | ✖        | `cuda` or `cpu` (default: uses `cuda` if available)                |

---

## 3. Expected Directory Structure

Your `processed_data_dir` should contain:

```text
train_pairs_a.jsonl
train_pairs_b.jsonl
train_pairs_c.jsonl
train_pairs_d.jsonl
train_pairs_e.jsonl
val_pairs.jsonl
```

Each `.jsonl` file should contain lines like:

```json
{"upkb_ac": "Q9FGK0", "pdb_id": "Q9FGK0-7ARB"}
```

---

## 4. Output

After training, the script will:

* Save the best model during training to:

  * `checkpoint_dir/best_3dclip_model.pt`
  * `checkpoint_dir/best_pdb_encoder_model.pt`
* Save the final models to:

  * `output_dir/final_alignment.pt`
  * `output_dir/final_pdb_encoder.pt`
* Log training and validation loss at each epoch

Early stopping will occur if validation loss does not improve for 40 epochs (can be tuned within the script).


