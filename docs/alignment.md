# CLASP: Generating Similarity Matrices

This document explains how to generate similarity matrices between PDB structures, amino acid sequences, and textual descriptions using pretrained CLASP components.

---

## 1. Prerequisites

Ensure you have the following input files available (see [preprocessing documentation](preprocessing.md) for details on how to create these):

### Required Inputs

| Type                         | Format  | Description                                                                                 |
| ---------------------------- | ------- | ------------------------------------------------------------------------------------------- |
| Amino acid embeddings        | `.h5`   | HDF5 file with UniProt accessions as keys and 1024D vectors as values (e.g., ProtT5 output) |
| Description embeddings       | `.h5`   | HDF5 file with UniProt accessions as keys and 1024D vectors as values (e.g., BioGPT output) |
| PDB structure graphs         | `.pt`   | Pickled Torch dictionary mapping `<upkb_ac>-<pdb_id>` to PyG `Data` objects                 |
| Target ID list               | `.json` | JSON dictionary with ordered lists of keys: `"pdb_ids"`, `"aas_ids"`, and `"desc_ids"`      |
| PDB Encoder model checkpoint | `.pt`   | Trained `CLASPEncoder` model weights                                                        |
| Alignment model checkpoint   | `.pt`   | Trained `CLASPAlignment` model weights (projecting all three modalities into shared space)  |

---

## 2. Running the Similarity Script

Run the matrix generation script with the following command:

```bash
python generate_similarity_matrices.py \
  --aas_embeddings_file path/to/aas_embeddings.h5 \
  --desc_embeddings_file path/to/desc_embeddings.h5 \
  --preprocessed_pdb_file path/to/preprocessed_pdb.pt \
  --encoder_model_path path/to/final_pdb_encoder.pt \
  --alignment_model_path path/to/final_alignment.pt \
  --target_file path/to/target_ids.json \
  --structure_to_sequence_matrix True \
  --structure_to_description_matrix True \
  --sequence_to_description_matrix True \
  --output_dir path/to/output \
  --device cuda
```


python generate_similarity_matrices.py \
  --aas_embeddings_file /projects/nbolo/CLIENT/clasp/data/amino_acid_embeddings.h5\
  --desc_embeddings_file /projects/nbolo/CLIENT/clasp/data/description_embeddings.h5 \
  --preprocessed_pdb_file /projects/nbolo/CLIENT/clasp/data/processed_pdb_data.pt\
  --encoder_model_path /projects/nbolo/CLIENT/clasp/final_models/seed_26855092/clasp_pdb_encoder.pt \
  --alignment_model_path /projects/nbolo/CLIENT/clasp/final_models/seed_26855092/clasp_alignment.pt \
  --target_file /projects/nbolo/CLIENT/clasp/data/query.jsonl

### Arguments

| Argument                            | Required | Description                                                                    |
| ----------------------------------- | -------- | ------------------------------------------------------------------------------ |
| `--aas_embeddings_file`             | ✔        | Path to `.h5` file containing amino acid sequence embeddings                   |
| `--desc_embeddings_file`            | ✔        | Path to `.h5` file containing protein description embeddings                   |
| `--preprocessed_pdb_file`           | ✔        | Path to `.pt` file containing PDB structure graphs                             |
| `--encoder_model_path`              | ✔        | Path to trained `CLASPEncoder` model checkpoint                                |
| `--alignment_model_path`            | ✔        | Path to trained `CLASPAlignment` model checkpoint                              |
| `--target_file`                     | ✔        | JSON file containing lists of IDs to include in each modality                  |
| `--structure_to_sequence_matrix`    | ✖        | If True, computes PDB-to-AAS similarity matrix (default: True)                 |
| `--structure_to_description_matrix` | ✖        | If True, computes PDB-to-description similarity matrix (default: True)         |
| `--sequence_to_description_matrix`  | ✖        | If True, computes AAS-to-description similarity matrix (default: True)         |
| `--output_dir`                      | ✖        | Directory to save the similarity matrices and projections (default: `output/`) |
| `--seed`                            | ✖        | Random seed (default: `42`)                                                    |
| `--device`                          | ✖        | Device to use: `cuda` or `cpu` (default: `cuda` if available)                  |

---

## 3. Target File Format

The `--target_file` must be a JSON file with the following keys:

```json
{
  "pdb_ids": ["Q9FGK0-7ARB", "Q6P4A7-8FHE", ...],
  "aas_ids": ["Q9FGK0", "Q6P4A7", ...],
  "desc_ids": ["Q9FGK0", "Q6P4A7", ...]
}
```

The script uses these ordered lists to index embeddings and compute matrix products accordingly.

---

## 4. Output

The following files will be saved in the specified `--output_dir`:

| Filename                      | Shape      | Description                                               |
| ----------------------------- | ---------- | --------------------------------------------------------- |
| `pdb_proj.pt`                 | `(Np, D)`  | Projected structure embeddings                            |
| `aas_proj.pt`                 | `(Na, D)`  | Projected amino acid sequence embeddings                  |
| `desc_proj.pt`                | `(Nd, D)`  | Projected description embeddings                          |
| `structure_to_sequence.pt`    | `(Np, Na)` | Similarity matrix: structure-to-sequence (dot product)    |
| `structure_to_description.pt` | `(Np, Nd)` | Similarity matrix: structure-to-description (dot product) |
| `sequence_to_description.pt`  | `(Na, Nd)` | Similarity matrix: sequence-to-description (dot product)  |

Where `Np`, `Na`, and `Nd` are the number of PDB, AAS, and description IDs, respectively.

All matrices are stored as PyTorch tensors in `.pt` format.

---

## 5. Notes

* The script will raise a `KeyError` if any ID listed in the target file is missing from the embeddings or PDB data.
* The encoder and alignment models must have been trained using compatible dimensions (e.g., `embed_dim=512`).
* Matrix multiplications are performed using dot products (i.e., raw similarity scores).

---

For further details on training the models, see [CLASP Training](train_clasp.md).
