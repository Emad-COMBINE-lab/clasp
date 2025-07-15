import os
import argparse
import h5py
import torch
from models import CLASPAlignment, CLASPEncoder
from utils import create_clip_model_with_random_weights
import json

import math
from pathlib import Path
import torch


def _stack_in_order(ids, emb_dict, device):
    """Utility: stack embeddings following the given order."""
    missing = [i for i in ids if i not in emb_dict]
    if missing:
        raise KeyError(f"Missing embeddings for ids: {missing[:5]} …")
    return torch.stack([torch.as_tensor(emb_dict[i]) for i in ids]).to(device)


def generate_similarity_matrices(
    aas_embeddings,
    desc_embeddings,
    pdb_data,
    pdb_encoder,
    alignment_model,
    ordered_pbd_ids,
    ordered_aas_ids,
    ordered_desc_ids,
    structure_to_sequence_matrix,
    structure_to_description_matrix,
    sequence_to_description_matrix,
    output_dir,
    device,
):
    """
    Generate matrices
    """
    pdb_batch_size = 64

    with torch.no_grad():
        # Projected PDB embeddings
        if len(ordered_pbd_ids) != 0:
            proj_pdb_chunks = []
            n_batches = math.ceil(len(ordered_pbd_ids) / pdb_batch_size)
            for b in range(n_batches):
                batch_ids = ordered_pbd_ids[
                    b * pdb_batch_size : (b + 1) * pdb_batch_size
                ]
                batch_graphs = {
                    pid: pdb_data[pid].to(device)
                    for pid in batch_ids
                    if pid in pdb_data
                }
                if len(batch_graphs) != len(batch_ids):
                    missing = set(batch_ids) - batch_graphs.keys()
                    raise KeyError(f"PDB graphs missing for ids: {missing}")
                pdb_embs = pdb_encoder(batch_graphs)
                proj_pdb = alignment_model.get_pdb_projection(pdb_embs)
                proj_pdb_chunks.append(proj_pdb)
            pdb_proj = torch.cat(proj_pdb_chunks, dim=0)

        # Projected AAS embeddings
        if len(ordered_aas_ids) != 0:
            aas_raw = _stack_in_order(ordered_aas_ids, aas_embeddings, device)
            aas_proj = alignment_model.get_aas_projection(aas_raw)

        # Projected descriptor embeddings
        if len(ordered_desc_ids) != 0:
            desc_raw = _stack_in_order(ordered_desc_ids, desc_embeddings, device)
            desc_proj = alignment_model.get_desc_projection(desc_raw)

        # Save projections
        torch.save(pdb_proj.cpu(), Path(output_dir) / "pdb_proj.pt")
        torch.save(aas_proj.cpu(), Path(output_dir) / "aas_proj.pt")
        torch.save(desc_proj.cpu(), Path(output_dir) / "desc_proj.pt")

        # Compute & store similarity matrices
        if structure_to_sequence_matrix:
            sim_pdb_aas = pdb_proj @ aas_proj.T  # (Np, Na)
            torch.save(
                sim_pdb_aas.cpu(),
                Path(output_dir) / "structure_to_sequence.pt",
            )

        if structure_to_description_matrix:
            sim_pdb_desc = pdb_proj @ desc_proj.T  # (Np, Nd)
            torch.save(
                sim_pdb_desc.cpu(),
                Path(output_dir) / "structure_to_description.pt",
            )

        if sequence_to_description_matrix:
            sim_aas_desc = aas_proj @ desc_proj.T  # (Na, Nd)
            torch.save(
                sim_aas_desc.cpu(),
                Path(output_dir) / "sequence_to_description.pt",
            )


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description="Train CLASP")

    # required args
    parser.add_argument("--aas_embeddings_file", type=str, required=True)
    parser.add_argument("--desc_embeddings_file", type=str, required=True)
    parser.add_argument("--preprocessed_pdb_file", type=str, required=True)

    parser.add_argument("--encoder_model_path", type=str, required=True)
    parser.add_argument("--alignment_model_path", type=str, required=True)

    parser.add_argument("--target_file", type=str, required=True)

    # optional args
    parser.add_argument("--structure_to_sequence_matrix", type=bool, default=True)
    parser.add_argument("--structure_to_description_matrix", type=bool, default=True)
    parser.add_argument("--sequence_to_description_matrix", type=bool, default=True)

    parser.add_argument("--output_dir", type=str, default="output")

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    # check if paths exist
    if not os.path.exists(args.aas_embeddings_file):
        raise FileNotFoundError(
            f"Amino acid embeddings file not found: {args.aas_embeddings_file}"
        )
    if not os.path.exists(args.desc_embeddings_file):
        raise FileNotFoundError(
            f"Descriptor embeddings file not found: {args.desc_embeddings_file}"
        )
    if not os.path.exists(args.preprocessed_pdb_file):
        raise FileNotFoundError(
            f"Preprocessed PDB file not found: {args.preprocessed_pdb_file}"
        )

    if not os.path.exists(args.encoder_model_path):
        raise FileNotFoundError(
            f"Encoder model path not found: {args.encoder_model_path}"
        )
    if not os.path.exists(args.alignment_model_path):
        raise FileNotFoundError(
            f"Alignment model path not found: {args.alignment_model_path}"
        )
    if not os.path.exists(args.target_file):
        raise FileNotFoundError(f"Target file not found: {args.target_file}")

    # create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = args.output_dir

    # set device and seed
    if args.device not in ["cpu", "cuda"]:
        raise ValueError("Device must be 'cpu' or 'cuda'")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this machine, use 'cpu' instead")
    device = torch.device(args.device)
    seed = args.seed

    # ensure data is in correct format
    try:
        with h5py.File(args.aas_embeddings_file, "r") as f:
            amino_acid_embeddings = {k: f[k][()] for k in f.keys()}
    except Exception as e:
        raise ValueError(f"Error loading amino acid embeddings: {e}")

    try:
        with h5py.File(args.desc_embeddings_file, "r") as f:
            desc_embeddings = {k: f[k][()] for k in f.keys()}
    except Exception as e:
        raise ValueError(f"Error loading descriptor embeddings: {e}")

    try:
        pdb_data = torch.load(args.preprocessed_pdb_file)
    except Exception as e:
        raise ValueError(f"Error loading preprocessed PDB file: {e}")

    try:
        with open(args.target_file, "r") as f:
            target_data_dict = json.load(f)
        ordered_pbd_ids = []
        ordered_aas_ids = []
        ordered_desc_ids = []

        if args.structure_to_sequence_matrix or args.structure_to_description_matrix:
            if "pdb_ids" in target_data_dict:
                ordered_pbd_ids = list(target_data_dict["pdb_ids"])
            else:
                raise KeyError("Target data file must contain 'pdb_ids' key")
        if args.structure_to_sequence_matrix or args.sequence_to_description_matrix:
            if "aas_ids" in target_data_dict:
                ordered_aas_ids = list(target_data_dict["aas_ids"])
            else:
                raise KeyError("Target data file must contain 'aas_ids' key")
        if args.structure_to_description_matrix or args.sequence_to_description_matrix:
            if "desc_ids" in target_data_dict:
                ordered_desc_ids = list(target_data_dict["desc_ids"])
            else:
                raise KeyError("Target data file must contain 'desc_ids' key")

    except Exception as e:
        raise ValueError(f"Error loading target data file: {e}")

    # ensure models are in correct format
    try:
        pdb_encoder = CLASPEncoder(
            in_channels=7, hidden_channels=16, final_embedding_size=512, target_size=512
        ).to(device)
        pdb_encoder.load_state_dict(
            torch.load(args.encoder_model_path, map_location=device)
        )
        pdb_encoder.eval()
    except Exception as e:
        raise ValueError(f"Error loading encoder model: {e}")

    try:
        base_clip_model = create_clip_model_with_random_weights(
            "ViT-B/32", device=device
        )
        alignment_model = CLASPAlignment(base_clip_model, embed_dim=512).to(device)
        alignment_model.load_state_dict(
            torch.load(args.alignment_model_path, map_location=device)
        )
        alignment_model.eval()
    except Exception as e:
        raise ValueError(f"Error loading alignment model: {e}")

    # generate similarity matrices
    generate_similarity_matrices(
        aas_embeddings=amino_acid_embeddings,
        desc_embeddings=desc_embeddings,
        pdb_data=pdb_data,
        pdb_encoder=pdb_encoder,
        alignment_model=alignment_model,
        ordered_pbd_ids=ordered_pbd_ids,
        ordered_aas_ids=ordered_aas_ids,
        ordered_desc_ids=ordered_desc_ids,
        structure_to_sequence_matrix=args.structure_to_sequence_matrix,
        structure_to_description_matrix=args.structure_to_description_matrix,
        sequence_to_description_matrix=args.sequence_to_description_matrix,
        output_dir=output_dir,
        device=device,
    )
