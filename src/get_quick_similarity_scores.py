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


def compute_and_print_quick_similarity_scores(
    pdb_id,
    aas_id,
    desc_id,
    pdb_data,
    aas_embeddings,
    desc_embeddings,
    pdb_encoder,
    alignment_model,
    device,
):
    """
    Compute and print quick similarity scores for debugging.
    """
    # get projections
    with torch.no_grad():
        # PDBs
        pdb_proj_emb = None
        if pdb_id is not None:
            pdb_data_item = pdb_data.get(pdb_id)
            if pdb_data_item is None:
                raise KeyError(f"PDB data missing for id: {pdb_id}")
            pdb_data_item = pdb_data_item.to(device)
            pdb_emb = pdb_encoder(pdb_data_item)
            pdb_emb_tensor = torch.tensor(pdb_emb, dtype=torch.float32).to(device)
            pdb_proj_emb = alignment_model.get_pdb_projection(pdb_emb_tensor)

        # AASs
        aas_proj_emb = None
        if aas_id is not None:
            aas_raw_emb = aas_embeddings.get(aas_id)
            if aas_raw_emb is None:
                raise KeyError(f"Amino acid embedding missing for id: {aas_id}")
            aas_raw_emb_tensor = torch.tensor(aas_raw_emb, dtype=torch.float32).to(
                device
            )
            aas_proj_emb = alignment_model.get_aas_projection(aas_raw_emb_tensor)

        # DESCs
        desc_proj_emb = None
        if desc_id is not None:
            desc_raw_emb = desc_embeddings.get(desc_id)
            if desc_raw_emb is None:
                raise KeyError(f"Descriptor embedding missing for id: {desc_id}")
            desc_raw_emb_tensor = torch.tensor(desc_raw_emb, dtype=torch.float32).to(
                device
            )
            desc_proj_emb = alignment_model.get_desc_projection(desc_raw_emb_tensor)

        # PDB <> AAS
        if pdb_proj_emb is not None and aas_proj_emb is not None:
            sim_pdb_aas = pdb_proj_emb @ aas_proj_emb.T
            print(f"Similarity score (PDB <> AAS) for {pdb_id} and {aas_id}:")
            print(sim_pdb_aas.cpu().numpy())
        else:
            print("Similarity score (PDB <> AAS) not computed due to missing data.")

        # PDB <> DESC
        if pdb_proj_emb is not None and desc_proj_emb is not None:
            sim_pdb_desc = pdb_proj_emb @ desc_proj_emb.T
            print(f"Similarity score (PDB <> DESC) for {pdb_id} and {desc_id}:")
            print(sim_pdb_desc.cpu().numpy())
        else:
            print("Similarity score (PDB <> DESC) not computed due to missing data.")

        # AAS <> DESC
        if aas_proj_emb is not None and desc_proj_emb is not None:
            sim_aas_desc = aas_proj_emb @ desc_proj_emb.T
            print(f"Similarity score (AAS <> DESC) for {aas_id} and {desc_id}:")
            print(sim_aas_desc.cpu().numpy())
        else:
            print("Similarity score (AAS <> DESC) not computed due to missing data.")


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description="Train CLASP")

    # required args
    parser.add_argument("--aas_embeddings_file", type=str, required=True)
    parser.add_argument("--desc_embeddings_file", type=str, required=True)
    parser.add_argument("--preprocessed_pdb_file", type=str, required=True)

    parser.add_argument("--encoder_model_path", type=str, required=True)
    parser.add_argument("--alignment_model_path", type=str, required=True)

    # optional args
    parser.add_argument("--pdb_id", type=str, default=None)
    parser.add_argument("--aas_id", type=str, default=None)
    parser.add_argument("--desc_id", type=str, default=None)

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

    # set device
    if args.device not in ["cpu", "cuda"]:
        raise ValueError("Device must be 'cpu' or 'cuda'")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this machine, use 'cpu' instead")
    device = torch.device(args.device)

    # ensure data is in correct format
    try:
        with h5py.File(args.aas_embeddings_file, "r") as f:
            print("Loading amino acid embeddings...")
            amino_acid_embeddings = {k: f[k][()] for k in f.keys()}
    except Exception as e:
        raise ValueError(f"Error loading amino acid embeddings: {e}")

    try:
        with h5py.File(args.desc_embeddings_file, "r") as f:
            print("Loading descriptor embeddings...")
            desc_embeddings = {k: f[k][()] for k in f.keys()}
    except Exception as e:
        raise ValueError(f"Error loading descriptor embeddings: {e}")

    try:
        print("Loading preprocessed PDB data...")
        pdb_data = torch.load(args.preprocessed_pdb_file, weights_only=False)
    except Exception as e:
        raise ValueError(f"Error loading preprocessed PDB file: {e}")

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

    # compute and print quick similarity scores
    compute_and_print_quick_similarity_scores(
        args.pdb_id,
        args.aas_id,
        args.desc_id,
        pdb_data,
        amino_acid_embeddings,
        desc_embeddings,
        pdb_encoder,
        alignment_model,
        device,
    )
