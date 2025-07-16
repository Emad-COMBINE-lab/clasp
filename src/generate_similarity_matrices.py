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
    Encode + project structures, sequences, and descriptions; optionally save
    similarity matrices. Encodes PDB graphs one at a time (safe path).
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # PDBs
        pdb_proj_embs = []
        if ordered_pbd_ids:
            for pdb_id in ordered_pbd_ids:
                pdb_data_item = pdb_data.get(pdb_id)
                if pdb_data_item is None:
                    raise KeyError(f"PDB data missing for id: {pdb_id}")
                pdb_data_item = pdb_data_item.to(device)
                pdb_emb = pdb_encoder(pdb_data_item)
                pdb_emb_tensor = torch.tensor(pdb_emb, dtype=torch.float32).to(device)
                projected_embedding = alignment_model.get_pdb_projection(pdb_emb_tensor)
                pdb_proj_embs.append(projected_embedding)

        # AASs
        aas_proj_embs = []
        if ordered_aas_ids:
            for aas_id in ordered_aas_ids:
                aas_raw_emb = aas_embeddings.get(aas_id)
                if aas_raw_emb is None:
                    raise KeyError(f"Amino acid embedding missing for id: {aas_id}")
                aas_raw_emb_tensor = torch.tensor(aas_raw_emb, dtype=torch.float32).to(
                    device
                )
                aas_proj_emb = alignment_model.get_aas_projection(aas_raw_emb_tensor)
                aas_proj_embs.append(aas_proj_emb)

        # DESCs
        desc_proj_embs = []
        if ordered_desc_ids:
            for desc_id in ordered_desc_ids:
                desc_raw_emb = desc_embeddings.get(desc_id)
                if desc_raw_emb is None:
                    raise KeyError(f"Description embedding missing for id: {desc_id}")
                desc_raw_emb_tensor = torch.tensor(
                    desc_raw_emb, dtype=torch.float32
                ).to(device)
                desc_proj_emb = alignment_model.get_desc_projection(desc_raw_emb_tensor)
                desc_proj_embs.append(desc_proj_emb)

        def _collapse_proj_list(tlist, name):
            """Stack list of proj tensors to shape (N,D) or return None if empty."""
            if not tlist:
                return None
            # standardize each tensor: squeeze all singleton dims to 1D, then unsqueeze(0)
            normed = []
            for t in tlist:
                if not torch.is_tensor(t):
                    t = torch.as_tensor(t, device=device, dtype=torch.float32)
                t = t.to(device)
                t = t.squeeze()  # drop all 1‑sz dims -> (D,)
                if t.dim() != 1:
                    raise RuntimeError(
                        f"{name}: expected 1D after squeeze, got {t.shape}"
                    )
                normed.append(t.unsqueeze(0))  # (1,D)
            return torch.cat(normed, dim=0)  # (N,D)

        pdb_proj = _collapse_proj_list(pdb_proj_embs, "pdb_proj")
        aas_proj = _collapse_proj_list(aas_proj_embs, "aas_proj")
        desc_proj = _collapse_proj_list(desc_proj_embs, "desc_proj")
        print(
            f"Generated projections: "
            f"pdb_proj={pdb_proj.shape if pdb_proj is not None else 'None'}, "
            f"aas_proj={aas_proj.shape if aas_proj is not None else 'None'}, "
            f"desc_proj={desc_proj.shape if desc_proj is not None else 'None'}"
        )

        # ----- Similarity matrices -----
        if (
            structure_to_sequence_matrix
            and (pdb_proj is not None)
            and (aas_proj is not None)
        ):
            sim_pdb_aas = pdb_proj @ aas_proj.T  # (Np,Na)
            print(f"Generated similarity matrix: sim_pdb_aas={sim_pdb_aas.shape}")
            torch.save(sim_pdb_aas.cpu(), outdir / "structure_to_sequence.pt")

        if (
            structure_to_description_matrix
            and (pdb_proj is not None)
            and (desc_proj is not None)
        ):
            sim_pdb_desc = pdb_proj @ desc_proj.T  # (Np,Nd)
            print(f"Generated similarity matrix: sim_pdb_desc={sim_pdb_desc.shape}")
            torch.save(sim_pdb_desc.cpu(), outdir / "structure_to_description.pt")

        if (
            sequence_to_description_matrix
            and (aas_proj is not None)
            and (desc_proj is not None)
        ):
            sim_aas_desc = aas_proj @ desc_proj.T  # (Na,Nd)
            print(f"Generated similarity matrix: sim_aas_desc={sim_aas_desc.shape}")
            torch.save(sim_aas_desc.cpu(), outdir / "sequence_to_description.pt")


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
