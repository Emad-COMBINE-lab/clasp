from transformers import BioGptTokenizer, BioGptModel
import os
import argparse
import torch
import h5py
import json
from models import CLASPAlignment
from utils import create_clip_model_with_random_weights


def embed_protein_description(description):
    """
    Embed a protein description using BioGPT.
    """
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    model = BioGptModel.from_pretrained("microsoft/biogpt")

    inputs = tokenizer(
        description, return_tensors="pt", truncation=True, max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    embedding = last_hidden_state.mean(dim=1)

    return embedding.squeeze()


def retrieve_amino_acid_embeddings(
    amino_acid_embeddings,
    query_desc,
    alignment_model,
    aas_universe,
    k,
    output_file,
    device,
):
    """
    Rank amino-acid sequences for a given protein description.
    """
    alignment_model.eval()
    with torch.no_grad():
        desc_emb = embed_protein_description(query_desc).to(device)
        proj_desc = alignment_model.get_desc_projection(desc_emb.unsqueeze(0))

    # Project amino-acid embeddings in batches and compute similarities
    aa_ids = []
    for aa in aas_universe:
        if aa in amino_acid_embeddings:
            aa_ids.append(aa)
        else:
            raise KeyError(f"Amino acid {aa} not found in embeddings")

    batch_size = 256
    sim_scores = []

    with torch.no_grad():
        for i in range(0, len(aa_ids), batch_size):
            batch_ids = aa_ids[i : i + batch_size]
            batch_emb = [
                torch.tensor(amino_acid_embeddings[aa], dtype=torch.float32)
                for aa in batch_ids
            ]
            batch_emb = torch.stack(batch_emb).to(device)
            proj_aas = alignment_model.get_aas_projection(batch_emb)
            sims = torch.sum(proj_aas * proj_desc, dim=1).cpu().numpy()

            sim_scores.extend(zip(sims, batch_ids))

    sim_scores.sort(key=lambda x: x[0], reverse=True)
    top_k = sim_scores[:k]

    print(f"\nTop {k} amino acids for the query description:")
    for rank, (score, aa_id) in enumerate(top_k, 1):
        print(f"{rank:2d}. {aa_id:<15}  score = {score: .4f}")

    results_jsonl = [
        {"rank": rank, "amino_acid_id": aa_id, "score": float(score)}
        for rank, (score, aa_id) in enumerate(top_k, 1)
    ]

    with open(output_file, "w") as f:
        for entry in results_jsonl:
            f.write(json.dumps(entry) + "\n")

    return results_jsonl


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description="Train CLASP")

    # required args
    parser.add_argument("--aas_embeddings_file", type=str, required=True)
    parser.add_argument("--query_description_file", type=str, required=True)

    parser.add_argument("--alignment_model_path", type=str, required=True)

    parser.add_argument("--aas_universe_file", type=str, required=True)

    # optional args
    parser.add_argument("--return_top_k", type=int, default=10)
    parser.add_argument("--output_file", type=str, default="ranked_aas.jsonl")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    # check if paths exist
    if not os.path.exists(args.aas_embeddings_file):
        raise FileNotFoundError(
            f"Amino acid embeddings file not found: {args.aas_embeddings_file}"
        )
    if not os.path.exists(args.query_description_file):
        raise FileNotFoundError(
            f"Descriptor embeddings file not found: {args.desc_embeddings_file}"
        )
    if not os.path.exists(args.alignment_model_path):
        raise FileNotFoundError(
            f"Alignment model path not found: {args.alignment_model_path}"
        )
    if not os.path.exists(args.aas_universe_file):
        raise FileNotFoundError(
            f"Amino acid universe file not found: {args.aas_universe_file}"
        )

    # output file
    output_file = args.output_file
    if not output_file.endswith(".jsonl"):
        raise ValueError("Output file must have a .jsonl extension")

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
        with open(args.query_description_file, "r") as f:
            query_description = f.read().strip()
    except Exception as e:
        raise ValueError(f"Error reading query description file: {e}")

    try:
        with open(args.aas_universe_file, "r") as f:
            aas_universe = json.load(f)
            if not isinstance(aas_universe, list):
                raise ValueError("Amino acid universe must be a list")
    except Exception as e:
        raise ValueError(f"Error loading amino acid universe: {e}")

    # ensure models are in correct format
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

    # retrieve amino acid embeddings
    retrieve_amino_acid_embeddings(
        amino_acid_embeddings,
        query_description,
        alignment_model,
        aas_universe,
        args.return_top_k,
        output_file,
        device,
    )
