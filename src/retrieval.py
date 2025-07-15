from transformers import BioGptTokenizer, BioGptModel
import os
import argparse
import torch


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


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description="Train CLASP")

    # required args
    parser.add_argument("--aas_embeddings_file", type=str, required=True)
    parser.add_argument("--query_description_file", type=str, required=True)

    parser.add_argument("--encoder_model_path", type=str, required=True)
    parser.add_argument("--alignment_model_path", type=str, required=True)

    parser.add_argument("--aas_universe", type=str, required=True)

    # optional args
    parser.add_argument("--return_top_k", type=int, default=10)
    parser.add_argument("--output_file", type=str, default="ranked_aas.jsonl")

    args = parser.parse_args()
