import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    accuracy_score,
    roc_curve,
    auc,
    matthews_corrcoef,
)
import types
import h5py
from models import CLASPEncoder, CLASPAlignment


class EvalPairDatasetPDB(Dataset):
    """
    Pair dataset for evaluating CLASP model with labeled PDB-X pairs.
    """

    def __init__(
        self, labeled_pairs, aas_or_desc_embeddings, structure_encoder, pdb_data, device
    ):
        self.labeled_pairs = labeled_pairs
        self.aas_or_desc_embeddings = aas_or_desc_embeddings
        self.structure_encoder = structure_encoder
        self.pdb_data = pdb_data
        self.device = device

    def __len__(self):
        return len(self.labeled_pairs)

    def __getitem__(self, idx):
        (upkb_ac, pdb_id), label = self.labeled_pairs[idx]
        graph_data = self.pdb_data.get(pdb_id, None)

        raw_aas_or_desc_embedding = self.aas_or_desc_embeddings.get(upkb_ac, None)
        if raw_aas_or_desc_embedding is None or graph_data is None:
            return None

        aas_or_desc_embedding = torch.tensor(
            raw_aas_or_desc_embedding, dtype=torch.float32
        )

        with torch.no_grad():
            structure_embedding = self.structure_encoder(graph_data.to(self.device))

        return aas_or_desc_embedding.to(self.device), structure_embedding, label


class EvalPairDatasetAASxDESC(Dataset):
    """
    Pair dataset for evaluating CLASP model with labeled AAS-DESC pairs.
    """

    def __init__(self, labeled_pairs, amino_acid_embeddings, desc_embeddings, device):
        self.labeled_pairs = labeled_pairs
        self.amino_acid_embeddings = amino_acid_embeddings
        self.desc_embeddings = desc_embeddings
        self.device = device

    def __len__(self):
        return len(self.labeled_pairs)

    def __getitem__(self, idx):
        (upkb_ac_aas, upkb_ac_desc), label = self.labeled_pairs[idx]

        raw_aas_embedding = self.amino_acid_embeddings.get(upkb_ac_aas, None)
        raw_desc_embedding = self.desc_embeddings.get(upkb_ac_desc, None)
        if raw_aas_embedding is None or raw_desc_embedding is None:
            return None

        amino_ac_embedding = torch.tensor(raw_aas_embedding, dtype=torch.float32).to(
            self.device
        )
        structure_embedding = torch.tensor(raw_desc_embedding, dtype=torch.float32).to(
            self.device
        )

        return amino_ac_embedding, structure_embedding, label


def pair_eval_collate_fn(batch):
    """
    Collate function to filter out None entries for evaluation pairs.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    amino_embeddings, structure_embeddings, labels = zip(*batch)
    return (
        torch.stack(amino_embeddings),
        torch.stack(structure_embeddings),
        torch.tensor(labels, dtype=torch.long),
    )


def compute_similarity_scores(model, test_loader):
    """
    Compute similarity scores and collect labels for all pairs in the test set.
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            amino_embeddings, structure_embeddings, labels = batch

            logits_per_text, _ = model(amino_embeddings, structure_embeddings)
            scores = torch.diag(logits_per_text).cpu().numpy()

            all_scores.extend(scores)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_scores), np.array(all_labels)


def find_optimal_threshold(ground_truth, similarity_scores):
    """
    Find the optimal threshold that maximizes F1 score.
    """
    precisions, recalls, thresholds = precision_recall_curve(
        ground_truth, similarity_scores
    )
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_threshold_idx = np.argmax(f1_scores)
    return (
        thresholds[optimal_threshold_idx]
        if optimal_threshold_idx < len(thresholds)
        else thresholds[-1]
    )


def evaluate_clip_model(model_name, clip_model, test_loader, threshold):
    """
    Evaluate a CLIP-based model at the given threshold.
    """
    clip_model.eval()

    test_scores, test_labels = compute_similarity_scores(clip_model, test_loader)
    binary_predictions = (test_scores >= threshold).astype(int)

    accuracy = accuracy_score(test_labels, binary_predictions)
    f1 = f1_score(test_labels, binary_predictions)

    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    roc_auc = auc(fpr, tpr)
    pr_precision, pr_recall, _ = precision_recall_curve(test_labels, test_scores)
    pr_auc = auc(pr_recall, pr_precision)

    mcc = matthews_corrcoef(test_labels, binary_predictions)

    return {
        "model_name": model_name,
        "scores": test_scores,
        "labels": test_labels,
        "accuracy": accuracy,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mcc": mcc,
    }


def load_classification_pairs(pairs_path):
    """
    Load classification pairs from a JSONL file.
    """
    pairs = []
    with open(pairs_path, "r") as f:
        for line in f:
            pair = json.loads(line.strip())
            pairs.append(((pair[0][0], pair[0][1]), pair[1]))
    return pairs


def evaluate_zero_shot_classification(
    amino_acid_embeddings,
    description_embeddings,
    pdb_data,
    clasp_encoder,
    clasp_alignment,
    pdb_val_pairs,
    pdb_test_pairs,
    aas_desc_val_pairs,
    aas_desc_test_pairs,
    device,
    pdb_aas=True,
    pdb_desc=True,
    aas_desc=True,
):
    """
    Evaluate CLASP model on three zero-shot classification tasks:
    1) PDB-AAS classification
    2) PDB-DESC classification
    3) AAS-DESC classification
    """
    metric_names = [
        "accuracy",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "mcc",
    ]

    tasks = ["PDB-AAS", "PDB-DESC", "AAS-DESC"]
    metrics_by_task = {task: {m: [] for m in metric_names} for task in tasks}

    # PDB-AAS CLASSIFICATION
    if pdb_aas:
        val_loader = DataLoader(
            EvalPairDatasetPDB(
                pdb_val_pairs,
                amino_acid_embeddings,
                clasp_encoder,
                pdb_data,
                device,
            ),
            batch_size=32,
            shuffle=False,
            collate_fn=pair_eval_collate_fn,
        )
        test_loader = DataLoader(
            EvalPairDatasetPDB(
                pdb_test_pairs,
                amino_acid_embeddings,
                clasp_encoder,
                pdb_data,
                device,
            ),
            batch_size=32,
            shuffle=False,
            collate_fn=pair_eval_collate_fn,
        )

        def _forward_pdb_aas(self, text_embeds, image_embeds):
            logits_pdb_aas, logits_aas_pdb = self.get_pdb_aas_logits(
                image_embeds, text_embeds
            )
            return logits_aas_pdb, logits_pdb_aas

        clasp_alignment.forward = types.MethodType(_forward_pdb_aas, clasp_alignment)

        val_scores, val_labels = compute_similarity_scores(clasp_alignment, val_loader)
        threshold = find_optimal_threshold(val_labels, val_scores)
        pdb_aas_results = evaluate_clip_model(
            "CLASP", clasp_alignment, test_loader, threshold
        )

        for m in metric_names:
            metrics_by_task["PDB-AAS"][m].append(pdb_aas_results[m])

    # PDB-DESC CLASSIFICATION
    if pdb_desc:
        val_loader = DataLoader(
            EvalPairDatasetPDB(
                pdb_val_pairs,
                description_embeddings,
                clasp_encoder,
                pdb_data,
                device,
            ),
            batch_size=32,
            shuffle=False,
            collate_fn=pair_eval_collate_fn,
        )
        test_loader = DataLoader(
            EvalPairDatasetPDB(
                pdb_test_pairs,
                description_embeddings,
                clasp_encoder,
                pdb_data,
                device,
            ),
            batch_size=32,
            shuffle=False,
            collate_fn=pair_eval_collate_fn,
        )

        def _forward_pdb_desc(self, text_embeds, image_embeds):
            logits_pdb_desc, logits_desc_pdb = self.get_pdb_desc_logits(
                image_embeds, text_embeds
            )
            return logits_desc_pdb, logits_pdb_desc

        clasp_alignment.forward = types.MethodType(_forward_pdb_desc, clasp_alignment)

        val_scores, val_labels = compute_similarity_scores(clasp_alignment, val_loader)
        threshold = find_optimal_threshold(val_labels, val_scores)
        pdb_desc_results = evaluate_clip_model(
            "CLASP", clasp_alignment, test_loader, threshold
        )

        for m in metric_names:
            metrics_by_task["PDB-DESC"][m].append(pdb_desc_results[m])

    # AAS-DESC CLASSIFICATION
    if aas_desc:
        val_loader = DataLoader(
            EvalPairDatasetAASxDESC(
                aas_desc_val_pairs,
                amino_acid_embeddings,
                description_embeddings,
                device,
            ),
            batch_size=32,
            shuffle=False,
            collate_fn=pair_eval_collate_fn,
        )
        test_loader = DataLoader(
            EvalPairDatasetAASxDESC(
                aas_desc_test_pairs,
                amino_acid_embeddings,
                description_embeddings,
                device,
            ),
            batch_size=32,
            shuffle=False,
            collate_fn=pair_eval_collate_fn,
        )

        def _forward_aas_desc(self, text_embeds, image_embeds):
            logits_aas_desc, logits_desc_aas = self.get_aas_desc_logits(
                text_embeds, image_embeds
            )
            return logits_desc_aas, logits_aas_desc

        clasp_alignment.forward = types.MethodType(_forward_aas_desc, clasp_alignment)

        val_scores, val_labels = compute_similarity_scores(clasp_alignment, val_loader)
        threshold = find_optimal_threshold(val_labels, val_scores)
        aas_desc_results = evaluate_clip_model(
            "CLASP", clasp_alignment, test_loader, threshold
        )

        for m in metric_names:
            metrics_by_task["AAS-DESC"][m].append(aas_desc_results[m])

    # Print results across all tasks
    print("\n" + "=" * 60)
    print("CLASP ZERO-SHOT CLASSIFICATION RESULTS")
    print("=" * 60)

    for task in tasks:
        print(f"\n--- {task} ---")
        for m in metric_names:
            vals = np.array(metrics_by_task[task][m])
            mean, std = vals.mean(), vals.std()
            print(f"{m:15s}: {mean:.4f} ± {std:.4f}")
        print()

    return metrics_by_task


def main() -> None:
    # parser = argparse.ArgumentParser(
    #     description="Generate CLASP similarity matrices for structure, sequence, and description"
    # )
    # parser.add_argument(
    #     "--aas_embeddings_file",
    #     type=Path,
    #     required=True,
    #     help="HDF5 file with amino-acid embeddings",
    # )
    # parser.add_argument(
    #     "--desc_embeddings_file",
    #     type=Path,
    #     required=True,
    #     help="HDF5 file with description embeddings",
    # )
    # parser.add_argument(
    #     "--preprocessed_pdb_file",
    #     type=Path,
    #     required=True,
    #     help=".pt file with preprocessed PDB graphs",
    # )
    # parser.add_argument(
    #     "--encoder_model_path",
    #     type=Path,
    #     required=True,
    #     help="Path to saved CLASPEncoder state_dict",
    # )
    # parser.add_argument(
    #     "--alignment_model_path",
    #     type=Path,
    #     required=True,
    #     help="Path to saved CLASPAlignment state_dict",
    # )
    # parser.add_argument(
    #     "--target_file",
    #     type=Path,
    #     required=True,
    #     help="JSON file listing pdb_ids, aas_ids, desc_ids",
    # )
    # parser.add_argument(
    #     "--structure_to_sequence_matrix",
    #     type=bool,
    #     default=True,
    #     help="Whether to compute structure-to-sequence similarities",
    # )
    # parser.add_argument(
    #     "--structure_to_description_matrix",
    #     type=bool,
    #     default=True,
    #     help="Whether to compute structure-to-description similarities",
    # )
    # parser.add_argument(
    #     "--sequence_to_description_matrix",
    #     type=bool,
    #     default=True,
    #     help="Whether to compute sequence-to-description similarities",
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     type=Path,
    #     default=Path("output"),
    #     help="Directory to save similarity matrices",
    # )
    # parser.add_argument(
    #     "--device",
    #     type=str,
    #     default="cuda" if torch.cuda.is_available() else "cpu",
    #     choices=["cpu", "cuda"],
    #     help="Compute device",
    # )

    # args = parser.parse_args()

    # # validate inputs
    # for p in (
    #     args.aas_embeddings_file,
    #     args.desc_embeddings_file,
    #     args.preprocessed_pdb_file,
    #     args.encoder_model_path,
    #     args.alignment_model_path,
    #     args.target_file,
    # ):
    #     if not p.exists():
    #         parser.error(f"File not found: {p}")

    # device = torch.device(args.device)
    aas_embeddings_file = "/projects/nbolo/CLIENT/clasp/data/amino_acid_embeddings.h5"
    desc_embeddings_file = "/projects/nbolo/CLIENT/clasp/data/description_embeddings.h5"
    preprocessed_pdb_file = "/projects/nbolo/CLIENT/clasp/data/processed_pdb_data.pt"

    encoder_model_path = "/projects/nbolo/CLIENT/clasp/checkpoints/best_encoder.pt"
    alignment_model_path = "/projects/nbolo/CLIENT/clasp/checkpoints/best_alignment.pt"

    aas_desc_val_pairs_path = "/projects/nbolo/CLIENT/clasp/data/processed/seed_26855092/balanced_pairs/aas_desc_test_pairs.jsonl"
    aas_desc_test_pairs_path = "/projects/nbolo/CLIENT/clasp/data/processed/seed_26855092/balanced_pairs/aas_desc_val_pairs.jsonl"
    pdb_val_pairs_path = "/projects/nbolo/CLIENT/clasp/data/processed/seed_26855092/balanced_pairs/pdb_val_pairs.jsonl"
    pdb_test_pairs_path = "/projects/nbolo/CLIENT/clasp/data/processed/seed_26855092/balanced_pairs/pdb_test_pairs.jsonl"

    # load pairs
    print("Loading classification pairs...")
    aas_desc_val_pairs = load_classification_pairs(aas_desc_val_pairs_path)
    aas_desc_test_pairs = load_classification_pairs(aas_desc_test_pairs_path)
    pdb_val_pairs = load_classification_pairs(pdb_val_pairs_path)
    pdb_test_pairs = load_classification_pairs(pdb_test_pairs_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load embeddings
    with h5py.File(aas_embeddings_file, "r") as f:
        print("Loading amino acid embeddings...")
        aas_embeddings = {k: f[k][()] for k in f.keys()}
    with h5py.File(desc_embeddings_file, "r") as f:
        print("Loading description embeddings...")
        desc_embeddings = {k: f[k][()] for k in f.keys()}

    # load PDB data and targets
    print("Loading preprocessed PDB data...")
    pdb_data = torch.load(str(preprocessed_pdb_file), weights_only=False)

    # load models
    encoder = CLASPEncoder(
        in_channels=7,
        hidden_channels=16,
        final_embedding_size=512,
        target_size=512,
    ).to(device)
    encoder.load_state_dict(torch.load(encoder_model_path, map_location=device))
    encoder.eval()

    alignment = CLASPAlignment(embed_dim=512).to(device)
    alignment.load_state_dict(torch.load(alignment_model_path, map_location=device))
    alignment.eval()

    evaluate_zero_shot_classification(
        amino_acid_embeddings=aas_embeddings,
        description_embeddings=desc_embeddings,
        pdb_data=pdb_data,
        clasp_encoder=encoder,
        clasp_alignment=alignment,
        pdb_val_pairs=pdb_val_pairs,
        pdb_test_pairs=pdb_test_pairs,
        aas_desc_val_pairs=aas_desc_val_pairs,
        aas_desc_test_pairs=aas_desc_test_pairs,
        device=device,
        pdb_aas=True,
        pdb_desc=True,
        aas_desc=True,
    )


if __name__ == "__main__":
    main()
