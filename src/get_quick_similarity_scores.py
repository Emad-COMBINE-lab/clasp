import os
import argparse

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

    args = parser.parse_args()


# THIS WILL PRINT A SIMPLE SCORE FOR THE DESIRED PAIRING(s) - TBD after sim matrices file finalised
