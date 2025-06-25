import os
import sys
import argparse
import torch
from pathlib import Path

import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    
# Ensure we can import from src/
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

from read_and_write_docs import read_jsonl, write_jsonl
from preprocessing import vectorize_df
from tokenize_and_score import score_dataframe

def parse_args():
    p = argparse.ArgumentParser(
        description="Score a single JSONL file with token-level log-probs"
    )
    p.add_argument(
        "--input_file",
        required=True,
        help="Path to the input .jsonl file",
    )
    p.add_argument(
        "--output_file",
        required=True,
        help="Path where the scored .jsonl should be written",
    )
    p.add_argument(
        "--model_loc",
        required=True,
        help="Local path or HuggingFace ID for the causal LM",
    )
    p.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help="Number of CPU threads to use (defaults to PyTorch’s choice)",
    )
    return p.parse_args()

def main():
    args = parse_args()

    # Thread configuration
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(args.num_threads)
    print(f"Using {torch.get_num_threads()} threads for inference")

    in_path = Path(args.input_file)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already exists
    if out_path.exists():
        print(f"Skipping {in_path.name} (already exists)")
        return

    print(f"Scoring '{in_path.name}' → '{out_path}'")

    # 1) Read & normalize
    df = read_jsonl(str(in_path))
    df['impostor_id'] = df.index + 1
    df = df[['doc_id', 'corpus', 'impostor_id', 'author', 'texttype', 'rephrased']]
    df = df.rename(columns={'rephrased': 'text'})

    # 2) Vectorize
    vectorized = vectorize_df(df, impostors=True)
    vectorized = vectorized.rename(columns={'sentence': 'text'})

    # 3) Score
    scored = score_dataframe(
        vectorized,
        text_column="text",
        model_loc=args.model_loc
    )

    # 4) Write out
    write_jsonl(scored, str(out_path))
    print("✔ Done.")

if __name__ == "__main__":
    main()

