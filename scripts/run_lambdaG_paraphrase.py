import sys
import argparse
import time

import pandas as pd
from from_root import from_root

# Ensure we can import from src/
sys.path.insert(0, str(from_root("src")))

from read_and_write_docs import read_jsonl, write_jsonl, read_rds
from utils import apply_temp_doc_id, build_metadata_df
from lambdaG import lambdaG_paraphrase

def parse_args():
    
    p = argparse.ArgumentParser(
        description="Run the LambdaG method on paraphrased documents"
    )
    p.add_argument("--known_loc",     required=True, help="Path to known .jsonl")
    p.add_argument("--unknown_loc",   required=True, help="Path to unknown .jsonl")
    p.add_argument("--reference_dir", required=True, help="Dir of reference docs")
    p.add_argument("--metadata_loc",  required=True, help="Path to metadata .rds")
    p.add_argument("--save_loc",      required=True, help="Where to write results")
    p.add_argument("--N",   type=int, default=10, help="Model order. Default: 10")
    p.add_argument("--r",   type=int, default=30, help="Iterations. Default: 30")
    p.add_argument("--num_repetitions", type=int, default=5,
                   help="How many times to repeat the test")
    p.add_argument("cores", type=int, nargs="?", default=1,
                   help="Cores for parallel (default: 1)")

    # Optional tags
    p.add_argument("--corpus",     default=None, help="Corpus tag")
    p.add_argument("--data_type",  default=None, help="Data type tag")
    p.add_argument("--token_type", default=None, help="Token type tag")
    p.add_argument("--description",default=None, help="Free-form description")

    return p.parse_args()


def main():
    args=parse_args()

    start_time = time.time()
    
    # Pull the arguments for the additional output metadata
    optional_tags = {
        'corpus': args.corpus,
        'data_type': args.data_type,
        'token_type': args.token_type,
        'description': args.description,
    }
    
    # Known preprocess
    known = read_jsonl(args.known_loc)
    known.rename(columns={'sentence': 'text'}, inplace=True)
    known = apply_temp_doc_id(known)
    known = known[known['num_tokens'] > 0]

    if args.corpus is not None:
        selected_corpus = [args.corpus]
    else:
        selected_corpus = known['corpus'].unique().tolist()

    # Unknown preprocess
    unknown = read_jsonl(args.unknown_loc)
    unknown.rename(columns={'sentence': 'text'}, inplace=True)
    unknown = apply_temp_doc_id(unknown)
    unknown = unknown[unknown['num_tokens'] > 0]

    # Metadata preprocess
    metadata = read_rds(args.metadata_loc)
    filtered_metadata = metadata[metadata['corpus'].isin(selected_corpus)]
    agg_metadata = build_metadata_df(filtered_metadata, known, unknown)

    all_results = []
    for rep in range(1, args.num_repetitions + 1):
        print(f"Repetition {rep}")
        df = lambdaG_paraphrase(
            unknown,
            known,
            metadata=agg_metadata,
            impostor_loc=args.reference_dir,
            N=args.N,
            r=args.r,
            cores=args.cores
        )

        # Insert the repetition index
        df.insert(0, 'repetition', rep)

        # Insert only the tags that were provided
        col_idx = 1
        for col_name, value in optional_tags.items():
            if value is not None:
                df.insert(col_idx, col_name, value)
                col_idx += 1

        all_results.append(df)
    
    # Combine all repetitions into one DataFrame and save
    results = pd.concat(all_results, ignore_index=True)
    write_jsonl(results, args.save_loc)

    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed:.2f} seconds")
    
if __name__ == "__main__":
    main()