#!/usr/bin/env python3
import os
import sys
import argparse
import time
import inspect
from multiprocessing import Pool

import pandas as pd

# Ensure we can import from src/
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

from read_and_write_docs import read_jsonl, write_jsonl, read_rds
from utils import apply_temp_doc_id, build_metadata_df
from lambdaG import lambdaG, lambdaG_paraphrase, lambdaG_perplexity, lambdaG_jsd, lambdaG_renyi, lambdaG_entropy_weighted, lambdaG_surprisal, lambdaG_hellinger

# Store all AV methods in the following registry allows us to load models from other modules
# in a single, unified way.
_METHOD_REGISTRY = {
    'lambdaG': lambdaG,
    'lambdaG_paraphrase': lambdaG_paraphrase,
    'lambdaG_max_perplexity': lambdaG_perplexity,
    'lambdaG_jsd': lambdaG_jsd,
    'lambdaG_renyi': lambdaG_renyi,
    'lambdaG_entropy': lambdaG_entropy_weighted,
    'lambdaG_surprisal': lambdaG_surprisal,
    'lambdaG_hellinger': lambdaG_hellinger
}

def parse_args():
    p = argparse.ArgumentParser(
        description="Run author-verification or impostor methods in parallel repetitions"
    )
    # Core flags
    p.add_argument("--method", required=True,
                   choices=list(_METHOD_REGISTRY.keys()),
                   help="Function name to run")
    p.add_argument("--known_loc", required=True, help="Path to known JSONL/CSV/pickle")
    p.add_argument("--unknown_loc", required=True, help="Path to unknown JSONL/CSV/pickle")
    p.add_argument("--metadata_loc", required=True, help="Path to metadata RDS/CSV/pickle")
    p.add_argument("--save_loc", required=True, help="Where to write combined results (JSONL/CSV)")

    # References vs impostors
    p.add_argument("--refs", default=None,
                   help="Path to reference corpus for calibration (if method takes 'refs')")
    p.add_argument("--impostor_loc", default=None,
                   help="Path or dir of impostor/reference docs (if method takes 'impostor_loc')")

    # Common hyperparameters
    p.add_argument("-N", "--N", "--ngram", type=int, default=10,
                   help="N-gram length (if method takes 'N')")
    p.add_argument("-r", "--r", type=int, default=None,
                   help="Number of iterations/bootstrap samples (if method takes 'r')")
    p.add_argument("--num_cores", type=int, default=1,
                   help="CPU cores for parallel processing (if method takes 'cores')")
    p.add_argument("--num_repetitions", type=int, default=5,
                   help="How many independent repeats to run")

    # Tags for output
    p.add_argument("--corpus", default=None, help="Corpus tag to insert in output")
    p.add_argument("--data_type", default=None, help="Data type tag to insert in output")
    p.add_argument("--token_type", default=None, help="Token type tag to insert in output")
    p.add_argument("--description", default=None, help="Description tag to insert in output")

    return p.parse_args()

def load_and_prep(args):
    """Preps the data for lambdaG method"""

    known = read_jsonl(args.known_loc)
    if 'sentence' in known.columns:
        known = known.rename(columns={'sentence':'text'})
    known = apply_temp_doc_id(known)
    known = known[known['num_tokens'] > 0]

    unknown = read_jsonl(args.unknown_loc)
    if 'sentence' in unknown.columns:
        unknown = unknown.rename(columns={'sentence':'text'})
    unknown = apply_temp_doc_id(unknown)
    unknown = unknown[unknown['num_tokens'] > 0]

    meta = read_rds(args.metadata_loc)
    if args.corpus and 'corpus' in meta.columns:
        meta = meta[meta['corpus'] == args.corpus]
    agg_metadata = build_metadata_df(meta, known, unknown)

    return known, unknown, agg_metadata

def single_run(rep_id, args, known, unknown, agg_metadata):
    func = _METHOD_REGISTRY[args.method]
    sig = inspect.signature(func)
    call_kwargs = {}

    # Inject pre-loaded DataFrames
    for name in ('known', 'unknown', 'metadata'):
        if name in sig.parameters:
            call_kwargs[name] = {'known':known, 'unknown':unknown, 'metadata':agg_metadata}[name]

    # Inject file-based paths
    if 'refs' in sig.parameters and args.refs is not None:
        call_kwargs['refs'] = args.refs
    if 'impostor_loc' in sig.parameters and args.impostor_loc is not None:
        call_kwargs['impostor_loc'] = args.impostor_loc

    # Inject any matching CLI args by name
    for name, param in sig.parameters.items():
        if name in call_kwargs:
            continue
        if hasattr(args, name) and getattr(args, name) is not None:
            call_kwargs[name] = getattr(args, name)

    # Run the function
    df = func(**call_kwargs)

    # Tag repetition and optional metadata
    df.insert(0, 'repetition', rep_id + 1)
    pos = 1
    for tag in ('corpus', 'data_type', 'token_type', 'description'):
        val = getattr(args, tag)
        if val is not None:
            df.insert(pos, tag, val)
            pos += 1
    return df

def main():
    args = parse_args()
    start = time.time()

    known, unknown, agg_metadata = load_and_prep(args)
    tasks = [(i, args, known, unknown, agg_metadata) for i in range(args.num_repetitions)]
    with Pool(processes=args.num_cores) as pool:
        results = pool.starmap(single_run, tasks)

    all_df = pd.concat(results, ignore_index=True)

    write_jsonl(all_df, args.save_loc)

    print(f"Completed {args.num_repetitions} runs in {time.time()-start:.1f}s")

if __name__ == '__main__':
    main()