#!/usr/bin/env python3
import argparse
import os
import sys
import torch

import pandas as pd

from from_root import from_root
from pathlib import Path

# Ensure we can import from src/
sys.path.insert(0, str(from_root("src")))

from read_and_write_docs import read_jsonl, read_rds
from model_loading import load_model, distinct_special_chars
from utils import apply_temp_doc_id, build_metadata_df
from n_gram_functions import (
    common_ngrams,
    filter_ngrams,
    pretty_print_common_ngrams,
    get_scored_df,
    get_scored_df_no_context
)

def create_excel_doc(known, unknown, no_context, metadata, docs, save_loc):
    
    path = Path(save_loc)
    
    # Create LLR table (distinct phrases)
    llr_cols = ['phrase_num', 'phrase_occurence', 'original_phrase']
    distinct_phrases = (
        pd.concat([unknown[llr_cols], known[llr_cols]], ignore_index=True)
        .drop_duplicates()
        .sort_values(['phrase_num', 'phrase_occurence'], kind='mergesort')
        .reset_index(drop=True)
    )
    
    # Choose writer mode safely
    writer_mode = "a" if path.exists() else "w"
    writer_kwargs = {"engine": "openpyxl", "mode": writer_mode}
    if writer_mode == "a":
        writer_kwargs["if_sheet_exists"] = "replace"  # only valid in append mode
        

    with pd.ExcelWriter(path, **writer_kwargs) as writer:
        # Write sheets
        docs.to_excel(writer, index=False, sheet_name="docs")
        known.to_excel(writer, index=False, sheet_name="known")
        unknown.to_excel(writer, index=False, sheet_name="unknown")
        no_context.to_excel(writer, index=False, sheet_name="no context")
        distinct_phrases.to_excel(writer, index=False, sheet_name="LLR")
        metadata.to_excel(writer, index=False, sheet_name="metadata")
    
def parse_args():
    ap = argparse.ArgumentParser(description="Pipeline to score the raw data")
    # Paths
    ap.add_argument("--known_loc")
    ap.add_argument("--unknown_loc")
    ap.add_argument("--metadata_loc")
    ap.add_argument("--model_loc")
    ap.add_argument("--save_loc")
    ap.add_argument("--completed_loc", default=None)
    # Dataset hinting
    ap.add_argument("--corpus", default="Wiki")
    ap.add_argument("--data_type", default="training")
    ap.add_argument("--known_doc")
    ap.add_argument("--unknown_doc")
    # N-gram
    ap.add_argument("--ngram_n", type=int, default=2)
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--no_lowercase", dest="lowercase", action="store_false")
    ap.set_defaults(lowercase=True)
    ap.add_argument("--order", default="len_desc", help="Order for pretty_print_common_ngrams")

    return ap.parse_args()

def main():
    
    args=parse_args()
    
    # Ensure the directory exists before beginning
    os.makedirs(args.save_loc, exist_ok=True)
    
    # -----
    # LOAD DATA & LOCAL MODEL
    # -----
    specific_problem = f"{args.known_doc} vs {args.unknown_doc}"
    save_loc = f"{args.save_loc}/{specific_problem}.xlsx"
    
    if args.completed_loc:
        completed_loc = f"{args.completed_loc}/{specific_problem}.xlsx"
        if os.path.exists(completed_loc):
            print(f"Result for {specific_problem} already exists in the completed folder. Exiting.")
            sys.exit()
    
    # Skip the problem if already exists
    if os.path.exists(save_loc):
        print(f"Path {save_loc} already exists. Exiting.")
        sys.exit()
        
    print(f"Working on problem: {specific_problem}")
    
    print("Loading model")
    tokenizer, model = load_model(args.model_loc)
    special_tokens = distinct_special_chars(tokenizer=tokenizer)
    model_name = os.path.basename(os.path.normpath(args.model_loc))
    
    print("Loading data")
    known = read_jsonl(args.known_loc)
    known = apply_temp_doc_id(known)
    
    unknown = read_jsonl(args.unknown_loc)
    unknown = apply_temp_doc_id(unknown)

    print("Data loaded")
    
    # NOTE - Is this used?
    metadata = read_rds(args.metadata_loc)
    filtered_metadata = metadata[metadata['corpus'] == args.corpus]
    agg_metadata = build_metadata_df(filtered_metadata, known, unknown)

    # -----
    # Get the chosen text & metadata
    # -----
    
    known_text = known[known['doc_id'] == args.known_doc].reset_index().loc[0, 'text'].lower()
    unknown_text = unknown[unknown['doc_id'] == args.unknown_doc].reset_index().loc[0, 'text'].lower()
    
    problem_metadata = agg_metadata[(agg_metadata['known_doc_id'] == args.known_doc)
                                    & (agg_metadata['unknown_doc_id'] == args.unknown_doc)].reset_index()
    problem_metadata['target'] = problem_metadata['known_author'] == problem_metadata['unknown_author']

    # -----
    # Create document dataframe
    # -----
    
    # This is used to display the text
    docs_df = pd.DataFrame(
    {
        "known":   [args.corpus, args.data_type, args.known_doc, known_text],
        "unknown": [args.corpus, args.data_type, args.unknown_doc, unknown_text],
    },
    index=["corpus", "data type", "doc", "text"],
    )
    
    # -----
    # Get common n-grams
    # -----
    
    print("Getting common n-grams")
    common = common_ngrams(known_text, unknown_text, args.ngram_n, model, tokenizer, lowercase=args.lowercase)
    
    # Filter to remove smaller n-grams which don't satisfy the rules
    common = filter_ngrams(common, special_tokens=special_tokens)
    n_gram_list = pretty_print_common_ngrams(common, tokenizer=tokenizer, order=args.order, return_format='flat', show_raw=True)
    print(f"There are {len(n_gram_list)} n-grams in common!")
    
    n_gram_dict = {}
    width = len(str(len(n_gram_list)))

    for idx, (phrase_pretty, phrase_raw) in enumerate(n_gram_list, start=1):
        key = f"phrase_{idx:0{width}d}"  # -> phrase_01, phrase_002, etc.
        n_gram_dict[key] = {"phrase": phrase_pretty, "paraphrases": []}
        
    print("Scoring phrases")
    print("    Scoring known text")
    known_scored = get_scored_df(n_gram_dict, known_text, tokenizer, model)
        
    print("    Scoring unknown text")
    unknown_scored = get_scored_df(n_gram_dict, unknown_text, tokenizer, model)
        
    print("    Scoring phrases with no context")
    no_context_scored = get_scored_df_no_context(n_gram_dict, tokenizer, model)
    
    create_excel_doc(
        known_scored,
        unknown_scored,
        no_context_scored,
        metadata,
        docs_df,
        save_loc
    )
    
if __name__ == "__main__":
    main()