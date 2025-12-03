#!/usr/bin/env python3
import argparse
import ast
import json
import sys
import os
import torch
import pandas as pd

from from_root import from_root
sys.path.insert(0, str(from_root("src")))

from read_and_write_docs import read_jsonl, read_rds
from model_loading import distinct_special_chars
from utils import apply_temp_doc_id, build_metadata_df
from n_gram_functions import (
    common_ngrams,
    filter_ngrams,
    pretty_print_common_ngrams,
)

from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedTokenizerBase, PreTrainedModel
from typing import List, Dict, Optional, Iterable

# --------------------
# Helpers
# --------------------

def find_phrase_token_spans(text: str, phrase_pretty: str, tokenizer):
    """
    Find all token spans for phrase_pretty inside text using tokenizer-consistent tokenization.
    Works for any tokenizer: BERT, RoBERTa, ALBERT, GPT-like, SentencePiece, etc.
    """

    # Tokenize full text WITH special tokens
    enc = tokenizer(text, add_special_tokens=True)
    text_tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])

    # Tokenize phrase using model rules
    phrase_tokens = tokenizer.tokenize(phrase_pretty)
    n = len(phrase_tokens)

    spans = []

    for i in range(len(text_tokens) - n + 1):
        if text_tokens[i:i+n] == phrase_tokens:
            spans.append(list(range(i, i+n)))
    print(f"    Spans: {spans}")
    return spans

@torch.no_grad()
def incremental_mlm_paraphrase(
    sentence: str,
    span_token_idxs: List[int],
    original_phrase: str,
    model,
    tokenizer,
    device,
    top_k: int = 5,
    beam_size: int = 30,
    min_log_prob: float = -7.0
):
    """
    BEAM-SEARCH hierarchical paraphraser.
    Prevents combinatorial explosion. Fast and high quality.
    """

    original_phrase = original_phrase.lower()

    # Correct tokenization with special tokens
    enc = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    base_ids = enc.input_ids[0].tolist()

    before_idx = span_token_idxs[0] - 1
    after_idx  = span_token_idxs[-1] + 1

    # Initial beam contains only the base sequence
    beam = [{"ids": base_ids, "score": 0.0}]

    # ---- Iterate through each token in the span ----
    for position in span_token_idxs:
        candidates = []

        # Batch all masked sequences at once for speed
        masked_batch = []
        for seq in beam:
            seq_ids = seq["ids"]
            masked = seq_ids.copy()
            masked[position] = tokenizer.mask_token_id
            masked_batch.append(masked)

        batch_tensor = torch.tensor(masked_batch, device=device)

        # Model forward pass ONCE for all beam items
        logits = model(input_ids=batch_tensor).logits[:, position, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        # Expand beam
        for beam_idx, seq in enumerate(beam):
            seq_ids = seq["ids"]
            token_logprobs = log_probs[beam_idx]

            top_vals, top_ids = torch.topk(token_logprobs, top_k)

            for lp, tid in zip(top_vals.tolist(), top_ids.tolist()):
                if lp < min_log_prob:
                    continue  # prune terrible tokens

                new_ids = seq_ids.copy()
                new_ids[position] = tid

                candidates.append({
                    "ids": new_ids,
                    "score": seq["score"] + lp
                })

        if not candidates:
            return []

        # Keep only the best beam_size candidates
        candidates.sort(key=lambda x: x["score"], reverse=True)
        beam = candidates[:beam_size]

    # ---- Extract paraphrased phrases ----
    results = []
    seen = set()

    for seq in beam:
        ids = seq["ids"]

        start = before_idx + 1 if before_idx >= 0 else 0
        end = after_idx if after_idx < len(ids) else len(ids)

        span_ids = ids[start:end]

        # Ensure text is in lowercase
        text = tokenizer.decode(span_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False).lower()
        
        # Check if new text is original phrase or if already been seen 
        if not text or text == original_phrase:
            continue
        if text in seen:
            continue

        seen.add(text)

        # Keep tokens with special spacing token etc.
        tokens = tokenizer.convert_ids_to_tokens(span_ids)

        results.append({
            "phrase": text,
            "tokens": tokens,
            "score": seq["score"]
        })

    # Sort by descending score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def save_results_to_excel(
    save_path: str,
    docs_df: pd.DataFrame,
    problem_metadata: pd.DataFrame,
    n_gram_dict: dict
):
    """
    Saves docs_df, problem_metadata, and paraphrase data (from n_gram_dict)
    into an Excel workbook with sheets: "docs", "metadata", "paraphrases".
    The paraphrases sheet has columns:
      - phrase_num  (the key from n_gram_dict, e.g. "phrase_001")
      - original_phrase  (the original phrase)
      - phrase  (either the reference phrase or a paraphrase)
      - phrase_type  ("reference" for the original phrase, "paraphrase" for paraphrases)
    """
    rows = []

    for phrase_num, info in n_gram_dict.items():
        orig = info.get("phrase")
        orig_tokens = info.get("orig_tokens", None)
        paraphrase_entries = info.get("paraphrases", [])
        # First, add the reference/original phrase row
        rows.append({
            "phrase_num": phrase_num,
            "phrase_type": "reference",
            "original_phrase": orig,
            "phrase": orig,
            "tokens": orig_tokens,
            "num_tokens": len(orig_tokens) if orig_tokens is not None else None
        })
        # Then add paraphrase rows
        for p in paraphrase_entries:
            text = p.get("text")
            tokens = p.get("tokens")
            num_tokens = len(tokens) if tokens is not None else None
            rows.append({
                "phrase_num": phrase_num,
                "phrase_type": "paraphrase",
                "original_phrase": orig,
                "phrase": text,
                "tokens": tokens,
                "num_tokens": num_tokens
                
            })

    paraphrases_df = pd.DataFrame(rows)
    
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        docs_df.to_excel(writer, sheet_name="docs", index=False)
        problem_metadata.to_excel(writer, sheet_name="metadata", index=False)
        paraphrases_df.to_excel(writer, sheet_name="no context", index=False)

    print(f"Saved all results to '{save_path}'")


# --------------------
# Parse Arguments
# --------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Token masking model AI paraphrase generation pipeline")
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
    ap.add_argument("--order", default="len_desc", help="Order for pretty_print_common_ngrams")
    
    return ap.parse_args()

# --------------------
# Main
# --------------------
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_loc)
    model = AutoModelForMaskedLM.from_pretrained(args.model_loc)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    special_tokens = distinct_special_chars(tokenizer=tokenizer)
    
    print("Loading data")
    known = read_jsonl(args.known_loc)
    known = apply_temp_doc_id(known)
    
    unknown = read_jsonl(args.unknown_loc)
    unknown = apply_temp_doc_id(unknown)

    # NOTE - Is this used?
    metadata = read_rds(args.metadata_loc)
    filtered_metadata = metadata[metadata['corpus'] == args.corpus]
    agg_metadata = build_metadata_df(filtered_metadata, known, unknown)
    
    print("Data loaded")

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
    
    # -----
    # Paraphrasing
    # -----
    
    print("Generating paraphrases")
    n_gram_dict = {}
    width = len(str(len(n_gram_list)))  # e.g., 10 -> 2, 100 -> 3

    for idx, (phrase_pretty, phrase_raw) in enumerate(n_gram_list, start=1):
        
        key = f"phrase_{idx:0{width}d}"
        print(f"    Working on paraphrase: {key}")
        phrase_list = list(ast.literal_eval(phrase_raw))  # token list
        phrase_token_spans = find_phrase_token_spans(known_text, phrase_pretty, tokenizer)

        paraphrase_records = []

        for span in phrase_token_spans:
            outputs = incremental_mlm_paraphrase(
                sentence=known_text,
                span_token_idxs=span,
                original_phrase=phrase_pretty,
                model=model,
                tokenizer=tokenizer,
                device=device,
                top_k=10,
                beam_size=30
            )
            for o in outputs:
                paraphrase_records.append({
                    "text": o["phrase"],
                    "tokens": o["tokens"]
                })

        # merge duplicates
        merged = {}
        for rec in paraphrase_records:
            txt = rec["text"]
            if txt not in merged:
                merged[txt] = rec

        n_gram_dict[key] = {
            "phrase": phrase_pretty,
            "orig_tokens": phrase_list,
            "paraphrases": list(merged.values())
        }

    # Save results
    save_results_to_excel(
        save_path=save_loc,
        docs_df=docs_df,
        problem_metadata=problem_metadata,
        n_gram_dict=n_gram_dict
    )

if __name__ == "__main__":
    main()