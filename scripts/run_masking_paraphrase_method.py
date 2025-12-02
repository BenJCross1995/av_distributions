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

def mask_phrase(
    text: str,
    phrase_tokens: List[str],
    tokenizer: PreTrainedTokenizerBase,
    mask_token: Optional[str] = "[MASK]",
) -> Dict[str, List[str]]:
    """
    For each occurrence of `phrase_tokens` (a token sequence) in the tokenized
    version of `text`, create two versions of the text:

    1. single_mask: the whole phrase replaced by a **single** mask_token.
    2. multi_mask: each token in the phrase replaced by its own mask_token.

    Returns a dict:
        {
            "single_mask": [ ... ],
            "multi_mask":  [ ... ],
        }
    where each list has length == number of occurrences (can be 0).
    """
    if not phrase_tokens:
        raise ValueError("phrase_tokens must be a non-empty list of tokens")

    if mask_token is None:
        if tokenizer.mask_token is None:
            raise ValueError("mask_token not provided and tokenizer.mask_token is None")
        mask_token = tokenizer.mask_token

    # Tokenize the text (no special tokens)
    tokens = tokenizer.tokenize(text)
    n = len(phrase_tokens)

    single_mask_variants: List[str] = []
    multi_mask_variants: List[str] = []

    # Find all subsequence matches of phrase_tokens in tokens
    for start_idx in range(len(tokens) - n + 1):
        if tokens[start_idx:start_idx + n] == phrase_tokens:
            # --- multi-mask: replace each token in the phrase by mask_token ---
            multi_tokens = tokens.copy()
            for j in range(n):
                multi_tokens[start_idx + j] = mask_token
            multi_text = tokenizer.convert_tokens_to_string(multi_tokens)
            multi_mask_variants.append(multi_text)

            # --- single-mask: collapse the whole phrase into a single mask_token ---
            single_tokens = tokens[:start_idx] + [mask_token] + tokens[start_idx + n:]
            single_text = tokenizer.convert_tokens_to_string(single_tokens)
            single_mask_variants.append(single_text)

    return {
        "single_mask": single_mask_variants,
        "multi_mask": multi_mask_variants,
    }

@torch.no_grad()
def fill_masks_beam(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    masked_text: str,
    top_k_per_mask: int = 25,   # widen/limit local options at each mask
    beam_size: int = 100,       # how many partial hypotheses to keep
    max_candidates: int = 50,   # truncate final list
    banned_token_ids: Optional[Iterable[int]] = None,
) -> List[Dict]:
    """
    Jointly fill one or more [MASK] tokens using a simple beam search.

    Returns a list of dicts: {text, score, tokens, token_ids, mask_positions}
    where 'score' is the sum of log-probs across all masks (higher is better).
    """
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer must define mask_token_id (MLM required).")

    enc = tokenizer(masked_text, return_tensors="pt", add_special_tokens=True, truncation=True)
    input_ids = enc["input_ids"]
    mask_id = tokenizer.mask_token_id
    mask_positions = (input_ids[0] == mask_id).nonzero(as_tuple=False).flatten().tolist()
    if not mask_positions:
        return []

    outputs = model(**enc)
    log_probs = outputs.logits.log_softmax(-1)  # [1, seq_len, vocab]

    specials = set(getattr(tokenizer, "all_special_ids", []) or [])
    banned = set(banned_token_ids or []) | specials

    # Precompute top-k candidates at each mask position
    per_mask_cands = []
    for pos in mask_positions:
        lp = log_probs[0, pos].clone()
        if banned:
            idx = torch.tensor(sorted(banned), dtype=torch.long)
            lp.index_fill_(0, idx, float("-inf"))
        topk = torch.topk(lp, k=min(top_k_per_mask, lp.numel()))
        per_mask_cands.append((topk.indices.tolist(), topk.values.tolist()))

    # Beam over masks (left-to-right in token order)
    beam = [(0.0, [])]  # (cum_logprob, chosen_token_ids_so_far)
    for cand_ids, cand_lps in per_mask_cands:
        new_beam = []
        for cum_lp, chosen in beam:
            for tid, lp in zip(cand_ids, cand_lps):
                new_beam.append((cum_lp + float(lp), chosen + [tid]))
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:beam_size]

    # Materialize and deduplicate by decoded text
    best = {}
    for cum_lp, choice_ids in beam:
        filled = input_ids.clone()
        for pos, tid in zip(mask_positions, choice_ids):
            filled[0, pos] = tid
        text_out = tokenizer.decode(filled[0], skip_special_tokens=True)
        prev = best.get(text_out)
        if (prev is None) or (cum_lp > prev["score"]):
            best[text_out] = {
                "text": text_out,
                "score": cum_lp,
                "tokens": tokenizer.convert_ids_to_tokens(choice_ids),
                "token_ids": choice_ids,
                "mask_positions": mask_positions,
            }
    return sorted(best.values(), key=lambda r: r["score"], reverse=True)[:max_candidates]

def variable_length_infill(
    model, tokenizer,
    masked_template: str,              # contains ONE [MASK] span
    length_options: Iterable[int] = (1, 2, 3, 4),
    per_length_topk: int = 10,
    normalize_by_masks: bool = True,   # de-bias longer spans
    **beam_kwargs
) -> List[Dict]:
    mask_tok = tokenizer.mask_token
    assert mask_tok in masked_template, "Template must contain one [MASK] span."
    rows = []
    for L in length_options:
        expanded = masked_template.replace(mask_tok, " ".join([mask_tok]*L), 1)
        outs = fill_masks_beam(model, tokenizer, expanded, **beam_kwargs)
        for o in outs[:per_length_topk]:
            score = o["score"] / L if normalize_by_masks else o["score"]
            rows.append({"text": o["text"], "length": L, "score": score, "raw_score": o["score"], "tokens": o["tokens"]})
    return sorted(rows, key=lambda r: r["score"], reverse=True)

def beam_outputs_to_phrases(
    outputs: List[Dict],
    tokenizer: PreTrainedTokenizerBase,
    original_phrase: Optional[str] = None,
    lowercase: bool = True,
    unique: bool = True,
) -> List[str]:
    """
    Convert MLM beam outputs (from fill_masks_beam / variable_length_infill)
    into a list of phrases, similar to parse_paraphrases.

    Assumes that `o["tokens"]` are the tokens for the masked span
    (which is true for variable_length_infill, and for single-span masks).
    """
    phrases: List[str] = []

    # Prepare comparison baseline (like parse_paraphrases)
    if original_phrase is not None and lowercase:
        original_cmp = original_phrase.lower()
    else:
        original_cmp = original_phrase

    for o in outputs:
        # Decode just the predicted tokens for the masked span
        candidate = tokenizer.convert_tokens_to_string(o["tokens"])

        # Compare and optionally lowercase
        cand_cmp = candidate.lower() if lowercase else candidate

        # Drop suggestions that are the same as the original phrase
        if original_cmp is not None and cand_cmp == original_cmp:
            continue

        phrases.append(cand_cmp if lowercase else candidate)

    # Deduplicate (like your set(), but preserve order)
    if unique:
        seen = set()
        deduped = []
        for p in phrases:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        return deduped

    return phrases

def beam_outputs_to_phrases(
    outputs: List[Dict],
    tokenizer: PreTrainedTokenizerBase,
    original_phrase: Optional[str] = None,
    lowercase: bool = True,
    unique: bool = True,
) -> List[Dict]:
    """
    Convert MLM beam outputs (from fill_masks_beam / variable_length_infill)
    into a list of distinct paraphrase‑records (dicts) preserving tokens.
    Each record: {"text": ..., "tokens": [...]}.
    If unique=True: deduplicate by text (after lowercasing/comparison).
    """
    records: List[Dict] = []
    orig_cmp = original_phrase.lower() if (original_phrase is not None and lowercase) else original_phrase

    for o in outputs:
        tokens = o.get("tokens")
        if not tokens:
            continue
        text = tokenizer.convert_tokens_to_string(tokens)
        comp = text.lower() if lowercase else text
        if orig_cmp is not None and comp == orig_cmp:
            continue
        records.append({"text": text, "tokens": tokens})

    if not unique:
        return records

    seen = set()
    deduped: List[Dict] = []
    for rec in records:
        comp = rec["text"].lower() if lowercase else rec["text"]
        if comp not in seen:
            seen.add(comp)
            deduped.append(rec)
    return deduped

@torch.no_grad()
def fill_masks_beam_phrases(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    masked_text: str,
    original_phrase: Optional[str] = None,
    lowercase: bool = True,
    top_k_per_mask: int = 25,
    beam_size: int = 100,
    max_candidates: int = 50,
    banned_token_ids: Optional[Iterable[int]] = None,
) -> List[str]:
    """
    Convenience wrapper around fill_masks_beam that returns only the
    predicted phrases (like parse_paraphrases).
    """
    beam_outputs = fill_masks_beam(
        model=model,
        tokenizer=tokenizer,
        masked_text=masked_text,
        top_k_per_mask=top_k_per_mask,
        beam_size=beam_size,
        max_candidates=max_candidates,
        banned_token_ids=banned_token_ids,
    )

    return beam_outputs_to_phrases(
        outputs=beam_outputs,
        tokenizer=tokenizer,
        original_phrase=original_phrase,
        lowercase=lowercase,
        unique=True,
    )

def variable_length_infill_phrases(
    model,
    tokenizer,
    masked_template: str,               # contains ONE [MASK] span
    original_phrase: Optional[str] = None,
    lowercase: bool = True,
    length_options: Iterable[int] = (1, 2, 3, 4),
    per_length_topk: int = 10,
    normalize_by_masks: bool = True,
    **beam_kwargs,
) -> List[str]:
    """
    Run variable_length_infill and return only the span phrases as a list,
    similar to parse_paraphrases.
    """

    rows = variable_length_infill(
        model=model,
        tokenizer=tokenizer,
        masked_template=masked_template,
        length_options=length_options,
        per_length_topk=per_length_topk,
        normalize_by_masks=normalize_by_masks,
        **beam_kwargs,
    )
    # rows are dicts with at least: {"text", "length", "score", "raw_score", "tokens"}

    return beam_outputs_to_phrases(
        outputs=rows,        # compatible: each row has "tokens"
        tokenizer=tokenizer,
        original_phrase=original_phrase,
        lowercase=lowercase,
        unique=True,
    )

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
      - phrase_method  ("mask_fill", "varlen_fill", or "both"; None for the reference)
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
            "phrase_method": None,
            "original_phrase": orig,
            "phrase": orig,
            "tokens": orig_tokens,
            "num_tokens": len(orig_tokens) if orig_tokens is not None else None
        })
        # Then add paraphrase rows
        for p in paraphrase_entries:
            text = p.get("text")
            method = p.get("method")
            tokens = p.get("tokens")
            num_tokens = len(tokens) if tokens is not None else None
            rows.append({
                "phrase_num": phrase_num,
                "phrase_type": "paraphrase",
                "phrase_method": method,
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
        phrase_list = list(ast.literal_eval(phrase_raw))
        num_phrase_tokens = len(phrase_list)
        
        # Potentially use for starting point for variable
        max_starting_point = max(args.ngram_n, num_phrase_tokens)
        
        # Mask the phrase in the text
        masked_data = mask_phrase(known_text, phrase_list, tokenizer, "[MASK]")
        string_based_masked_list = masked_data['single_mask']
        token_based_masked_list = masked_data['multi_mask']

        paraphrase_records = []  

        for i in range(len(string_based_masked_list)):
            # single-/multi‑mask fill
            token_paraphrases = fill_masks_beam_phrases(
                model=model, tokenizer=tokenizer,
                masked_text=token_based_masked_list[i],
                original_phrase=phrase_pretty,
                lowercase=True,
                top_k_per_mask=25, beam_size=100, max_candidates=50
            )
            for p in token_paraphrases:
                paraphrase_records.append({ "text": p['text'], "tokens": p['tokens'], "method": "mask_fill" })

            # variable‑length (string‑based) fill
            string_paraphrases = variable_length_infill_phrases(
                model=model, tokenizer=tokenizer,
                masked_template=string_based_masked_list[i],
                original_phrase=phrase_pretty,
                lowercase=True,
                length_options=tuple(range(args.ngram_n, num_phrase_tokens + 1)),
                per_length_topk=10, normalize_by_masks=True
            )
            for p in string_paraphrases:
                paraphrase_records.append({ "text": p['text'], "tokens": p['tokens'], "method": "varlen_fill" })

        # Merge duplicates, annotate if both methods generated same paraphrase
        merged = {}
        for rec in paraphrase_records:
            txt = rec["text"]
            tok = rec["tokens"]
            method = rec["method"]
            if txt not in merged:
                merged[txt] = { "text": txt, "tokens": tok, "method": method }
            else:
                prev = merged[txt]
                prev_m = prev["method"]
                if prev_m != method:
                    prev["method"] = "both"

        # Save to your dictionary
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