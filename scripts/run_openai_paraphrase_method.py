#!/usr/bin/env python3

import argparse
import sys
import json
import re
import os
from from_root import from_root

import pandas as pd

# Ensure we can import from src/
sys.path.insert(0, str(from_root("src")))

from read_and_write_docs import read_jsonl, read_rds
from tokenize_and_score import load_model
from utils import apply_temp_doc_id, build_metadata_df
from n_gram_functions import (
    common_ngrams,
    pretty_print_common_ngrams,
    get_scored_df,
    get_scored_df_no_context
)
from open_ai import initialise_client, llm

# --------------------
# Helpers
# --------------------

# remove illegal control chars (keep \t, \n, \r)
_ILLEGAL_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")

def _clean_cell(x):
    if isinstance(x, str):
        return _ILLEGAL_RE.sub("", x)
    return x

def clean_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].applymap(_clean_cell)
    return df

def create_system_prompt(prompt_loc):
    """Reads the prompt as a .txt file for better versioning"""
    with open(prompt_loc, "r", encoding="utf-8") as f:
        return f.read()
    
def create_user_prompt(known_text, phrase):
    """The method of input to the LLM as described in the system prompt"""
    user_prompt = f"""
<DOC>
{known_text}
</DOC>
<NGRAM>
"{phrase}"
</NGRAM>
"""
    
    return user_prompt

def parse_paraphrases(response, phrase):
    """Extract paraphrases from OpenAI response (JSON mode)."""
    paraphrase_list = []
    for i in range(1, len(response.choices)):
        content = response.choices[i].message.content
        
        try:
            content_json = json.loads(content)
            for para in content_json['paraphrases']:
                if para != phrase:
                    paraphrase_list.append(para)  
        except Exception:
            continue
        
    unique_list = list(set(paraphrase_list))
    
    return unique_list

def parse_args():
    ap = argparse.ArgumentParser(description="OpenAI N-gram paraphrase pipeline")
    # Paths
    ap.add_argument("--known_loc")
    ap.add_argument("--unknown_loc")
    ap.add_argument("--metadata_loc")
    ap.add_argument("--model_loc")
    ap.add_argument("--save_loc")
    # Dataset hinting
    ap.add_argument("--corpus", default="Wiki")
    ap.add_argument("--data_type", default="training")
    ap.add_argument("--known_doc")
    ap.add_argument("--unknown_doc")
    # Env
    ap.add_argument("--credentials_loc", default=str(from_root("credentials.json")))
    ap.add_argument("--prompt_loc", default=str(from_root("prompts", "exhaustive_constrained_ngram_paraphraser_prompt_JSON.txt")))
    # N-gram
    ap.add_argument("--ngram_n", type=int, default=2)
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--no_lowercase", dest="lowercase", action="store_false")
    ap.set_defaults(lowercase=True)
    ap.add_argument("--order", default="len_desc", help="Order for pretty_print_common_ngrams")
    # OpenAI
    ap.add_argument("--openai_model", default="gpt-4.1")
    ap.add_argument("--max_tokens", type=int, default=5000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--n", type=int, default=10)

    return ap.parse_args()

def main():
    
    args=parse_args()
    
    # Ensure the directory exists before beginning
    os.mkdir(args.save_loc, exist_ok=True)
    
    # -----
    # LOAD DATA & LOCAL MODEL
    # -----
    specific_problem = f"{args.known_doc} vs {args.unknown_doc}"
    save_loc = f"{args.save_loc}/{specific_problem}.xlsx"
    
    # Skip the problem if already exists
    if os.path.exists(save_loc):
        print(f"Path {save_loc} already exists. Exiting.")
        sys.exit()
    
    print(f"Working on problem: {specific_problem}")
    
    print("Loading model")
    tokenizer, model = load_model(args.model_loc)
    
    print("Loading data")
    known = read_jsonl(args.known_loc)
    known = apply_temp_doc_id(known)
    
    unknown = read_jsonl(args.unknown_loc)
    unknown = apply_temp_doc_id(unknown)
    
    # NOTE - Is this used?
    metadata = read_rds(args.metadata_loc)
    filtered_metadata = metadata[metadata['corpus'] == args.corpus]
    agg_metadata = build_metadata_df(filtered_metadata, known, unknown)
    
    # -----
    # Get the chosen text & metadata
    # -----
    
    known_text = known[known['doc_id'] == args.known_doc].reset_index().loc[0, 'text']
    unknown_text = unknown[unknown['doc_id'] == args.unknown_doc].reset_index().loc[0, 'text']
    
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
    n_gram_list = pretty_print_common_ngrams(common, tokenizer=tokenizer, order=args.order, return_format='flat')
    print(f"There are {len(n_gram_list)} n-grams in common!")
    
    # -----
    # OpenAI bits
    # -----
    
    print("Generating paraphrases")
    client = initialise_client(args.credentials_loc)
    
    n_gram_dict = {}
    width = len(str(len(n_gram_list)))  # e.g., 10 -> 2, 100 -> 3

    for idx, phrase in enumerate(n_gram_list, start=1):
        user_prompt = create_user_prompt(known_text, phrase)
        response = llm(
            create_system_prompt(args.prompt_loc),
            user_prompt,
            client,
            model=args.openai_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            n=args.n,
            response_format={"type": "json_object"},
        )
        paraphrases = parse_paraphrases(response, phrase)
        key = f"phrase_{idx:0{width}d}"  # -> phrase_01, phrase_002, etc.
        n_gram_dict[key] = {"phrase": phrase, "paraphrases": paraphrases}
        
    # -----
    # Score phrases
    # -----
    
    print("Scoring phrases")
    print("    Scoring known text")
    known_scored = get_scored_df(n_gram_dict, known_text, tokenizer, model)
    
    print("    Scoring unknown text")
    unknown_scored = get_scored_df(n_gram_dict, unknown_text, tokenizer, model)
    
    print("    Scoring phrases with no context")
    score_df_no_context = get_scored_df_no_context(n_gram_dict, tokenizer, model)
    
    # -----
    # Final cleaning and saving
    # -----
    
    print(f"Writing file: {specific_problem}")
    distinct_phrases = score_df_no_context[['phrase_num', 'original_phrase']].drop_duplicates()
        
    with pd.ExcelWriter(save_loc, engine="openpyxl") as xls:
        
        clean_for_excel(docs_df).to_excel(xls, sheet_name="docs", index=False)
        clean_for_excel(problem_metadata).to_excel(xls, sheet_name="metadata", index=False)
        clean_for_excel(score_df_no_context).to_excel(xls, sheet_name="no context", index=False)
        clean_for_excel(known_scored).to_excel(xls, sheet_name="known", index=False)
        clean_for_excel(unknown_scored).to_excel(xls, sheet_name="unknown", index=False)
        clean_for_excel(distinct_phrases).to_excel(xls, sheet_name="LLR", index=False)
        
if __name__ == "__main__":
    main()