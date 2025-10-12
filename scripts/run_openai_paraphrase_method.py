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
from model_loading import load_model, distinct_special_chars
from utils import apply_temp_doc_id, build_metadata_df
from n_gram_functions import (
    common_ngrams,
    filter_ngrams,
    pretty_print_common_ngrams,
    get_scored_df,
    get_scored_df_no_context
)
from open_ai import initialise_client, llm
from excel_functions import create_excel_template

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
    
def create_user_prompt(known_text, phrase, raw_phrase):
    """The method of input to the LLM as described in the system prompt"""
    user_prompt = f"""
<DOC>
{known_text}
</DOC>
<RAW NGRAM>
"{raw_phrase}"
</RAW NGRAM>
<NGRAM>
"{phrase}"
</NGRAM>
"""
    
    return user_prompt

def parse_paraphrases(response, phrase, lowercase=True):
    """Extract paraphrases from OpenAI response (JSON mode)."""
    paraphrase_list = []
    for i in range(1, len(response.choices)):
        content = response.choices[i].message.content
        
        try:
            content_json = json.loads(content)
            for para in content_json['paraphrases']:
                if para != phrase:
                    if lowercase:
                        paraphrase_list.append(para.lower())
                    else:
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
    ap.add_argument("--completed_loc", default=None)
    # Dataset hinting
    ap.add_argument("--corpus", default="Wiki")
    ap.add_argument("--data_type", default="training")
    ap.add_argument("--known_doc")
    ap.add_argument("--unknown_doc")
    # Env
    ap.add_argument("--credentials_loc", default=str(from_root("credentials.json")))
    ap.add_argument("--prompt_loc", default=str(from_root("prompts", "exhaustive_constrained_ngram_paraphraser_prompt_JSON_new.txt")))
    # N-gram
    ap.add_argument("--ngram_n", type=int, default=2)
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--no_lowercase", dest="lowercase", action="store_false")
    ap.set_defaults(lowercase=True)
    ap.add_argument("--order", default="len_desc", help="Order for pretty_print_common_ngrams")
    ap.add_argument("--score_texts", action="store_true")
    # OpenAI
    ap.add_argument("--openai_model", default="gpt-4.1")
    ap.add_argument("--max_tokens", type=int, default=5000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--n", type=int, default=10)

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
    
    # -----
    # OpenAI bits
    # -----
    
    print("Generating paraphrases")
    client = initialise_client(args.credentials_loc)
    
    n_gram_dict = {}
    width = len(str(len(n_gram_list)))  # e.g., 10 -> 2, 100 -> 3

    for idx, (phrase_pretty, phrase_raw) in enumerate(n_gram_list, start=1):
        user_prompt = create_user_prompt(known_text, phrase_pretty, raw_phrase=phrase_raw)
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
        paraphrases = parse_paraphrases(response, phrase_pretty)
        key = f"phrase_{idx:0{width}d}"  # -> phrase_01, phrase_002, etc.
        n_gram_dict[key] = {"phrase": phrase_pretty, "paraphrases": paraphrases}
        
    # -----
    # Score phrases
    # -----
    if args.score_texts:
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
        
        # Run the new Excel function
        create_excel_template(
            known=known_scored,
            unknown=unknown_scored,
            no_context=score_df_no_context,
            metadata=problem_metadata,
            docs=docs_df,
            path=save_loc,
            known_sheet="known",
            unknown_sheet="unknown",
            nc_sheet="no context",
            metadata_sheet="metadata",
            docs_sheet="docs",
            llr_sheet="LLR",
            use_xlookup=False
        )

#         print(f"Writing file: {specific_problem}")
#         llr_cols = ['phrase_num', 'phrase_occurence', 'original_phrase']

#         distinct_phrases = (
#             pd.concat([unknown_scored[llr_cols], known_scored[llr_cols]], ignore_index=True)
#             .drop_duplicates()
#             .sort_values(['phrase_num', 'phrase_occurence'], kind='mergesort')
#             .reset_index(drop=True)
# )
            
#         with pd.ExcelWriter(save_loc, engine="openpyxl") as xls:
            
#             clean_for_excel(docs_df).to_excel(xls, sheet_name="docs", index=False)
#             clean_for_excel(problem_metadata).to_excel(xls, sheet_name="metadata", index=False)
#             clean_for_excel(score_df_no_context).to_excel(xls, sheet_name="no context", index=False)
#             clean_for_excel(known_scored).to_excel(xls, sheet_name="known", index=False)
#             clean_for_excel(unknown_scored).to_excel(xls, sheet_name="unknown", index=False)
#             clean_for_excel(distinct_phrases).to_excel(xls, sheet_name="LLR", index=False)
    
    else:
        print("Not scoring texts")
        print("<<<RESULT_JSON_START>>>")
        print(json.dumps(n_gram_dict))
        print("<<<RESULT_JSON_END>>>")
    
if __name__ == "__main__":
    main()