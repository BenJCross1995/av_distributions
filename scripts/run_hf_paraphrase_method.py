#!/usr/bin/env python3

import argparse
import sys
import json
import re
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
from from_root import from_root
import pandas as pd

# Ensure we can import from src/
sys.path.insert(0, str(from_root("src")))

from read_and_write_docs import read_jsonl, read_rds
from model_loading import load_model, distinct_special_chars, load_model_efficient
from utils import apply_temp_doc_id, build_metadata_df
from n_gram_functions import (
    common_ngrams,
    filter_ngrams,
    pretty_print_common_ngrams,
    get_scored_df,
    get_scored_df_no_context
)
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
    """Extract paraphrases from OpenAI-like response (JSON mode)."""
    paraphrase_list = []
    for i in range(1, len(response.choices)):
        content = response.choices[i].message.content
        try:
            content_json = json.loads(content)
            for para in content_json['paraphrases']:
                if para != phrase:
                    if (lowercase) & (para.lower() != phrase):
                        paraphrase_list.append(para.lower())
                    else:
                        paraphrase_list.append(para)
        except Exception:
            continue
    unique_list = list(set(paraphrase_list))
    return unique_list

# --------------------
# HF Generation utilities
# --------------------

@dataclass
class _Msg:
    content: str

@dataclass
class _Choice:
    message: _Msg

@dataclass
class _FakeResponse:
    choices: List[_Choice]

def _extract_json_string(text: str) -> Optional[str]:
    """
    Be tolerant to extra text around JSON. Grab the first top-level {...} block.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1].strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            return None
    return None

def _apply_chat_template_if_available(tokenizer, system_prompt: str, user_prompt: str) -> str:
    """
    Use tokenizer.chat_template when available for better model formatting;
    otherwise fall back to a plain concatenated prompt.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        # Works for chat/instruct models that ship a template
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback for plain decoder models
        return f"{system_prompt}\n\n{user_prompt}\n\n"

def paraphrase_with_hf(
    tokenizer,
    model,
    system_prompt: str,
    user_prompt: str,
    n: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    base_seed: int | None = None,   # <- optional: reproducible but varied outputs
) -> _FakeResponse:
    """
    Generate paraphrases with a local HF model, **one sample per loop**.

    - Lower peak memory than num_return_sequences>1 (which generates in parallel).
    - Optional `base_seed` for reproducibility: uses (base_seed + i) per sample.
    - Returns an OpenAI-like response object so parse_paraphrases() works unchanged.
    """

    prompt = _apply_chat_template_if_available(tokenizer, system_prompt, user_prompt)
    device = next(model.parameters()).device

    # Tokenize once; reuse on-device tensors for each iteration
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pad_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    texts: list[str] = []
    with torch.inference_mode():
        for i in range(max(1, n)):
            # Per-sample generator (so seeds differ but stay reproducible if base_seed is set)
            generator = None
            if base_seed is not None:
                generator = torch.Generator(device=device).manual_seed(base_seed + i)

            out = model.generate(
                **inputs,
                do_sample=True,                     # enable sampling (temp/top_p/top_k take effect)
                temperature=max(0.01, temperature),
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,             # <- one at a time
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_id,
                return_dict_in_generate=True,
                generator=generator,                # reproducible randomness if provided
            )

            # Keep only generated tokens (drop the prompt)
            gen_only = out.sequences[:, inputs["input_ids"].shape[-1]:]
            t = tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0]
            texts.append(t)

    # Build OpenAI-like response with a dummy first choice (your parser starts at 1)
    choices = [_Choice(_Msg(""))]
    for t in texts:
        jtxt = _extract_json_string(t) or json.dumps({"paraphrases": [t.strip()]})
        choices.append(_Choice(_Msg(jtxt)))

    return _FakeResponse(choices=choices)

# --------------------
# CLI
# --------------------

def parse_args():
    ap = argparse.ArgumentParser(description="HF N-gram paraphrase pipeline (local model)")

    # Paths
    ap.add_argument("--known_loc")
    ap.add_argument("--unknown_loc")
    ap.add_argument("--metadata_loc")
    ap.add_argument("--model_loc")  # n-gram/token scoring model (kept)
    ap.add_argument("--paraphrase_model_loc", required=True, help="Path to local HF chat/causal model for paraphrasing")
    ap.add_argument("--save_loc")
    ap.add_argument("--completed_loc", default=None)

    # Dataset hinting
    ap.add_argument("--corpus", default="Wiki")
    ap.add_argument("--data_type", default="training")
    ap.add_argument("--known_doc")
    ap.add_argument("--unknown_doc")

    # Prompts
    ap.add_argument("--prompt_loc", default=str(from_root("prompts", "exhaustive_constrained_ngram_paraphraser_prompt_JSON_new.txt")))

    # N-gram
    ap.add_argument("--ngram_n", type=int, default=2)
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--no_lowercase", dest="lowercase", action="store_false")
    ap.set_defaults(lowercase=True)
    ap.add_argument("--order", default="len_desc", help="Order for pretty_print_common_ngrams")
    ap.add_argument("--score_texts", action="store_true")

    # HF generation params (keeping your OpenAI-like names)
    ap.add_argument("--max_tokens", type=int, default=512, help="HF max_new_tokens")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--n", type=int, default=10)

    # Efficient loader options
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model if supported")

    return ap.parse_args()

# --------------------
# Main
# --------------------

def main():
    args = parse_args()

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

    # Model used for n-gram/token scoring (kept as-is)
    print("Loading tokenizer/model for n-gram scoring")
    tokenizer, model = load_model(args.model_loc)
    special_tokens = distinct_special_chars(tokenizer=tokenizer)

    # Paraphrasing model (HF local)
    print("Loading local HF model for paraphrasing")
    para_tokenizer, para_model = load_model_efficient(
        model_path=args.paraphrase_model_loc,
        device=args.device,
        dtype=args.dtype,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        compile_model=args.compile,
    )

    print("Loading data")
    known = read_jsonl(args.known_loc)
    known = apply_temp_doc_id(known)

    unknown = read_jsonl(args.unknown_loc)
    unknown = apply_temp_doc_id(unknown)

    print("Data loaded")

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

    # Filter to remove smaller n-grams which don't satisfy the rules
    common = filter_ngrams(common, special_tokens=special_tokens)
    n_gram_list = pretty_print_common_ngrams(common, tokenizer=tokenizer, order=args.order, return_format='flat', show_raw=True)
    print(f"There are {len(n_gram_list)} n-grams in common!")

    # -----
    # HF paraphrasing (local)
    # -----
    print("Generating paraphrases with local HF model")
    n_gram_dict = {}
    total_phrases = len(n_gram_list)
    width = len(str(total_phrases))  # e.g., 10 -> 2, 100 -> 3
    
    system_prompt = create_system_prompt(args.prompt_loc)

    for idx, (phrase_pretty, phrase_raw) in enumerate(n_gram_list, start=1):
        print(f"Paraphrase {idx} out of {total_phrases}", flush=True)
        user_prompt = create_user_prompt(known_text, phrase_pretty, raw_phrase=phrase_raw)
        response = paraphrase_with_hf(
            para_tokenizer,
            para_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            n=args.n,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        paraphrases = parse_paraphrases(response, phrase_pretty)
        key = f"phrase_{idx:0{width}d}"
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
            use_xlookup=False,
            highlight_phrases=False
        )

    else:
        print("Not scoring texts")
        print("<<<RESULT_JSON_START>>>")
        print(json.dumps(n_gram_dict))
        print("<<<RESULT_JSON_END>>>")

if __name__ == "__main__":
    main()
