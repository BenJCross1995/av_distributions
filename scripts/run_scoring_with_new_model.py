import sys
import pandas as pd

from pathlib import Path
from from_root import from_root
from typing import Iterable

sys.path.insert(0, str(from_root("src")))

from read_and_write_docs import read_excel_sheets
from model_loading import load_model
from n_gram_functions import (
    get_scored_df,
    get_scored_df_no_context
)
from excel_functions import create_excel_template

def build_write_path(data_loc: Path, save_dir: Path) -> Path:
    """
    Build the output Excel path using the input file's stem and the save directory,
    always appending '.xlsx'.
    """
    return save_dir / f"{data_loc.stem}.xlsx"


def process_file(
    data_loc: str | Path,
    save_dir: str | Path,
    model_loc: str | Path,
    phrase_loc: str | Path = None,
    sheet_names: Iterable[str] = ["docs", "no context", "metadata"],
    overwrite: bool = False,
) -> Path:
    """
    End-to-end pipeline.

    - Resolves write_path FIRST from data_loc + save_dir + '.xlsx'
    - If write_path exists and overwrite=False, exits early and returns the path
    - Ensures save_dir exists
    - Loads model, reads sheets, computes scores, writes Excel

    Returns:
        Path to the written (or existing) Excel file.
    """
    data_loc = Path(data_loc)
    save_dir = Path(save_dir)
    print(f"Working on file {data_loc.stem}.xlsx")
    
    # 1) Determine write_path FIRST
    write_path = build_write_path(data_loc, save_dir)

    # 2) If it already exists and we're not overwriting, exit early
    if write_path.exists() and not overwrite:
        print(f"[INFO] Output already exists, skipping: {write_path}")
        return write_path

    # 3) Ensure the save directory exists
    save_dir.mkdir(parents=True, exist_ok=True)

    # 4) Load model
    tokenizer, model = load_model(model_loc)

    # 5) Read required sheets (defaults mirror your example)
    data = read_excel_sheets(data_loc, list(sheet_names))

    # --- Your original logic, kept faithful ---

    # Get the full texts from docs
    docs = data["docs"]
    texts = docs.tail(1).copy().reset_index()

    # Get the texts, will be the last row in docs
    known_text = texts.loc[0, "known"]
    unknown_text = texts.loc[0, "unknown"]

    # Pull only the needed columns from metadata
    metadata = data["metadata"]
    metadata_subset = metadata.loc[:, :"target"]

    no_context = data["no context"]

    # Get phrases to keep - only do this if phrase list exists
    if phrase_loc:
        
        # Read the location and just keep phrase column
        phrase_list = pd.read_excel(phrase_loc)
        phrases_to_keep = phrase_list[
            (phrase_list['keep_phrase'] == 1) | (phrase_list['keep_phrase'].isna())
        ].copy()
        phrases_to_keep = phrases_to_keep[['phrase']]
        
        reference_phrases = no_context[no_context['phrase_type'] == 'reference'].copy()

        # Perform the merge using the tuple-based key
        merged_phrases = pd.merge(reference_phrases, phrases_to_keep, on='phrase', how='inner')
        merged_phrases = merged_phrases[['phrase_num']]

        # Now no_context only includes relevant phrases
        no_context = pd.merge(no_context, merged_phrases, on='phrase_num', how='inner')
        
    # Get the n-grams in a way necessary to compute scores
    print("Getting the n-gram dictionary")
    n_gram_dict = {}
    for phrase_num, df in no_context.groupby("phrase_num"):
        ref_series = df.loc[df["phrase_type"] == "reference", "phrase"]
        reference_phrase = ref_series.iloc[0] if not ref_series.empty else None
        paraphrases = df.loc[df["phrase_type"] == "paraphrase", "phrase"].tolist()

        n_gram_dict[phrase_num] = {
            "phrase": reference_phrase,
            "paraphrases": paraphrases,
        }

    # Score the n-grams vs the text
    print("Scoring known text.")
    known_scored = get_scored_df(n_gram_dict, known_text, tokenizer, model)
    
    print("Scoring unknown text.")
    unknown_scored = get_scored_df(n_gram_dict, unknown_text, tokenizer, model)
    
    print("Scoring no context text.")
    scored_no_context = get_scored_df_no_context(n_gram_dict, tokenizer, model)

    # Write the Excel template
    create_excel_template(
        known=known_scored,
        unknown=unknown_scored,
        no_context=scored_no_context,
        metadata=metadata_subset,
        docs=docs,
        path=write_path,
        known_sheet="known",
        unknown_sheet="unknown",
        nc_sheet="no context",
        metadata_sheet="metadata",
        docs_sheet="docs",
        llr_sheet="LLR",
        use_xlookup=False,
    )

    print(f"[INFO] Wrote: {write_path}")
    return write_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run scoring pipeline and write Excel output.")
    parser.add_argument("--data_loc", help="Path to the input Excel file (source data).")
    parser.add_argument("--save_dir", help="Directory to save the output .xlsx file.")
    parser.add_argument("--model_loc",help="Model directory")
    parser.add_argument("--phrase_loc", help="Location of a file containing phrases to filter out.")
    parser.add_argument(
        "--sheet",
        dest="sheets",
        action="append",
        help=f"Sheet names to load (can be repeated). Default: {', '.join(["docs", "no context", "metadata"])}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    args = parser.parse_args()

    sheets = args.sheets if args.sheets else ["docs", "no context", "metadata"]
    process_file(
        data_loc=args.data_loc,
        save_dir=args.save_dir,
        model_loc=args.model_loc,
        sheet_names=sheets,
        overwrite=args.overwrite,
    )
