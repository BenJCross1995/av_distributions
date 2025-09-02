import re
import pandas as pd

from itertools import product

def create_temp_doc_id(input_text):
    """Create a new doc id by preprocessing the current id"""
    
    # Extract everything between the brackets
    match = re.search(r'\[(.*?)\]', input_text)
    
    if match:
        extracted_text = match.group(1)
        # Replace all punctuation and spaces with "_"
        cleaned_text = re.sub(r'[^\w]', '_', extracted_text)
        # Replace multiple underscores with a single "_"
        final_text = re.sub(r'_{2,}', '_', cleaned_text)
        return final_text.lower()
        
    return None

def apply_temp_doc_id(df):
    """Apply the doc id function on the dataframe"""

    # If both ID columns already exist then return df as it is
    if 'doc_id' in df.columns and 'orig_doc_id' in df.columns:
        return df
        
    # Rename doc_id to orig_doc_id first    
    df.rename(columns={'doc_id': 'orig_doc_id'}, inplace=True)

    # Create the new doc_id column directly
    df['doc_id'] = df['orig_doc_id'].apply(create_temp_doc_id)

    # df.drop("orig_doc_id", axis=1, inplace=True)
    
    # Move the new doc_id column to the front
    cols = ['doc_id', 'orig_doc_id'] + [col for col in df.columns if col not in ['doc_id', 'orig_doc_id', 'text']] + ['text']

    df = df[cols]

    return df

def build_metadata_df(filtered_metadata: pd.DataFrame,
                      known_df: pd.DataFrame,
                      unknown_df: pd.DataFrame) -> pd.DataFrame:
    """
    From filtered_metadata (with columns problem, corpus, known_author, unknown_author)
    and known_df (with columns author, doc_id), build a metadata table exploded so that
    each known_doc_id gets its own row, and assign a running sample_id.
    """
    # Step 1: build the initial DataFrame with a list-column
    records = []
    for _, met in filtered_metadata.iterrows():
        problem        = met['problem']
        corpus         = met['corpus']
        known_author   = met['known_author']
        unknown_author = met['unknown_author']

        # collect all doc_ids for this author
        doc_ids = known_df.loc[
            known_df['author'] == known_author,
            'doc_id'
        ].unique().tolist()

        unknown_doc_id = unknown_df.loc[
            unknown_df['author'] == unknown_author,
            'doc_id'
        ].iloc[0]
        
        records.append({
            'problem':        problem,
            'corpus':         corpus,
            'known_author':   known_author,
            'unknown_author': unknown_author,
            'unknown_doc_id': unknown_doc_id,
            'known_doc_ids':  doc_ids
        })

    meta = pd.DataFrame(records)

    # Step 2: explode the list-column into individual rows
    exploded = (
        meta
        .explode('known_doc_ids')
        .rename(columns={'known_doc_ids': 'known_doc_id'})
        .reset_index(drop=True)
    )

    # Step 3: add sample_id starting at 1
    exploded.insert(0, 'sample_id', range(1, len(exploded) + 1))

    return exploded

def problem_data_prep(
    unknown: pd.DataFrame,
    known: pd.DataFrame,
    metadata: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Build a DataFrame of (known vs. unknown) author pairs for LambdaG.

    If `metadata` is provided, it must contain columns
    `'problem'`, `'known_author'` and `'unknown_author'`, and only rows
    where both authors appear in `known['author']` and
    `unknown['author']` are kept.

    Otherwise, if exactly one known author exists, all pairings
    with each unknown author are generated and a `problem` column
    is added by concatenating "known_author vs unknown_author".

    Parameters:
    - unknown (pandas.DataFrame): DataFrame of questioned (disputed) documents.
    - known   (pandas.DataFrame): DataFrame of known (undisputed) documents.
    - refs    (pandas.DataFrame): DataFrame of reference documents, can be the same as known.

    Returns:
    - pandas.DataFrame: A dataframe of possible samples for the LambdaG method
    """
    known_authors   = known['author'].unique().tolist()
    unknown_authors = unknown['author'].unique().tolist()

    # Use metadata if available
    if metadata is not None and not metadata.empty:
        problem_df = (
            metadata.loc[
                metadata['known_author'].isin(known_authors) &
                metadata['unknown_author'].isin(unknown_authors),
                ['problem','known_author','unknown_author']
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    # If multiple known authors but no metadata, that's ambiguous
    elif len(known_authors) > 1:
        raise ValueError(
            f"There are {len(known_authors)} known authors but no metadata provided"
        )

    # Single known author: generate all vs. unknown
    elif len(known_authors) == 1 and unknown_authors:
        pairs = [
            (k, u)
            for k, u in product(known_authors, unknown_authors)
        ]
        problem_df = pd.DataFrame(pairs, columns=['known_author', 'unknown_author'])
        problem_df.insert(
            loc=0,
            column='problem',
            value=problem_df['known_author'] + ' vs ' + problem_df['unknown_author']
        )

    else:
        raise ValueError("No valid author pairs could be generated")

    print(f"    There are {len(known_authors)} known author(s) and {len(problem_df)} problem(s) in the dataset.")
    return problem_df