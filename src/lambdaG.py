import math
import random
import sys

import numpy as np
import pandas as pd

from collections import Counter, defaultdict
from itertools import product, islice

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

def extract_ngrams(sentences, N):
    """Return Counter of all k‑grams (size 1..N) across sentences, padded."""
    counts = {n: Counter() for n in range(1, N+1)}
    for sent in sentences:
        tokens = ['<s>'] * (N-1) + sent + ['</s>']
        for n in range(1, N+1):
            for i in range(len(tokens)-n+1):
                ngram = tuple(tokens[i:i+n])
                counts[n][ngram] += 1
    return counts

def continuation_counts(counts, n):
    cont = Counter()
    seen = defaultdict(set)
    for w1_to_wn, c in counts[n].items():
        w_context = w1_to_wn[:-1]
        w_last = w1_to_wn[-1]
        seen[w_last].add(w_context)
    for w, ctxts in seen.items():
        cont[w] = len(ctxts)
    return cont

def total_distinct_bigrams(counts):
    return len([1 for _ in counts[2].keys()])

def kn_prob(w_tuple, counts, D=0.75, N=3):
    """
    Recursively calculate P_KN(w_N | w_1..w_{N-1})
    counts: dict n->Counter; D: discount
    """
    n = len(w_tuple)
    if n == 1:
        # unigram continuation prob: unique preceding contexts / total bigram types
        cont = continuation_counts(counts, 2)
        return cont[w_tuple[0]] / total_distinct_bigrams(counts)
    prev = w_tuple[:-1]
    c_full = counts[n][w_tuple]
    c_prev = sum(cnt for ng, cnt in counts[n].items() if ng[:-1] == prev)
    if c_prev == 0:
        return kn_prob(w_tuple[1:], counts, D, N)  # backed off
    first = max(c_full - D, 0) / c_prev
    # back-off weight λ
    unique_follow = len({ng[-1] for ng in counts[n] if ng[:-1] == prev})
    lam = D * unique_follow / c_prev
    return first + lam * kn_prob(w_tuple[1:], counts, D, N)

def sentence_prob(sent, counts, D=0.75, N=3):
    tokens = ['<s>'] * (N-1) + sent + ['</s>']
    probs = []
    for i in range(N-1, len(tokens)):
        context = tuple(tokens[i-N+1:i+1])
        p = kn_prob(context, counts, D, N)
        probs.append(p)
    return probs

def lambdaG(unknown, known, refs, metadata=None, N=10, r=30, cores=1, vectorise=False):
    """
    Run the LambdaG author‐verification method.

    Parameters:
    - unknown   (pandas.DataFrame): DataFrame of questioned (disputed) documents.
    - known     (pandas.DataFrame): DataFrame of known (undisputed) documents.
    - refs      (pandas.DataFrame): Reference corpus for calibration. This can be the same as known.
    - metadata  (pandas.DataFrame, optional): A dataframe of problem metadata, used if known contains more than one author.
    - N         (int, optional): Order of the model (n-gram length). Default is 10.
    - r         (int, optional): Number of iterations/bootstrap samples. Default is 30.
    - cores     (int, optional): Number of CPU cores for parallel processing. Default is 1.
    - vectorise (bool, optional): If True, splits documents into sentences before feature extraction. Default is False.

    Returns:
    - pandas.DataFrame: Uncalibrated log-likelihood ratios (LambdaG) for each document in `unknown`.
    """

    problem_df = problem_data_prep(unknown, known, metadata)

    if vectorise:
        print("Vectorising data into sentences")
        # NOTE - Need to add in vetorising method
        
    total = len(problem_df)
    results = []
    for idx, row in problem_df.iterrows():
        known_author = row['known_author']
        unknown_author = row['unknown_author']
        problem = row['problem']
        target = known_author == unknown_author
        
        print(f"        Working on problem {idx+1} of {total}: {problem}")

        # Filter the known and unknown for the current problem
        known_filtered = known[known['author'] == known_author]
        unknown_filtered = unknown[unknown['author'] == unknown_author]

        # Filter the reference dataset
        refs_filtered = refs[~refs['author'].isin([known_author, unknown_author])]

        known_sentences = known_filtered['tokens']
        unknown_sentences = unknown_filtered['tokens']

        num_known_sentences = len(known_sentences)
        num_unknown_sentences = len(unknown_sentences)

        if num_known_sentences > len(refs_filtered):
            raise ValueError(
                f"Not enough reference sentences ({len(refs_filtered)}) to sample {num_known_sentences}"
            )
        
        # turn the Series into a list first
        all_refs = refs_filtered['tokens'].tolist()

        known_counts = extract_ngrams(known_sentences, N)

        known_probs = []
        for q in unknown_sentences:
            ps = sentence_prob(q, known_counts, D=0.75, N=N)
            # replace zeros
            ps = [p if p > 0 else sys.float_info.min for p in ps]
            known_probs.append(ps)

        lambda_score = 0
        
        for _ in range(r):
            ref_sentences = random.sample(all_refs, num_known_sentences)

            ref_counts = extract_ngrams(ref_sentences, N)
            
            ref_probs = []
            for q in unknown_sentences:
                ps = sentence_prob(q, ref_counts, D=0.75, N=N)
                # replace zeros
                ps = [p if p > 0 else sys.float_info.min for p in ps]
                ref_probs.append(ps)

            lr_sum = 0.0
            for kp_sent, rp_sent in zip(known_probs, ref_probs):
                assert len(kp_sent) == len(rp_sent)
                for k, r_ in zip(kp_sent, rp_sent):
                    lr_sum += math.log10(k / r_)
        
            # 4. update lambda
            lambda_score += lr_sum / r

        results.append({
            'problem': problem,
            'known_author': known_author,
            'unknown_author': unknown_author,
            'target': target,
            'score': lambda_score
        })
        
    return pd.DataFrame(results)

def lambdaG_paraphrase(unknown, known, refs, metadata=None, N=10, r=30, cores=1, vectorise=False):
    """
    Run the LambdaG author‐verification method.

    Parameters:
    - unknown   (pandas.DataFrame): DataFrame of questioned (disputed) documents.
    - known     (pandas.DataFrame): DataFrame of known (undisputed) documents.
    - refs      (pandas.DataFrame): Reference corpus for calibration. This can be the same as known.
    - metadata  (pandas.DataFrame, optional): A dataframe of problem metadata, used if known contains more than one author.
    - N         (int, optional): Order of the model (n-gram length). Default is 10.
    - r         (int, optional): Number of iterations/bootstrap samples. Default is 30.
    - cores     (int, optional): Number of CPU cores for parallel processing. Default is 1.
    - vectorise (bool, optional): If True, splits documents into sentences before feature extraction. Default is False.

    Returns:
    - pandas.DataFrame: Uncalibrated log-likelihood ratios (LambdaG) for each document in `unknown`.
    """

    problem_df = problem_data_prep(unknown, known, metadata)

    if vectorise:
        print("Vectorising data into sentences")
        # NOTE - Need to add in vetorising method
        
    total = len(problem_df)
    results = []
    for idx, row in problem_df.iterrows():
        known_author = row['known_author']
        unknown_author = row['unknown_author']
        problem = row['problem']
        target = known_author == unknown_author
        
        print(f"        Working on problem {idx+1} of {total}: {problem}")

        # Filter the known and unknown for the current problem
        known_filtered = known[known['author'] == known_author]
        unknown_filtered = unknown[unknown['author'] == unknown_author]

        # Filter the reference dataset
        refs_filtered = refs[~refs['author'].isin([known_author, unknown_author])]

        known_sentences = known_filtered['tokens']
        unknown_sentences = unknown_filtered['tokens']

        num_known_sentences = len(known_sentences)
        num_unknown_sentences = len(unknown_sentences)

        if num_known_sentences > len(refs_filtered):
            raise ValueError(
                f"Not enough reference sentences ({len(refs_filtered)}) to sample {num_known_sentences}"
            )
        
        # turn the Series into a list first
        all_refs = refs_filtered['tokens'].tolist()

        known_counts = extract_ngrams(known_sentences, N)

        known_probs = []
        for q in unknown_sentences:
            ps = sentence_prob(q, known_counts, D=0.75, N=N)
            # replace zeros
            ps = [p if p > 0 else sys.float_info.min for p in ps]
            known_probs.append(ps)

        lambda_score = 0
        
        for _ in range(r):
            ref_sentences = random.sample(all_refs, num_known_sentences)

            ref_counts = extract_ngrams(ref_sentences, N)
            
            ref_probs = []
            for q in unknown_sentences:
                ps = sentence_prob(q, ref_counts, D=0.75, N=N)
                # replace zeros
                ps = [p if p > 0 else sys.float_info.min for p in ps]
                ref_probs.append(ps)

            lr_sum = 0.0
            for kp_sent, rp_sent in zip(known_probs, ref_probs):
                assert len(kp_sent) == len(rp_sent)
                for k, r_ in zip(kp_sent, rp_sent):
                    lr_sum += math.log10(k / r_)
        
            # 4. update lambda
            lambda_score += lr_sum / r

        results.append({
            'problem': problem,
            'known_author': known_author,
            'unknown_author': unknown_author,
            'target': target,
            'score': lambda_score
        })
        
    return pd.DataFrame(results)

def lambdaG_perplexity(unknown, known, refs, metadata=None, N=10, r=30, cores=1, vectorise=False):
    """
    Run the LambdaG author‐verification method.

    Parameters:
    - unknown   (pandas.DataFrame): DataFrame of questioned (disputed) documents.
    - known     (pandas.DataFrame): DataFrame of known (undisputed) documents.
    - refs      (pandas.DataFrame): Reference corpus for calibration. This can be the same as known.
    - metadata  (pandas.DataFrame, optional): A dataframe of problem metadata, used if known contains more than one author.
    - N         (int, optional): Order of the model (n-gram length). Default is 10.
    - r         (int, optional): Number of iterations/bootstrap samples. Default is 30.
    - cores     (int, optional): Number of CPU cores for parallel processing. Default is 1.
    - vectorise (bool, optional): If True, splits documents into sentences before feature extraction. Default is False.

    Returns:
    - pandas.DataFrame: Uncalibrated log-likelihood ratios (LambdaG) for each document in `unknown`.
    """

    problem_df = problem_data_prep(unknown, known, metadata)

    if vectorise:
        print("Vectorising data into sentences")
        # NOTE - Need to add in vetorising method
        
    total = len(problem_df)
    results = []
    for idx, row in problem_df.iterrows():
        known_author = row['known_author']
        unknown_author = row['unknown_author']
        problem = row['problem']
        target = known_author == unknown_author
        
        print(f"        Working on problem {idx+1} of {total}: {problem}")

        # Filter the known and unknown for the current problem
        known_filtered = known[known['author'] == known_author]
        unknown_filtered = unknown[unknown['author'] == unknown_author]

        # Filter the reference dataset
        refs_filtered = refs[~refs['author'].isin([known_author, unknown_author])]

        known_perplexities   = known_filtered['perplexity'].tolist()     # length K
        unknown_perplexities = unknown_filtered['perplexity'].tolist() # length U
        refs_perplexities    = refs_filtered['perplexity'].tolist()   

        # Dimensions
        K = len(known_perplexities)
        U = len(unknown_perplexities)

        if K > len(refs_perplexities):
            raise ValueError(f"Need at least {K} reference sentences, but only have {len(refs_perplexities)}")

        # ── 1) Initialize sentence scores ───────────────────────────────────────────
        sentence_scores = np.zeros(U, dtype=float)

        # ── 2) Loop over iterations, sampling one shared batch each time ─────────────
        for _ in range(r):
            # 2a) Sample K reference perplexities
            ref_batch = random.sample(refs_perplexities, K)
        
            # 2b) For each unknown, compute its array of ref-differences: |u - r|
            #     This gives an (U x K) matrix
            ref_diffs = np.abs(np.subtract.outer(unknown_perplexities, ref_batch))
        
            # 2c) Now for each unknown i:
            for i in range(U):
                # Grab that sentence’s ref-diff vector and sort it
                sorted_refs = np.sort(ref_diffs[i])
        
                # For each known j:
                for k_p in known_perplexities:
                    # Compute the genuine diff for this (u_i, k_p)
                    d_known = abs(unknown_perplexities[i] - k_p)
        
                    # Find rank among the ref_diffs (1-based)
                    rank = np.searchsorted(sorted_refs, d_known, side='left') + 1
        
                    # Accumulate normalized score
                    sentence_scores[i] += 1.0 / (rank * r * K)

        # ── 3) Each sentence_scores[i] ∈ [0,1] and will vary by sentence ───────────
        
        # ── 4) Document-level score = median of sentence scores ────────────────────
        doc_score = float(np.median(sentence_scores))
        print(f"\nDocument-level score (median of sentences): {doc_score:.4f}")

        results.append({
            'problem': problem,
            'known_author': known_author,
            'unknown_author': unknown_author,
            'target': target,
            'score': doc_score
        })
        
    return pd.DataFrame(results)