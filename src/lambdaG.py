import math
import nltk
import random
import sys

import numpy as np
import pandas as pd

from collections import Counter, defaultdict
from itertools import product, islice
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline, padded_everygrams

from read_and_write_docs import read_jsonl

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

def build_kn_model(sentences, N):
    """
    Build an N-gram language model with Kneser-Ney smoothing.

    Parameters:
    - sentences: List[List[str]] -- tokenized sentences
    - N: int -- order of the model (n-gram length)
    """
    train_data, vocab = padded_everygram_pipeline(N, sentences)
    model = KneserNeyInterpolated(order=N)
    model.fit(train_data, vocab)
    return model

def sentence_log10_prob(model, tokens, N):
    """
    Compute log10 probability of a token list under the given language model.

    Parameters:
    - model: KneserNeyInterpolated
    - tokens: List[str] -- one tokenized sentence
    - N: int
    """
    logp = 0.0
    for ngram in padded_everygrams(N, tokens):
        context, word = tuple(ngram[:-1]), ngram[-1]
        p = model.score(word, context)
        # Replace zeros with the smallest positive float
        if p <= 0:
            p = sys.float_info.min
        logp += math.log10(p)
    return logp

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

def extract_ngrams(sentences, N):
    """
    Return a dict mapping each n (1..N) to a Counter of n-grams,
    with (N−1) '<s>' pads and one '</s>' pad per sentence.
    """
    counts = {n: Counter() for n in range(1, N+1)}
    for sent in sentences:
        # pad once
        tokens = ['<s>']*(N-1) + sent + ['</s>']
        # for each n, produce all n-grams in one go via a generator
        for n in range(1, N+1):
            # zip(*(tokens[i:] for i in range(n))) is a very fast way to
            # slide an n-length window across tokens without slicing
            grams = zip(*(tokens[i:] for i in range(n)))
            counts[n].update(grams)
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

def lambdaG(unknown, known, refs=None, metadata=None, N=10, r=30, cores=1, vectorise=False):
    """
    Run the LambdaG author‐verification method.

    Parameters:
    - unknown   (pandas.DataFrame): DataFrame of questioned (disputed) documents.
    - known     (pandas.DataFrame): DataFrame of known (undisputed) documents.
    - refs      (pandas.DataFrame, optional): Reference corpus for calibration. This can be the same as known.
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

def lambdaG_v2(unknown, known, refs=None, metadata=None, N=10, r=30, cores=1, vectorise=False):
    """
    Run the LambdaG author‐verification method.

    Parameters:
    - unknown   (pandas.DataFrame): DataFrame of questioned (disputed) documents.
    - known     (pandas.DataFrame): DataFrame of known (undisputed) documents.
    - refs      (pandas.DataFrame, optional): Reference corpus for calibration. This can be the same as known.
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

        # Build the candidate model
        k_model = build_kn_model(known_sentences, N)
        # Precompute known vs unknown log-probs
        known_logps = [sentence_log10_prob(k_model, sent, N) for sent in unknown_sentences]

        lambda_score = 0
        
        for _ in range(r):
            ref_sentences = random.sample(all_refs, num_known_sentences)

            ref_model = build_kn_model(ref_sentences, N)
            ref_logps = [sentence_log10_prob(ref_model, sent, N) for sent in unknown_sentences]

            # Sum per-sentence log-LR
            lr = sum(k - r_ for k, r_ in zip(known_logps, ref_logps))
            lambda_score += lr / r

        results.append({
            'problem': problem,
            'known_author': known_author,
            'unknown_author': unknown_author,
            'target': target,
            'score': lambda_score
        })
        
    return pd.DataFrame(results)

def lambdaG_paraphrase(unknown, known, refs=None, metadata=None,
                       impostor_loc=None, N=10, r=30, cores=1, vectorise=False):
    """
    Run the LambdaG author‐verification method.

    Parameters:
    - unknown   (pandas.DataFrame): DataFrame of questioned (disputed) documents.
    - known     (pandas.DataFrame): DataFrame of known (undisputed) documents.
    - refs      (pandas.DataFrame, optional): Reference corpus for calibration. This can be the same as known.
    - metadata  (pandas.DataFrame, optional): A dataframe of problem metadata, used if known contains more than one author.
    - impostor_loc (str, optional): The directory where the impostors are stored for paraphrase version of function.

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
        known_docs = known_filtered['doc_id'].unique().tolist()

        # NOTE - NEED TO ACCOUNT FOR NOT ALL KNOWN DOCS BEING IN IMPOSTOR LOC
        # Filter the reference dataset
        if refs is not None and not refs.empty:
            refs_filtered = refs[refs['doc_id'].isin(known_docs)]
            print(f"Original refs count: {refs.shape[0]}")
        elif impostor_loc:
            ref_list = []
            for doc in known_docs:
                file_path = f"{impostor_loc}/{doc}.jsonl"
                try:
                    ref_df = read_jsonl(file_path)
                    ref_list.append(ref_df)
                except FileNotFoundError:
                    print(f"File not found, skipping: {file_path}")
            if ref_list:
                refs_filtered = pd.concat(ref_list, ignore_index=True)
            else:
                raise ValueError("No reference files were found in the impostor location.")
        else:
            raise ValueError("No `refs` provided and no `impostor_loc` set.")

        # Validate that all known_docs are in refs_filtered
        available_docs = refs_filtered['doc_id'].unique().tolist()
        missing_docs = [doc for doc in known_docs if doc not in available_docs]
        
        if missing_docs:
            print(f"{len(missing_docs)} known_docs are missing from refs_filtered and will be skipped: {missing_docs}")
            # Filter out missing docs from known
            known_docs = [doc for doc in known_docs if doc in available_docs]
            known_filtered = known_filtered[known_filtered['doc_id'].isin(known_docs)]

        known_sentences = known_filtered['tokens']
        unknown_sentences = unknown_filtered['tokens']

        num_known_sentences = len(known_sentences)
        num_unknown_sentences = len(unknown_sentences)

        if num_known_sentences > len(refs_filtered):
            raise ValueError(
                f"Not enough reference sentences ({len(refs_filtered)}) to sample {num_known_sentences}"
            )

        known_counts = extract_ngrams(known_sentences, N)

        known_probs = []
        for q in unknown_sentences:
            ps = sentence_prob(q, known_counts, D=0.75, N=N)
            # replace zeros
            ps = [p if p > 0 else sys.float_info.min for p in ps]
            known_probs.append(ps)

        # Generate a list of samples of length r
        all_impostors = refs_filtered['impostor_id'].unique().tolist()
        impostor_samples = random.sample(all_impostors, r)
        
        lambda_score = 0
        for impostor_id in impostor_samples:
            # Filter refs_filtered for the sampled impostor document
            impostor_refs = refs_filtered[refs_filtered['impostor_id'] == impostor_id]
        
            # Get all tokenized sentences from this impostor document
            ref_sentences = impostor_refs['tokens'].tolist()
        
            if not ref_sentences:
                print(f"Warning: No sentences found for impostor {impostor_id}, skipping.")
                continue
        
            ref_counts = extract_ngrams(ref_sentences, N)
        
            ref_probs = []
            for q in unknown_sentences:
                ps = sentence_prob(q, ref_counts, D=0.75, N=N)
                ps = [p if p > 0 else sys.float_info.min for p in ps]
                ref_probs.append(ps)
        
            lr_sum = 0.0
            for kp_sent, rp_sent in zip(known_probs, ref_probs):
                assert len(kp_sent) == len(rp_sent)
                for k, r_ in zip(kp_sent, rp_sent):
                    lr_sum += math.log10(k / r_)
        
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