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
from kneser_ney import KneserNeyLanguageModel

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

def cosine_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray
) -> float:
    """
    Compute the cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity score between -1 and 1.
    """
    # Convert inputs to numpy arrays
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)

    # Compute dot product and norms
    dot_prod = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("One or both vectors have zero magnitude, cosine similarity is undefined.")

    return dot_prod / (norm1 * norm2)

def get_top_n_closest(
    df: pd.DataFrame,
    query_vec: np.ndarray,
    embedding_column: str = 'embedding',
    top_n: int = 5
) -> pd.DataFrame:
    """
    Retrieve the top N rows from a DataFrame whose embeddings are closest to a query vector.

    Args:
        df (pd.DataFrame): DataFrame containing an embedding column.
        query_vec (np.ndarray or list): The query embedding vector.
        embedding_column (str): Name of the column with embeddings.
        top_n (int): Number of top similar rows to return.

    Returns:
        pd.DataFrame: Subset of the original DataFrame sorted by descending similarity,
                      with an additional 'similarity' column.
    """
    # Validate input
    if embedding_column not in df.columns:
        raise ValueError(f"Embedding column '{embedding_column}' not found in DataFrame.")

    # Convert query vector to numpy array
    qv = np.array(query_vec, dtype=float)
    if np.linalg.norm(qv) == 0:
        raise ValueError("Query vector has zero magnitude, cosine similarity is undefined.")

    # Compute similarities
    sims = []
    for emb in df[embedding_column]:
        sims.append(cosine_similarity(qv, np.array(emb, dtype=float)))

    # Create a copy with similarity scores
    df_with_sim = df.copy()
    df_with_sim['similarity'] = sims

    # Sort and select top N
    top_df = df_with_sim.sort_values(by='similarity', ascending=False).head(top_n)

    return top_df.reset_index(drop=True)
    
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

def lambdaG_max_similarity(unknown, known, refs=None, metadata=None, N=10, r=30, cores=1, vectorise=False, embed=False):
    """
    Run the LambdaG author‐verification method, instead of random sampling of sentences take the most similar.

    Parameters:
    - unknown   (pandas.DataFrame): DataFrame of questioned (disputed) documents.
    - known     (pandas.DataFrame): DataFrame of known (undisputed) documents.
    - refs      (pandas.DataFrame, optional): Reference corpus for calibration. This can be the same as known.
    - metadata  (pandas.DataFrame, optional): A dataframe of problem metadata, used if known contains more than one author.
    - N         (int, optional): Order of the model (n-gram length). Default is 10.
    - r         (int, optional): Number of iterations/bootstrap samples. Default is 30.
    - cores     (int, optional): Number of CPU cores for parallel processing. Default is 1.
    - vectorise (bool, optional): If True, splits documents into sentences before feature extraction. Default is False.
    - embed     (bool, optional): If True, embeds the dataframes prior to completing. Default is False.

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

        known_counts = extract_ngrams(known_sentences, N)

        known_probs = []
        for q in unknown_sentences:
            ps = sentence_prob(q, known_counts, D=0.75, N=N)
            # replace zeros
            ps = [p if p > 0 else sys.float_info.min for p in ps]
            known_probs.append(ps)

        # ----------
        # NOTE - New element of getting max similarity sentence for each in the known
        # Build neighbor lists: list of (text, tokens) per known sentence
        refs_emb_df = refs_filtered[['text', 'tokens', 'embedding']].reset_index(drop=True)
        neighbor_lists = []
        for emb in known_filtered['embedding']:
            top_df = get_top_n_closest(refs_emb_df, emb, embedding_column='embedding', top_n=r)
            # store pairs of (text, tokens)
            neighbor_lists.append(list(zip(top_df['text'], top_df['tokens'])))

        # Generate r bootstrap samples ensuring no duplicate texts within each sample
        max_attempts = 100
        sample_sets = []
        for sample_idx in range(r):
            success = False
            for attempt in range(max_attempts):
                sample_tokens = []
                chosen_texts = set()
                # Shuffle neighbor lists to randomize selection
                shuffled_lists = [random.sample(nbrs, len(nbrs)) for nbrs in neighbor_lists]
                for nbrs in shuffled_lists:
                    # pick first (text, tokens) whose text not yet chosen
                    for text, tokens in nbrs:
                        if text not in chosen_texts:
                            sample_tokens.append(tokens)
                            chosen_texts.add(text)
                            break
                    else:
                        # this shuffled attempt failed, break to retry
                        break
                if len(sample_tokens) == len(neighbor_lists):
                    sample_sets.append(sample_tokens)
                    success = True
                    break
            if not success:
                raise ValueError(f"Could not generate sample {sample_idx+1} without duplicate texts after {max_attempts} attempts")
        # ----------

        # Compute LambdaG using these sample sets
        lambda_score = 0.0
        for current_refs in sample_sets:
            ref_counts = extract_ngrams(current_refs, N)
            ref_probs = []
            for q in unknown_sentences:
                ps = sentence_prob(q, ref_counts, D=0.75, N=N)
                ps = [p if p > 0 else sys.float_info.min for p in ps]
                ref_probs.append(ps)

            lr_sum = 0.0
            for kp, rp in zip(known_probs, ref_probs):
                for k_val, r_val in zip(kp, rp):
                    lr_sum += math.log10(k_val / r_val)
            lambda_score += lr_sum / r

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

        # known_counts = extract_ngrams(known_sentences, N)
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

def lambdaG_paraphrase_v2(unknown, known, refs=None, metadata=None,
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

        # known_counts = extract_ngrams(known_sentences, N)
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

        all_refs = refs_filtered['tokens'].tolist()

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

def lambdaG_jsd(unknown, known, refs=None, metadata=None, impostor_loc=None, 
               N=10, r=30, cores=1, vectorise=False):
    """
    LambdaG variant using Jensen-Shannon Divergence.
    
    For each problem (known vs unknown pair), computes the Jensen-Shannon divergence 
    between the unknown text's token distribution and the known author's distribution, 
    and between the unknown and each reference impostor distribution. The final score 
    is the average difference JSD(unknown, ref) - JSD(unknown, known).
    """
    problem_df = problem_data_prep(unknown, known, metadata)
    if vectorise:
        print("Vectorising data into sentences")
        # (Vectorization not implemented)
    total = len(problem_df)
    results = []
    for idx, row in problem_df.iterrows():
        known_author = row['known_author']; unknown_author = row['unknown_author']
        problem = row['problem']; target = (known_author == unknown_author)
        print(f"        Working on problem {idx+1} of {total}: {problem}")
        # Filter data for this author pair
        known_filtered = known[known['author'] == known_author]
        unknown_filtered = unknown[unknown['author'] == unknown_author]
        known_docs = known_filtered['doc_id'].unique().tolist()
        # Prepare reference impostor set (paraphrased docs for known author)
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
        # Ensure all known_docs have references
        available_docs = refs_filtered['doc_id'].unique().tolist()
        missing_docs = [doc for doc in known_docs if doc not in available_docs]
        if missing_docs:
            print(f"{len(missing_docs)} known_docs missing in refs; skipping: {missing_docs}")
            known_docs = [doc for doc in known_docs if doc in available_docs]
            known_filtered = known_filtered[known_filtered['doc_id'].isin(known_docs)]
        # Collect tokens and build distributions
        known_sentences = known_filtered['tokens']; unknown_sentences = unknown_filtered['tokens']
        # Flatten all tokens from known and unknown (list of tokens for each doc)
        known_tokens_all = [tok for tokens in known_sentences for tok in tokens]
        unknown_tokens_all = [tok for tokens in unknown_sentences for tok in tokens]
        # Probability distributions (unigram) for known and unknown
        known_count = Counter(known_tokens_all)
        unknown_count = Counter(unknown_tokens_all)
        total_known = sum(known_count.values()); total_unknown = sum(unknown_count.values())
        P_known = {w: cnt/total_known for w, cnt in known_count.items()}
        P_unknown = {w: cnt/total_unknown for w, cnt in unknown_count.items()}
        # Pre-compute JSD between unknown and known distributions (constant for this problem)
        def jensen_shannon(P, Q):
            """Helper to compute Jensen-Shannon divergence between two distributions."""
            # Merge keys of P and Q
            vocab = set(P.keys()) | set(Q.keys())
            # Compute midpoint M = 0.5*(P+Q)
            M = {}
            for w in vocab:
                p = P.get(w, 0.0); q = Q.get(w, 0.0)
                M[w] = 0.5 * (p + q)
            # KLD(P||M) and KLD(Q||M) in base-2 (skip terms where P or Q prob is 0)
            KLD_PM = 0.0
            for w in vocab:
                if w in P and P[w] > 0:
                    KLD_PM += P[w] * math.log2(P[w] / M[w])
            KLD_QM = 0.0
            for w in vocab:
                if w in Q and Q[w] > 0:
                    KLD_QM += Q[w] * math.log2(Q[w] / M[w])
            return 0.5 * (KLD_PM + KLD_QM)
        jsd_known = jensen_shannon(P_unknown, P_known)  # JSD(unknown||known)
        # Sample r reference impostors and compute average difference
        all_impostors = refs_filtered['impostor_id'].unique().tolist()
        if r > len(all_impostors):
            raise ValueError(f"Need at least {r} impostors, but only have {len(all_impostors)}")
        impostor_samples = random.sample(all_impostors, r)
        score_sum = 0.0
        for impostor_id in impostor_samples:
            impostor_refs = refs_filtered[refs_filtered['impostor_id'] == impostor_id]
            ref_sentences = impostor_refs['tokens'].tolist()
            if not ref_sentences:
                print(f"Warning: no sentences for impostor {impostor_id}, skipping.")
                continue
            # Flatten tokens for this reference impostor document and compute distribution
            ref_tokens_all = [tok for tokens in ref_sentences for tok in tokens]
            ref_count = Counter(ref_tokens_all)
            total_ref = sum(ref_count.values())
            P_ref = {w: cnt/total_ref for w, cnt in ref_count.items()}
            jsd_ref = jensen_shannon(P_unknown, P_ref)    # JSD(unknown||ref)
            score_diff = jsd_ref - jsd_known
            score_sum += score_diff
        lambda_score = score_sum / r
        results.append({
            'problem': problem, 'known_author': known_author, 
            'unknown_author': unknown_author, 'target': target, 'score': lambda_score
        })
    return pd.DataFrame(results)

def lambdaG_renyi(unknown, known, refs=None, metadata=None, impostor_loc=None, 
                 N=10, r=30, alpha=0.5, cores=1, vectorise=False):
    """
    LambdaG variant using Rényi divergence of order `alpha`.
    
    Computes D_alpha(Q_unknown || P_known) and D_alpha(Q_unknown || P_ref) for each 
    reference impostor, and returns the average difference (ref - known). A higher 
    score suggests the unknown's distribution is closer to known's than to references.
    """
    problem_df = problem_data_prep(unknown, known, metadata)
    if vectorise:
        print("Vectorising data into sentences")
    results = []
    total = len(problem_df)
    for idx, row in problem_df.iterrows():
        known_author = row['known_author']; unknown_author = row['unknown_author']
        problem = row['problem']; target = (known_author == unknown_author)
        print(f"        Working on problem {idx+1} of {total}: {problem}")
        # Filter data
        known_filtered = known[known['author'] == known_author]
        unknown_filtered = unknown[unknown['author'] == unknown_author]
        known_docs = known_filtered['doc_id'].unique().tolist()
        # Prepare references similarly as above
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
                raise ValueError("No reference files found in impostor location.")
        else:
            raise ValueError("No `refs` provided and no `impostor_loc` set.")
        # Handle missing reference docs
        available_docs = refs_filtered['doc_id'].unique().tolist()
        known_docs = [doc for doc in known_docs if doc in available_docs]
        known_filtered = known_filtered[known_filtered['doc_id'].isin(known_docs)]
        # Flatten tokens and get distributions
        known_tokens = [tok for tokens in known_filtered['tokens'] for tok in tokens]
        unknown_tokens = [tok for tokens in unknown_filtered['tokens'] for tok in tokens]
        P_known_count = Counter(known_tokens); Q_unknown_count = Counter(unknown_tokens)
        total_known = sum(P_known_count.values()); total_unknown = sum(Q_unknown_count.values())
        P_known = {w: cnt/total_known for w, cnt in P_known_count.items()}
        Q_unknown = {w: cnt/total_unknown for w, cnt in Q_unknown_count.items()}
        # Helper to compute Renyi divergence D_alpha(Q||P)
        def renyi_divergence(Q, P, alpha):
            # If alpha == 1, we could return KL divergence, but we assume alpha != 1 here.
            if alpha <= 0:
                raise ValueError("Alpha must be > 0 and != 1.")
            # Sum_{i} Q_i^alpha * P_i^(1-alpha)
            sum_val = 0.0
            for w, q in Q.items():
                if q <= 0: 
                    continue  # no contribution if Q prob is 0
                p = P.get(w, 0.0)
                if p == 0:
                    # Avoid infinite divergence: use a tiny probability for P
                    p = sys.float_info.min
                sum_val += (q ** alpha) * (p ** (1 - alpha))
            # Compute divergence = 1/(alpha-1) * log2(sum_val)
            return (1.0 / (alpha - 1)) * math.log2(sum_val)
        # Compute D_alpha for unknown vs known once
        D_known = renyi_divergence(Q_unknown, P_known, alpha)
        # Iterate references
        all_impostors = refs_filtered['impostor_id'].unique().tolist()
        if r > len(all_impostors):
            raise ValueError(f"Need at least {r} impostors, but have {len(all_impostors)}")
        impostor_samples = random.sample(all_impostors, r)
        diff_sum = 0.0
        for impostor_id in impostor_samples:
            impostor_refs = refs_filtered[refs_filtered['impostor_id'] == impostor_id]
            ref_tokens = [tok for tokens in impostor_refs['tokens'].tolist() for tok in tokens]
            P_ref_count = Counter(ref_tokens)
            total_ref = sum(P_ref_count.values())
            P_ref = {w: cnt/total_ref for w, cnt in P_ref_count.items()}
            # D_alpha for unknown vs this reference
            D_ref = renyi_divergence(Q_unknown, P_ref, alpha)
            diff_sum += (D_ref - D_known)
        lambda_score = diff_sum / r
        results.append({
            'problem': problem, 'known_author': known_author,
            'unknown_author': unknown_author, 'target': target, 'score': lambda_score
        })
    return pd.DataFrame(results)

def lambdaG_entropy_weighted(unknown, known, refs=None, metadata=None, impostor_loc=None,
                             N=10, r=30, cores=1, vectorise=False):
    """
    LambdaG variant with entropy-weighted log-likelihood ratio.
    
    Computes the standard LambdaG score (log-likelihood ratio) for each reference 
    sample, then scales it by (H_ref / H_known), where H_known and H_ref are the 
    Shannon entropies of the known author's and reference's token distributions.
    """
    problem_df = problem_data_prep(unknown, known, metadata)
    if vectorise:
        print("Vectorising data into sentences")
    results = []
    total = len(problem_df)
    for idx, row in problem_df.iterrows():
        known_author = row['known_author']; unknown_author = row['unknown_author']
        problem = row['problem']; target = (known_author == unknown_author)
        print(f"        Working on problem {idx+1} of {total}: {problem}")
        # Filter data and references as usual
        known_filtered = known[known['author'] == known_author]
        unknown_filtered = unknown[unknown['author'] == unknown_author]
        known_docs = known_filtered['doc_id'].unique().tolist()
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
        available_docs = refs_filtered['doc_id'].unique().tolist()
        known_docs = [doc for doc in known_docs if doc in available_docs]
        known_filtered = known_filtered[known_filtered['doc_id'].isin(known_docs)]
        # Prepare n-gram counts for known once
        known_counts = extract_ngrams(known_filtered['tokens'], N)
        # Compute Shannon entropy of known author's unigram distribution
        # (We use n=1 n-grams counts for entropy)
        total_known_tokens = sum(known_counts[1].values())
        H_known = 0.0
        for ng, cnt in known_counts[1].items():  # ng is a tuple like ('word',)
            p = cnt / total_known_tokens
            H_known -= p * math.log2(p) if p > 0 else 0
        # Precompute unknown text probabilities under known model (as in LambdaG)
        unknown_sentences = unknown_filtered['tokens']
        known_probs = []
        for sent in unknown_sentences:
            probs = sentence_prob(sent, known_counts, D=0.75, N=N)
            probs = [p if p > 0 else sys.float_info.min for p in probs]
            known_probs.append(probs)
        # Randomly sample r reference impostors
        all_impostors = refs_filtered['impostor_id'].unique().tolist()
        if r > len(all_impostors):
            raise ValueError(f"Need at least {r} impostors, but have {len(all_impostors)}")
        impostor_samples = random.sample(all_impostors, r)
        score_sum = 0.0
        for impostor_id in impostor_samples:
            impostor_refs = refs_filtered[refs_filtered['impostor_id'] == impostor_id]
            ref_sentences = impostor_refs['tokens'].tolist()
            if not ref_sentences:
                print(f"Warning: No sentences for impostor {impostor_id}, skipping.")
                continue
            # Compute counts and entropy for reference distribution
            ref_counts = extract_ngrams(ref_sentences, N)
            total_ref_tokens = sum(ref_counts[1].values())
            H_ref = 0.0
            for ng, cnt in ref_counts[1].items():
                p = cnt / total_ref_tokens
                H_ref -= p * math.log2(p) if p > 0 else 0
            # Compute unknown text probabilities under reference model
            ref_probs = []
            for sent in unknown_sentences:
                probs = sentence_prob(sent, ref_counts, D=0.75, N=N)
                probs = [p if p > 0 else sys.float_info.min for p in probs]
                ref_probs.append(probs)
            # Calculate log-likelihood ratio sum for this reference (base-10 as in LambdaG)
            lr_sum = 0.0
            for kp, rp in zip(known_probs, ref_probs):
                for pk, pr in zip(kp, rp):
                    lr_sum += math.log10(pk / pr)
            # Apply entropy weight factor = H_ref / H_known (avoid zero by adding a tiny value)
            weight = H_ref / (H_known + 1e-12)
            score_sum += (lr_sum * weight)
        lambda_score = score_sum / r
        results.append({
            'problem': problem, 'known_author': known_author,
            'unknown_author': unknown_author, 'target': target, 'score': lambda_score
        })
    return pd.DataFrame(results)

def lambdaG_surprisal(unknown, known, refs=None, metadata=None, impostor_loc=None, 
                     N=10, r=30, cores=1, vectorise=False):
    """
    LambdaG variant using surprisal-based token-wise comparison.
    
    For each token in the unknown text, check if the known author's model or the 
    reference model predicts that token with higher probability (lower surprisal). 
    The score is the fraction of tokens better predicted by the known author, averaged 
    over r random impostor references.
    """
    problem_df = problem_data_prep(unknown, known, metadata)
    if vectorise:
        print("Vectorising data into sentences")
    results = []
    total = len(problem_df)
    for idx, row in problem_df.iterrows():
        known_author = row['known_author']; unknown_author = row['unknown_author']
        problem = row['problem']; target = (known_author == unknown_author)
        print(f"        Working on problem {idx+1} of {total}: {problem}")
        # Filter data and references
        known_filtered = known[known['author'] == known_author]
        unknown_filtered = unknown[unknown['author'] == unknown_author]
        known_docs = known_filtered['doc_id'].unique().tolist()
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
        available_docs = refs_filtered['doc_id'].unique().tolist()
        known_docs = [doc for doc in known_docs if doc in available_docs]
        known_filtered = known_filtered[known_filtered['doc_id'].isin(known_docs)]
        # Build n-gram counts for known author (for probability model)
        known_counts = extract_ngrams(known_filtered['tokens'], N)
        unknown_sentences = unknown_filtered['tokens']
        # Precompute known model probabilities for each token in the unknown text
        known_probs = []
        for sent in unknown_sentences:
            probs = sentence_prob(sent, known_counts, D=0.75, N=N)
            probs = [p if p > 0 else sys.float_info.min for p in probs]
            known_probs.append(probs)
        # Randomly sample reference impostors
        all_impostors = refs_filtered['impostor_id'].unique().tolist()
        if r > len(all_impostors):
            raise ValueError(f"Need at least {r} impostors, but have {len(all_impostors)}")
        impostor_samples = random.sample(all_impostors, r)
        # Count total tokens and how many times known model is more likely
        total_tokens = 0
        total_known_better = 0
        for impostor_id in impostor_samples:
            impostor_refs = refs_filtered[refs_filtered['impostor_id'] == impostor_id]
            ref_sentences = impostor_refs['tokens'].tolist()
            if not ref_sentences:
                print(f"Warning: No sentences for impostor {impostor_id}, skipping.")
                continue
            ref_counts = extract_ngrams(ref_sentences, N)
            # Compute probabilities for unknown text under this reference model
            ref_probs = []
            for sent in unknown_sentences:
                probs = sentence_prob(sent, ref_counts, D=0.75, N=N)
                probs = [p if p > 0 else sys.float_info.min for p in probs]
                ref_probs.append(probs)
            # Compare token by token
            for kp_sent, rp_sent in zip(known_probs, ref_probs):
                assert len(kp_sent) == len(rp_sent)
                for pk, pr in zip(kp_sent, rp_sent):
                    total_tokens += 1
                    if pk > pr:
                        total_known_better += 1
        if total_tokens == 0:
            score = 0.0
        else:
            score = total_known_better / total_tokens
        results.append({
            'problem': problem, 'known_author': known_author,
            'unknown_author': unknown_author, 'target': target, 'score': score
        })
    return pd.DataFrame(results)
    
def lambdaG_hellinger(unknown, known, refs=None, metadata=None, impostor_loc=None,
                     N=10, r=30, cores=1, vectorise=False):
    """
    LambdaG variant using Hellinger distance.
    
    Computes Hellinger distance between the unknown text's token distribution and 
    the known author's, and between unknown and reference distributions. The score 
    is the average (HD_unknown_ref - HD_unknown_known) over r samples.
    """
    problem_df = problem_data_prep(unknown, known, metadata)
    if vectorise:
        print("Vectorising data into sentences")
    results = []
    total = len(problem_df)
    for idx, row in problem_df.iterrows():
        known_author = row['known_author']; unknown_author = row['unknown_author']
        problem = row['problem']; target = (known_author == unknown_author)
        print(f"        Working on problem {idx+1} of {total}: {problem}")
        # Filter and reference handling similar to above
        known_filtered = known[known['author'] == known_author]
        unknown_filtered = unknown[unknown['author'] == unknown_author]
        known_docs = known_filtered['doc_id'].unique().tolist()
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
                raise ValueError("No reference files found in impostor location.")
        else:
            raise ValueError("No `refs` provided and no `impostor_loc` set.")
        available_docs = refs_filtered['doc_id'].unique().tolist()
        known_docs = [doc for doc in known_docs if doc in available_docs]
        known_filtered = known_filtered[known_filtered['doc_id'].isin(known_docs)]
        # Build distributions (unigram) for known and unknown
        known_tokens = [tok for tokens in known_filtered['tokens'] for tok in tokens]
        unknown_tokens = [tok for tokens in unknown_filtered['tokens'] for tok in tokens]
        P_known_count = Counter(known_tokens); Q_unknown_count = Counter(unknown_tokens)
        total_known = sum(P_known_count.values()); total_unknown = sum(Q_unknown_count.values())
        P_known = {w: cnt/total_known for w, cnt in P_known_count.items()}
        Q_unknown = {w: cnt/total_unknown for w, cnt in Q_unknown_count.items()}
        # Helper to compute Hellinger distance between distributions
        def hellinger_distance(P, Q):
            vocab = set(P.keys()) | set(Q.keys())
            sum_sq = 0.0
            for w in vocab:
                p = P.get(w, 0.0); q = Q.get(w, 0.0)
                # accumulate (sqrt(p) - sqrt(q))^2
                sum_sq += (math.sqrt(p) - math.sqrt(q))**2
            # 1/sqrt(2) * sqrt(sum_sq)
            return math.sqrt(sum_sq) / math.sqrt(2)
        # Precompute Hellinger for unknown vs known
        hd_known = hellinger_distance(Q_unknown, P_known)
        # Sample references and accumulate differences
        all_impostors = refs_filtered['impostor_id'].unique().tolist()
        if r > len(all_impostors):
            raise ValueError(f"Need at least {r} impostors, but have {len(all_impostors)}")
        impostor_samples = random.sample(all_impostors, r)
        diff_sum = 0.0
        for impostor_id in impostor_samples:
            impostor_refs = refs_filtered[refs_filtered['impostor_id'] == impostor_id]
            ref_tokens = [tok for tokens in impostor_refs['tokens'].tolist() for tok in tokens]
            P_ref_count = Counter(ref_tokens)
            total_ref = sum(P_ref_count.values())
            P_ref = {w: cnt/total_ref for w, cnt in P_ref_count.items()}
            hd_ref = hellinger_distance(Q_unknown, P_ref)
            diff_sum += (hd_ref - hd_known)
        lambda_score = diff_sum / r
        results.append({
            'problem': problem, 'known_author': known_author,
            'unknown_author': unknown_author, 'target': target, 'score': lambda_score
        })
    return pd.DataFrame(results)
