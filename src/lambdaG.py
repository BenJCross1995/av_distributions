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
from utils import problem_data_prep

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
        
        if num_known_sentences > len(refs_filtered):
            raise ValueError(
                f"Not enough reference sentences ({len(refs_filtered)}) to sample {num_known_sentences}"
            )
        
        # turn the Series into a list first
        all_refs = refs_filtered['tokens'].tolist()

        lm_k = KneserNeyLanguageModel(10, 0.5)
        lm_k.fit_corpus(known_sentences)

        known_probs = []
        for q in unknown_sentences:
            ps = lm_k.probabilities(q)
            # replace zeros
            ps = [p if p > 0 else sys.float_info.min for p in ps]
            known_probs.append(ps)

        lambda_score = 0
        
        for _ in range(r):
            ref_sentences = random.sample(all_refs, num_known_sentences)

            lm_ref = KneserNeyLanguageModel(10, 0.5)
            lm_ref.fit_corpus(ref_sentences)
            
            ref_probs = []
            for q in unknown_sentences:
                ps = lm_ref.probabilities(q)
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

def lambdaG_no_ref(unknown, known, metadata=None, N=10, r=30, D=0.75, cores=1, vectorise=False):
    """
    Run the LambdaG author‐verification method without reference documents.

    Parameters:
    - unknown   (pandas.DataFrame): DataFrame of questioned (disputed) documents.
    - known     (pandas.DataFrame): DataFrame of known (undisputed) documents.
    - metadata  (pandas.DataFrame, optional): A dataframe of problem metadata, used if known contains more than one author.
    - N         (int, optional): Order of the model (n-gram length). Default is 10.
    - r         (int, optional): Number of iterations/bootstrap samples. Default is 30.
    - D         (float, optional): Discount parameter for the language model. Default is 0.75.
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

        known_sentences = known_filtered['tokens']
        unknown_sentences = unknown_filtered['tokens']
        unknown_log_probs = unknown_filtered['log_probs']
        
        lm_k = KneserNeyLanguageModel(N, D)
        lm_k.fit_corpus(known_sentences)

        known_probs = []
        for q in unknown_sentences:
            ps = lm_k.probabilities(q)
            # replace zeros
            ps = [p if p > 0 else sys.float_info.min for p in ps]
            known_probs.append(ps)
            
        # --- FIX: flatten list of lists, then take logs ---
        flat_known_log_probs = [math.log2(p) for ps in known_probs for p in ps]
        # If your unknown_log_probs is itself a list-of-lists, flatten it too:
        if any(isinstance(x, (list, np.ndarray)) for x in unknown_log_probs):
            flat_unknown_log_probs = [lp for ups in unknown_log_probs for lp in ups]
        else:
            flat_unknown_log_probs = list(unknown_log_probs)

        known_perplexity = 2**(-np.mean(flat_known_log_probs))
        unknown_perplexity = 2**(-np.mean(flat_unknown_log_probs))
        
        lambda_score = known_perplexity / unknown_perplexity
        
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
    
