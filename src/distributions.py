from utils import problem_data_prep
from scipy.stats import wasserstein_distance, ttest_ind

import pandas as pd

import numpy as np

def compute_emd(vec1, vec2, log_probs=False):
    """
    Computes Earth Mover's Distance (Wasserstein Distance) between two vectors.
    
    Parameters:
    - vec1, vec2: Input vectors (same length or will be truncated to min length)
    - log_probs: If True, inputs are treated as log probabilities and exponentiated
    
    Returns:
    - emd: The Earth Mover’s Distance between the two vectors
    """
    vec1, vec2 = np.array(vec1), np.array(vec2)
    
    # Truncate to the same length if needed
    #min_len = min(len(vec1), len(vec2))
    #vec1, vec2 = vec1[:min_len], vec2[:min_len]
    
    if log_probs:
        vec1 = np.exp(vec1)
        vec2 = np.exp(vec2)
    
    # Normalize to sum to 1 (optional, but often useful for probability comparisons)
    vec1 = vec1 / np.sum(vec1)
    vec2 = vec2 / np.sum(vec2)

    emd = wasserstein_distance(vec1, vec2)
    return emd

def distribution_av(unknown, known, metadata=None, r=30,
                    cores=1, vectorise=False, comparison_col='log_probs'):
    """Function to perform Author Verification by comparing two distributions"""
    
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
        
        known_sentences = known_filtered[comparison_col]
        unknown_sentences = unknown_filtered[comparison_col]
        
        num_known_sentences = len(known_sentences)
        num_unknown_sentences = len(unknown_sentences)
    
        if num_known_sentences == num_unknown_sentences:
            print(f"Completing 1 repetition")
            known_dist = [np.mean(sent) for sent in known_sentences]
            unknown_dist = [np.mean(sent) for sent in unknown_sentences]
            # t_stat, p_value = ttest_ind(known_dist, unknown_dist)
            score = compute_emd(known_dist, unknown_dist, log_probs=True)
        else:
            if num_known_sentences > num_unknown_sentences:
                bigger, smaller = known_sentences, unknown_sentences
                size = num_unknown_sentences
            else:
                bigger, smaller = unknown_sentences, known_sentences
                size = num_known_sentences

            scores = []
            for _ in range(r):
                # sample without replacement from the larger set
                sentence_sample = np.random.choice(bigger, size=size, replace=False)
                dist_1 = [np.mean(sent) for sent in sentence_sample]
                dist_2 = [np.mean(sent) for sent in smaller]
                # t_stat, p_value = ttest_ind(dist_1, dist_2)
                score_ = compute_emd(dist_1, dist_2, log_probs=True)
                scores.append(score_)

            score = float(np.mean(scores))
        
        results.append({
            'problem': problem,
            'known_author': known_author,
            'unknown_author': unknown_author,
            'target': target,
            'score': score
        })
        
    return pd.DataFrame(results)