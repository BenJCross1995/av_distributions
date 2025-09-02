from read_and_write_docs import read_jsonl
from utils import problem_data_prep

import random

import pandas as pd
import numpy as np

def impostors(unknown, known, refs=None, metadata=None, impostor_loc=None, 
              N=10, r=30, cores=1, vectorise=False, comparison_col='log_probs'):
    """Function to calculate the Impostors method"""
    
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

        # compute k_vs_u once per problem
        known_values  = [v for sent in known_filtered[comparison_col]  for v in sent]
        unknown_values= [v for sent in unknown_filtered[comparison_col] for v in sent]
        k_vs_u        = np.mean(known_values) - np.mean(unknown_values)
        
        # Generate a list of samples of length r
        all_impostors = refs_filtered['impostor_id'].unique().tolist()
 
        score = 0.0
        for i in range(N):
            # 1) sample r impostors
            sample_ids = random.sample(all_impostors, r)

            # 2) compute ref_vs_u for each
            ref_vs_u = []
            for imp_id in sample_ids:
                vals = [v for sent in
                        refs_filtered[refs_filtered['impostor_id']==imp_id][comparison_col]
                        for v in sent]
                ref_vs_u.append(np.mean(vals) - np.mean(unknown_values))

            # 3) rank k_vs_u among the r impostors:
            #    we build a combined list and see where k_vs_u falls (1 = highest)
            all_scores = ref_vs_u + [k_vs_u]
            # sort descending, then find the position of k_vs_u
            sorted_desc = sorted(all_scores, reverse=True)
            rank = sorted_desc.index(k_vs_u) + 1

            # 4) normalize by total (r impostors + the one true score)
            score += rank / len(all_scores)

        # optional: average over N
        avg_score = score / N

        results.append({
            'known_author':   known_author,
            'unknown_author': unknown_author,
            'true_match':     target,
            'impostor_score': avg_score
        })

    return pd.DataFrame(results)