import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_loc: str):
    """Load a local AutoModelForCausalLM and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_loc)
    model = AutoModelForCausalLM.from_pretrained(model_loc)
    model.eval()
    return tokenizer, model
    
def compute_log_probs_with_median(text: str, tokenizer, model):
    """
    For each token (excluding first), return:
    - tokens: list of tokens in the text
    - log_probs: list of chosen-token log-probs
    - median_logprobs: list of median log-probs for each token
    """
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    tokens = tokenizer.decode(input_ids[0]).split()  # Convert input_ids to tokens

    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs.logits  # [batch_size=1, seq_len, vocab_size]
    
    log_probs = []
    median_logprobs = []
    # We start from the second token, as the first one has no previous token to condition on
    for i in range(0, input_ids.size(1)):
        if i == 0:
            logits_prev = logits[0, 0]
        else:
            logits_prev = logits[0, i - 1]
        dist = torch.log_softmax(logits_prev, dim=-1)
        
        # Extract the log probabilities
        log_prob = dist[input_ids[0, i].item()].item()
        median_logprob = float(dist.median().item())
        
        # Append to lists
        log_probs.append(log_prob)
        median_logprobs.append(median_logprob)
    
    # The tokens list starts from the first token, but the log_probs and median_logprobs start from the second
    # To align them, we need to slice the tokens list to match the lengths
    tokens = tokens[0:]  # Match the length of log_probs and median_logprobs
    
    return tokens, log_probs, median_logprobs
    
def compute_perplexity(logprobs):
    """Sentence perplexity from natural‐log token log-probs."""
    return float(np.exp(-np.mean(logprobs)))

def score_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    model_loc: str  = "path/to/Qwen_2.5_1.5B"
) -> pd.DataFrame:
    """
    Adds token-level and aggregate scoring columns to `df[text_column]`.
    """
    # load once
    tokenizer, model = load_model(model_loc)

    # ensure tqdm on pandas
    tqdm.pandas(desc="Scoring texts")

    df = df.copy()
    # run our per‐text scorer
    def _score_row(txt):
        toks, lps, meds = compute_log_probs_with_median(txt, tokenizer, model)
        return pd.Series((toks, lps, meds))
    df[['tokens', 'log_probs', 'med_log_prob']] = df[text_column].progress_apply(_score_row)

    # differences
    df['differences']     = df.apply(lambda r: [lp - m for lp, m in zip(r['log_probs'], r['med_log_prob'])], axis=1)
    df['abs_differences'] = df['differences'].apply(lambda lst: [abs(x) for x in lst])

    # aggregates
    df['num_tokens']   = df['log_probs'].apply(len)
    df['sum_log_prob'] = df['log_probs'].apply(sum)
    df['avg_log_prob'] = df['sum_log_prob'] / df['num_tokens']
    df['mean_diff']    = df['differences'].apply(np.mean)
    df['mean_abs_diff']= df['abs_differences'].apply(np.mean)
    df['perplexity']   = df['log_probs'].apply(compute_perplexity)

    return df