"""
paraphrase_gap_qwen.py

Gap-only paraphrase generation and scoring using a Qwen causal LLM with Fill-In-the-Middle (FIM).
- Generates paraphrase candidates for a missing span between a given prefix and suffix.
- Scores each candidate by conditional log-likelihood, summing ONLY over the span tokens.
- Keeps generating until "probability mass" (normalized to the current best candidate) saturates.

USAGE (example):
    python paraphrase_gap_qwen.py

Or import the functions:
    from paraphrase_gap_qwen import (
        load_qwen, generate_gap_candidates_until_saturation
    )

Requirements:
    pip install transformers accelerate torch sentencepiece
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math
import re
import random
import sys
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    NoBadWordsLogitsProcessor,
    set_seed,
)

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"


@dataclass
class GenerationConfig:
    # Decoding settings
    num_beams: int = 10
    num_return_sequences: int = 10
    num_beam_groups: int = 5
    diversity_penalty: float = 0.7
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    do_sample: bool = False  # False => beam, True => sampling
    max_new_tokens: int = 16
    no_repeat_ngram_size: int = 3

    # Constraints
    ban_original_first_token: bool = True
    min_len: int = 0  # token-level; optional enforcement can be manual post-filter
    max_len: int = 30

    # Randomness
    seed: Optional[int] = 42


@dataclass
class SaturationConfig:
    # Stopping rule based on mass growth relative to the best candidate (Z grows towards a stable value).
    epsilon: float = 0.02         # stop when fractional increase < epsilon
    patience_rounds: int = 2       # need this many consecutive rounds under epsilon
    max_rounds: int = 10           # hard cap on generation rounds
    batch_candidates: int = 30     # aim to add up to this many NEW candidates per round
    max_total_candidates: int = 200


@dataclass
class Candidate:
    text: str
    total_logprob: float
    token_count: int


def load_qwen(model_name: str = "Qwen/Qwen1.5-7B", device: Optional[str] = None):
    """
    Load Qwen model + tokenizer.
    """
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    if device:
        model.to(device)
    model.eval()
    return tok, model


def fim_format(prefix: str, suffix: str) -> str:
    """
    Build the Fill-In-the-Middle prompt for Qwen.
    """
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"


def _clean_text(x: str) -> str:
    # Basic normalization: strip spaces and awkward leading punctuation.
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
    return x


def _unique_keep_order(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _first_token_id(tokenizer, text: str) -> Optional[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 0:
        return None
    return ids[0]


def _decode_gap_only(tokenizer, input_ids: torch.LongTensor, full_output_ids: torch.LongTensor) -> str:
    """
    Given full generation that starts with the FIM prompt, extract only the generated gap text.
    """
    gen_ids = full_output_ids[0]
    prompt_len = input_ids.shape[-1]
    gap_ids = gen_ids[prompt_len:]
    return tokenizer.decode(gap_ids, skip_special_tokens=True).strip()


def _score_span_logprob(
    tokenizer, model, prefix: str, span: str, suffix: str
) -> Tuple[float, int]:
    """
    Compute total log-probability ONLY for the span tokens in: prefix + span + suffix.
    Returns (total_logprob, token_count_for_span).
    """
    # Tokenize pieces separately to get boundaries
    ids_prefix = tokenizer.encode(prefix, add_special_tokens=False)
    ids_span = tokenizer.encode(span, add_special_tokens=False)
    ids_suffix = tokenizer.encode(suffix, add_special_tokens=False)

    # Build combined
    all_ids = ids_prefix + ids_span + ids_suffix
    if len(all_ids) == 0:
        return float("-inf"), 0

    input_ids = torch.tensor([all_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss is average NLL over ALL tokens except first; we want token-level to sum just span
        # We'll recompute per-token NLL via logits and shift
        logits = outputs.logits  # [1, T, V]
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]  # next token IDs
        log_probs = torch.log_softmax(shift_logits, dim=-1)  # [1, T-1, V]

        # Indices: tokens 0..len(all_ids)-2 in shift; each predicts token t+1
        # span occupies positions [len(ids_prefix), len(ids_prefix)+len(ids_span)-1] in the *original* all_ids
        # The predicted indices for the span are those indices t where t+1 is inside the span range.
        start = len(ids_prefix)
        end = len(ids_prefix) + len(ids_span) - 1  # inclusive in original space
        # In shift space (predicting token t+1), we sum for t indices from (start-1) to (end-1)
        t_start = max(start - 1, 0)
        t_end = max(end - 1, -1)

        total_logprob = 0.0
        count = 0
        for t in range(t_start, t_end + 1):
            gold_id = int(shift_labels[0, t].item())
            lp = float(log_probs[0, t, gold_id].item())
            total_logprob += lp
            count += 1

    return total_logprob, count


def _build_logits_processors(
    tokenizer, original_span: Optional[str], cfg: GenerationConfig
) -> LogitsProcessorList:
    processors = LogitsProcessorList()
    # Optionally ban the first token of the original span to discourage verbatim copies.
    if cfg.ban_original_first_token and original_span:
        first_id = _first_token_id(tokenizer, original_span)
        if first_id is not None:
            processors.append(NoBadWordsLogitsProcessor(bad_words_ids=[[first_id]], eos_token_id=tokenizer.eos_token_id))
    return processors


def generate_one_round(
    tokenizer,
    model,
    prefix: str,
    suffix: str,
    original_span: Optional[str],
    gen_cfg: GenerationConfig,
) -> List[str]:
    """
    Run one round of generation and return deduped candidate gap strings.
    """
    if gen_cfg.seed is not None:
        set_seed(gen_cfg.seed)

    fim_prompt = fim_format(prefix, suffix)
    input_ids = tokenizer(fim_prompt, return_tensors="pt").to(model.device)

    processors = _build_logits_processors(tokenizer, original_span, gen_cfg)

    generate_kwargs = dict(
        max_new_tokens=gen_cfg.max_new_tokens,
        no_repeat_ngram_size=gen_cfg.no_repeat_ngram_size,
        logits_processor=processors,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,   # <-- force off by default; avoids inherited True from model.generation_config
    )

    if gen_cfg.do_sample:
        generate_kwargs.update(
            dict(
                do_sample=True,              # <-- explicitly override to True when sampling
                top_p=gen_cfg.top_p,
                top_k=gen_cfg.top_k,
                temperature=gen_cfg.temperature,
                num_return_sequences=gen_cfg.num_return_sequences,
                num_beams=1,                # sampling + beams can work, but keep it simple here
                num_beam_groups=1,
                diversity_penalty=0.0,
            )
        )
    else:
        generate_kwargs.update(
            dict(
                num_beams=gen_cfg.num_beams,
                num_return_sequences=gen_cfg.num_return_sequences,
                num_beam_groups=gen_cfg.num_beam_groups,
                diversity_penalty=gen_cfg.diversity_penalty,
                do_sample=False,            # <-- be explicit for group beam search
            )
        )


    with torch.no_grad():
        out = model.generate(
            **input_ids,
            **generate_kwargs
        )

    # Decode gap-only
    cand_texts = []
    for i in range(out.shape[0]):
        text = _decode_gap_only(tokenizer, input_ids["input_ids"], out[i:i+1])
        text = _clean_text(text)
        if not text:
            continue
        if original_span and text.lower() == original_span.lower():
            continue
        cand_texts.append(text)

    # Deduplicate while preserving order
    cand_texts = _unique_keep_order(cand_texts)

    # Optional length filtering (approx via tokens)
    if gen_cfg.min_len or gen_cfg.max_len:
        kept = []
        for t in cand_texts:
            tok_len = len(tokenizer.encode(t, add_special_tokens=False))
            if gen_cfg.min_len and tok_len < gen_cfg.min_len:
                continue
            if gen_cfg.max_len and tok_len > gen_cfg.max_len:
                continue
            kept.append(t)
        cand_texts = kept

    return cand_texts


def generate_gap_candidates_until_saturation(
    tokenizer,
    model,
    prefix: str,
    suffix: str,
    original_span: Optional[str] = None,
    gen_cfg: Optional[GenerationConfig] = None,
    sat_cfg: Optional[SaturationConfig] = None,
) -> Dict:
    """
    Iteratively generate paraphrase candidates for the gap and stop when mass saturates.

    "Mass" is defined relative to the best candidate found so far:
        Z = sum_i exp(LL_i - LL_max)

    We stop when fractional growth in Z is below epsilon for `patience_rounds` consecutive rounds,
    or when limits are reached.

    Returns a dict with:
        {
          "candidates": List[Candidate],
          "Z": float,
          "rounds": int,
          "stopped_reason": str,
        }
    """
    if gen_cfg is None:
        gen_cfg = GenerationConfig()
    if sat_cfg is None:
        sat_cfg = SaturationConfig()

    all_texts: List[str] = []
    all_scores: Dict[str, Candidate] = {}

    rounds = 0
    best_ll = float("-inf")
    Z = 0.0
    prev_Z = 0.0
    under_epsilon_streak = 0
    stopped_reason = ""

    while True:
        rounds += 1
        # Generate one round
        new_texts = generate_one_round(tokenizer, model, prefix, suffix, original_span, gen_cfg)

        # Keep only NEW strings
        new_texts = [t for t in new_texts if t not in all_texts]

        # If too many, trim to batch size target to control runtime
        if len(new_texts) > sat_cfg.batch_candidates:
            new_texts = new_texts[:sat_cfg.batch_candidates]

        # Score new ones
        for t in new_texts:
            ll, tok_count = _score_span_logprob(tokenizer, model, prefix, t, suffix)
            all_texts.append(t)
            all_scores[t] = Candidate(text=t, total_logprob=ll, token_count=tok_count)
            if ll > best_ll:
                best_ll = ll

        # Recompute Z relative to best
        Z = 0.0
        for cand in all_scores.values():
            Z += math.exp(cand.total_logprob - best_ll)

        # Check saturation
        growth = 0.0
        if Z > 0:
            growth = (Z - prev_Z) / Z

        if growth < sat_cfg.epsilon:
            under_epsilon_streak += 1
        else:
            under_epsilon_streak = 0

        # Stopping conditions
        if under_epsilon_streak >= sat_cfg.patience_rounds:
            stopped_reason = f"saturated: fractional growth {growth:.4f} < epsilon {sat_cfg.epsilon} for {sat_cfg.patience_rounds} rounds"
            break
        if rounds >= sat_cfg.max_rounds:
            stopped_reason = f"max_rounds reached ({sat_cfg.max_rounds})"
            break
        if len(all_texts) >= sat_cfg.max_total_candidates:
            stopped_reason = f"max_total_candidates reached ({sat_cfg.max_total_candidates})"
            break
        if not new_texts:
            stopped_reason = "no new candidates found in this round"
            break

        prev_Z = Z

    # Prepare sorted candidates by logprob
    cands_sorted = sorted(all_scores.values(), key=lambda c: c.total_logprob, reverse=True)

    return {
        "candidates": cands_sorted,
        "Z": Z,
        "rounds": rounds,
        "stopped_reason": stopped_reason,
        "best_logprob": best_ll,
    }


def demo():
    """
    Demonstration with fake text:
        "The committee reached a decision after [GAP] carefully."
    where the original span is "considering all the evidence".
    """
    model_name = "Qwen/Qwen1.5-1.8B"  # smaller for local tests; change as needed
    prefix = "The committee reached a decision after "
    suffix = " carefully."
    original_span = "considering all the evidence"

    print("Loading model:", model_name, file=sys.stderr)
    tok, model = load_qwen(model_name)

    gen_cfg = GenerationConfig(
        num_beams=12,
        num_return_sequences=12,
        num_beam_groups=6,
        diversity_penalty=0.7,
        temperature=0.7,
        top_p=0.95,
        top_k=0,
        do_sample=False,        # start with diverse beams; you can switch to True for sampling later rounds
        max_new_tokens=12,
        no_repeat_ngram_size=3,
        ban_original_first_token=True,
        min_len=0,
        max_len=20,
        seed=42
    )

    sat_cfg = SaturationConfig(
        epsilon=0.03,
        patience_rounds=2,
        max_rounds=8,
        batch_candidates=40,
        max_total_candidates=200
    )

    results = generate_gap_candidates_until_saturation(
        tok, model, prefix, suffix, original_span, gen_cfg, sat_cfg
    )

    print("\nStopped:", results["stopped_reason"])
    print("Rounds:", results["rounds"])
    print("Z (relative mass):", f"{results['Z']:.3f}")
    print("Best logprob:", f"{results['best_logprob']:.3f}")
    print("\nTop candidates:\n")
    for i, c in enumerate(results["candidates"][:20], 1):
        print(f"{i:>2}. {c.text}   [LL={c.total_logprob:.3f}, tokens={c.token_count}]")


if __name__ == "__main__":
    demo()