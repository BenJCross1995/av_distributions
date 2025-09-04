import re
import torch
import math

import pandas as pd

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Set, Tuple, Optional, Iterable, Union

def common_ngrams(
    text1: str,
    text2: str,
    n: int,
    model: Any = None,
    tokenizer: Any = None,
    include_subgrams: bool = False,
    lowercase: bool = True,
) -> Dict[int, Set[Tuple[Any, ...]]]:
    """
    Return shared n-grams of length >= n between two texts.

    If include_subgrams is False (default), remove any shared n-gram that is a
    contiguous subspan of a longer shared n-gram. (So a 5-gram that’s part of a
    shared 6-gram is excluded; unrelated 5-grams remain.)

    Parameters
    ----------
    lowercase : bool, default True
        If True, normalize text using str.casefold() before tokenization.
        Applies to both the simple regex tokenization path and the Hugging Face
        tokenizer path (by case-folding the raw text before calling the tokenizer).
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    def _word_tokens(s: str) -> List[str]:
        s2 = s.casefold() if lowercase else s
        return re.findall(r"\w+", s2)

    def _hf_tokens(txt: str) -> List[Any]:
        src = txt.casefold() if lowercase else txt
        if hasattr(tokenizer, "tokenize"):
            return list(tokenizer.tokenize(src))
        enc = tokenizer(
            src,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = enc.get("input_ids", [])
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            input_ids = input_ids[0]
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            return tokenizer.convert_ids_to_tokens(input_ids)
        return input_ids

    def _ngrams_by_len(seq: Sequence[Any], min_n: int) -> Dict[int, Set[Tuple[Any, ...]]]:
        out: Dict[int, Set[Tuple[Any, ...]]] = {}
        L = len(seq)
        for k in range(min_n, L + 1):
            s: Set[Tuple[Any, ...]] = set()
            for i in range(0, L - k + 1):
                s.add(tuple(seq[i : i + k]))
            if s:
                out[k] = s
        return out

    token_mode = (model is not None) and (tokenizer is not None)
    seq1 = _hf_tokens(text1) if token_mode else _word_tokens(text1)
    seq2 = _hf_tokens(text2) if token_mode else _word_tokens(text2)

    ngrams1 = _ngrams_by_len(seq1, n)
    ngrams2 = _ngrams_by_len(seq2, n)

    common: Dict[int, Set[Tuple[Any, ...]]] = {}
    for k in set(ngrams1.keys()).intersection(ngrams2.keys()):
        inter = ngrams1[k] & ngrams2[k]
        if inter:
            common[k] = inter

    if include_subgrams or not common:
        return common

    # Remove n-grams that are contiguous subspans of any longer shared n-gram
    to_remove: Dict[int, Set[Tuple[Any, ...]]] = defaultdict(set)
    lengths = sorted(common.keys())
    for k in lengths:
        # For each longer length, generate all contiguous subspans down to n
        for longer_k in [L for L in lengths if L > k]:
            for g in common[longer_k]:
                # produce all subspans of length k from g
                for i in range(0, longer_k - k + 1):
                    to_remove[k].add(g[i : i + k])

    # Apply removals
    for k, rem in to_remove.items():
        if k in common:
            common[k] = {g for g in common[k] if g not in rem}
            if not common[k]:
                del common[k]

    return common

from typing import Any, Dict, List, Set, Tuple, Union

def pretty_print_common_ngrams(
    common: Dict[int, Set[Tuple[Any, ...]]],
    sep: str = " ",
    order: str = "count_desc",      # "count_desc" | "len_asc" | "len_desc"
    tokenizer=None,                 # Optional HuggingFace tokenizer
    return_format: str = "print",   # "print" | "flat" | "grouped"
) -> Union[None, List[str], Dict[int, List[str]]]:
    """
    Pretty-print or return shared n-grams.

    - Groups by n (the integer length).
    - If `tokenizer` is None: converts each n-gram tuple into a string joined by `sep` (original behavior).
    - If `tokenizer` is provided: decodes token ids/strings to readable text (special tokens removed).
    - `order` controls group ordering (and the flattening order for "flat").
    - `return_format`:
        * "print"   -> prints grouped output and returns None (default; original behavior).
        * "flat"    -> returns a single flattened list of strings across all n values, ordered by `order`.
        * "grouped" -> returns a dict[int, list[str]] of strings per n (keys are the n values).
    """
    if not common:
        if return_format == "print":
            print("{}")
            return None
        return [] if return_format == "flat" else {}

    def stringify_ngram(ngram: Tuple[Any, ...]) -> str:
        # Original behavior (no tokenizer): join items with sep
        if tokenizer is None:
            return sep.join(map(str, ngram))

        # With tokenizer: decode to human-readable text
        toks = list(ngram)

        # If everything is ids, use fast decode
        if all(isinstance(t, int) for t in toks):
            return tokenizer.decode(
                toks,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        # Otherwise, we may have token *strings* or a mix of ids & strings
        specials = set(getattr(tokenizer, "all_special_tokens", []))
        norm_tokens: List[str] = []
        for t in toks:
            if isinstance(t, int):
                # convert id -> token string
                norm_tokens.append(tokenizer.convert_ids_to_tokens(t))
            else:
                norm_tokens.append(str(t))

        # Drop special tokens (e.g., <s>, </s>)
        norm_tokens = [t for t in norm_tokens if t not in specials]

        # Let the tokenizer handle spacing/newlines between tokens
        return tokenizer.convert_tokens_to_string(norm_tokens)

    # Convert tuples to strings per length key
    grouped: Dict[int, List[str]] = {
        n: sorted(stringify_ngram(g) for g in grams)
        for n, grams in common.items()
    }

    # Choose group ordering
    if order == "count_desc":
        items = sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    elif order == "len_asc":
        items = sorted(grouped.items(), key=lambda kv: kv[0])
    elif order == "len_desc":
        items = sorted(grouped.items(), key=lambda kv: -kv[0])
    else:
        # Fallback to default
        items = sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    if return_format == "flat":
        # Flatten respecting the chosen group order and per-group alpha sorting
        flat: List[str] = [s for _, strings in items for s in strings]
        return flat

    if return_format == "grouped":
        # Return the grouped mapping (unordered by default dict semantics).
        # If you want the ordering preserved, convert `items` to an OrderedDict externally.
        return grouped

    # Default: print grouped nicely and return None
    for n, strings in items:
        print(f"{n}-grams ({len(strings)}): {strings}")
    return None

        
def highest_common(common: Dict[int, Set[Tuple[Any, ...]]]) -> Tuple[int, Set[Tuple[Any, ...]]]:
    """
    Given the dict returned by `common_ngrams`, return (max_n, ngrams_at_max).
    If there are none, returns (0, empty set).
    """
    if not common:
        return 0, set()
    max_k = max(common.keys())
    return max_k, common[max_k]

def largest_common_ngram_problems(
    metadata: pd.DataFrame,
    known: pd.DataFrame,
    unknown: pd.DataFrame,
    n: int,
    model: Any = None,
    tokenizer: Any = None,
    print_progress: bool = True,
) -> pd.DataFrame:
    """
    For each metadata row (keeps, problem, known_author, unknown_author, known_doc_id, unknown_doc_id),
    filter `known`/`unknown` by doc_id, take text via .reset_index().loc[0, 'text'],
    compute common_ngrams(...) then highest_common(common), and extract:
      - highes_common_count: the number (first element)
      - highes_common_ngram: the n-gram as a space-joined string

    Returns columns:
      ['keeps','problem','known_author','unknown_author','known_doc_id','unknown_doc_id',
       'highes_common_count','highes_common_ngram']
    """
    required_meta_cols = [
        "problem", "known_author", "unknown_author",
        "known_doc_id", "unknown_doc_id",
    ]
    missing_meta = [c for c in required_meta_cols if c not in metadata.columns]
    if missing_meta:
        raise ValueError(f"metadata missing columns: {missing_meta}")

    for df_name, df in [("known", known), ("unknown", unknown)]:
        if "doc_id" not in df.columns:
            raise ValueError(f"'{df_name}' is missing required column 'doc_id'")
        if "text" not in df.columns:
            raise ValueError(f"'{df_name}' is missing required column 'text'")

    def _pick_ngram_string(ngrams: Any) -> str:
        """
        Accepts one of:
          - a tuple/list of tokens (single n-gram),
          - a set/list of n-gram tuples (choose deterministic first),
          - an already-joined string.
        Returns a single space-joined n-gram string.
        """
        # If we received a collection of n-grams, pick a deterministic one
        if isinstance(ngrams, (set, list, tuple)) and ngrams and isinstance(next(iter(ngrams)), (tuple, list, str)):
            # If it's a set/list of tuples/lists/strings, sort deterministically
            if isinstance(ngrams, (set, list)) and ngrams and not isinstance(ngrams, str):
                try:
                    candidate = sorted(ngrams)[0]
                except Exception:
                    candidate = next(iter(ngrams))
            else:
                candidate = ngrams  # already a single n-gram tuple/list/str
        else:
            candidate = ngrams

        # If candidate is a sequence of tokens, join with spaces; otherwise cast to str
        if isinstance(candidate, (tuple, list)):
            return " ".join(map(str, candidate))
        return str(candidate)

    rows: List[Dict[str, Any]] = []
    total = len(metadata)
    it = metadata[required_meta_cols].itertuples(index=False, name="MetaRow")

    for i, row in enumerate(it, 1):
        problem, known_author, unknown_author, known_doc_id, unknown_doc_id = row

        kdf = known.loc[known["doc_id"] == known_doc_id].reset_index(drop=True)
        udf = unknown.loc[unknown["doc_id"] == unknown_doc_id].reset_index(drop=True)

        if kdf.empty or udf.empty:
            count_val = None
            ngram_str = None
        else:
            text_known = kdf.loc[0, "text"]
            text_unknown = udf.loc[0, "text"]

            # If your common_ngrams expects n, switch to: common_ngrams(text_known, text_unknown, n)
            common = common_ngrams(text_known, text_unknown, n=n, model=model, tokenizer=tokenizer)
            hc = highest_common(common)

            if hc is None:
                count_val = None
                ngram_str = None
            else:
                # Expecting (number, ngrams)
                try:
                    count_val, ngrams_obj = hc
                except Exception:
                    # Fallback: treat whole object as the ngram payload and set count None
                    count_val = None
                    ngrams_obj = hc
                ngram_str = _pick_ngram_string(ngrams_obj)

        rows.append({
            "problem": problem,
            "known_author": known_author,
            "unknown_author": unknown_author,
            "known_doc_id": known_doc_id,
            "unknown_doc_id": unknown_doc_id,
            "highest_common_count": count_val,      # extracted number
            "highest_common_ngram": ngram_str,      # tokens joined by ' '
        })

        if print_progress and total:
            if (i % max(1, total // 50) == 0) or (i == total):
                pct = int(i * 100 / total)
                print(f"\rProcessed {i}/{total} ({pct}%)", end="")

    if print_progress:
        print()

    return pd.DataFrame(rows)

def concat_text_by_ids(
    doc_ids: Iterable,
    df: pd.DataFrame,
    id_col: str = "doc_id",
    text_col: str = "text",
    sep: str = "\n",
    unique: bool = False,   # de-dupe doc_ids while preserving order
    strict: bool = False,   # raise if any doc_id is missing
    dropna: bool = True,    # drop rows where text is NaN
) -> str:
    """
    Return a single string with texts for the given doc_ids concatenated by `sep`.

    - Order follows `doc_ids`.
    - If a doc_id matches multiple rows, their texts are joined by `sep` first,
      then that block is joined into the overall result (also using `sep`).
    """
    # Optionally de-duplicate the provided IDs, preserving order
    if unique:
        seen = set()
        doc_ids = [d for d in doc_ids if not (d in seen or seen.add(d))]

    # Optionally drop NaNs in the text column
    if dropna:
        df = df[df[text_col].notna()].copy()

    # Build a mapping: doc_id -> [texts...], preserving row order
    grouped = df.groupby(id_col, sort=False)[text_col].apply(list).to_dict()

    parts = []
    missing = []
    for d in doc_ids:
        texts = grouped.get(d)
        if texts is None:
            missing.append(d)
            continue
        parts.append(sep.join(str(t) for t in texts))

    if strict and missing:
        raise KeyError(f"Missing doc_ids in dataframe: {missing}")

    # If not strict, we just skip missing IDs
    return sep.join(parts)

def concat_text_by_group(
    df: pd.DataFrame,
    group_cols: Union[str, Sequence[str]] = "author",
    text_col: str = "text",
    sep: str = "\n",
    dropna: bool = True,
    keep_group_order: bool = True,              # keep first-seen group order
    keep_row_order_within_group: bool = True,   # keep row order inside each group
    output_col: str = "concat_text",
) -> pd.DataFrame:
    """
    Group `df` by one or more columns and concatenate each group's `text_col`
    joined by `sep`. Returns a DataFrame with the group columns plus `output_col`.

    Parameters
    ----------
    group_cols : str | Sequence[str]
        A single column name or a list/tuple of column names to group by.
    """
    # Normalize group_cols to a list
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    else:
        group_cols = list(group_cols)

    # Validate columns
    required = set(group_cols + [text_col])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in DataFrame: {missing}")

    # Minimal copy
    df2 = df[group_cols + [text_col]].copy()

    if dropna:
        df2 = df2[df2[text_col].notna()]

    # Ensure text is string
    df2[text_col] = df2[text_col].astype(str)

    # Control row order within groups
    if not keep_row_order_within_group:
        # Basic option: sort texts lexicographically within each group
        df2 = df2.sort_values(group_cols + [text_col])

    # Group and concatenate
    out = (
        df2.groupby(group_cols, sort=not keep_group_order)[text_col]
           .apply(lambda s: sep.join(s.tolist()))
           .reset_index()
           .rename(columns={text_col: output_col})
    )
    return out

def largest_common_ngram_profile_problems(
    metadata: pd.DataFrame,
    known: pd.DataFrame,
    unknown: pd.DataFrame,
    n: int,
    model: Any = None,
    tokenizer: Any = None,
    lowercase: bool = True,         # passed through to common_ngrams
    include_subgrams: bool = False, # passed through to common_ngrams if supported
    sep: str = "\n",                # newline separator when concatenating
    print_progress: bool = True,
) -> pd.DataFrame:
    """
    For each metadata row (problem, known_author, unknown_author):
      1) In `known`, filter to `known_author`, then group by (corpus, author, texttype).
         Concatenate each group's texts (preserving row order) with `sep`, then
         concatenate those group blobs (also with `sep`) to make ONE big `known_text`.
      2) In `unknown`, filter to `unknown_author`. If multiple rows exist, concatenate with `sep`
         to make ONE big `unknown_text`.
      3) Compute common_ngrams(known_text, unknown_text, n, ...) and take the largest/common
         profile via `highest_common`.

    Returns columns in this exact order:
      ['problem','known_author','unknown_author',
       'known_text','unknown_text',
       'highest_common_count','highest_common_ngram']
    """

    # --- validations ---
    required_meta = ["problem", "known_author", "unknown_author"]
    miss_meta = [c for c in required_meta if c not in metadata.columns]
    if miss_meta:
        raise ValueError(f"`metadata` missing columns: {miss_meta}")

    for df_name, df, req_cols in [
        ("known", known, ["corpus", "author", "texttype", "text"]),
        ("unknown", unknown, ["author", "text"]),
    ]:
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            raise ValueError(f"`{df_name}` missing columns: {missing}")

    # --- helpers ---
    def _pick_ngram_string(ngrams: Any) -> Optional[str]:
        """Pick a deterministic n-gram and join tokens with spaces."""
        if ngrams is None:
            return None
        candidate = ngrams
        if isinstance(ngrams, (set, list)) and ngrams:
            try:
                candidate = sorted(ngrams)[0]
            except Exception:
                candidate = next(iter(ngrams))
        if isinstance(candidate, (tuple, list)):
            return " ".join(map(str, candidate))
        return str(candidate)

    def _highest_common_fallback(common: Dict[int, Any]) -> Optional[Tuple[int, Any]]:
        """Fallback if `highest_common` isn't defined."""
        if not common:
            return None
        longest_n = max(common.keys())
        grams = common[longest_n]
        try:
            count = len(grams)
        except Exception:
            count = None
        return (count, grams)

    _highest_common = globals().get("highest_common", _highest_common_fallback)

    def _concat_known_for_author(author: Any) -> Optional[str]:
        ksub = known.loc[known["author"] == author, ["corpus", "author", "texttype", "text"]].copy()
        if ksub.empty:
            return None
        ksub = ksub[ksub["text"].notna()]
        if ksub.empty:
            return None
        ksub["text"] = ksub["text"].astype(str)

        parts: List[str] = []
        # First-seen order for groups and rows within groups
        for _, g in ksub.groupby(["corpus", "author", "texttype"], sort=False):
            grp_text = sep.join(g["text"].tolist())
            if grp_text:
                parts.append(grp_text)
        return sep.join(parts) if parts else None

    def _concat_unknown_for_author(author: Any) -> Optional[str]:
        usub = unknown.loc[unknown["author"] == author, ["author", "text"]].copy()
        if usub.empty:
            return None
        usub = usub[usub["text"].notna()]
        if usub.empty:
            return None
        usub["text"] = usub["text"].astype(str)
        return sep.join(usub["text"].tolist())

    # --- main loop ---
    rows: List[Dict[str, Any]] = []
    total = len(metadata)
    it = metadata[required_meta].itertuples(index=False, name="MetaRow")

    for i, meta_row in enumerate(it, 1):
        problem, known_author, unknown_author = meta_row

        known_text   = _concat_known_for_author(known_author)
        unknown_text = _concat_unknown_for_author(unknown_author)

        if not known_text or not unknown_text:
            count_val = None
            ngram_str = None
        else:
            # Compute common n-grams
            try:
                common = common_ngrams(
                    known_text,
                    unknown_text,
                    n=n,
                    model=model,
                    tokenizer=tokenizer,
                    include_subgrams=include_subgrams,
                    lowercase=lowercase,  # requires your updated common_ngrams
                )
            except TypeError:
                # Backward-compat if your common_ngrams doesn't take lowercase/include_subgrams
                common = common_ngrams(
                    known_text,
                    unknown_text,
                    n=n,
                    model=model,
                    tokenizer=tokenizer,
                )

            hc = _highest_common(common) if common else None
            if hc is None:
                count_val = None
                ngram_str = None
            else:
                try:
                    count_val, ngrams_obj = hc
                except Exception:
                    count_val = None
                    ngrams_obj = hc
                ngram_str = _pick_ngram_string(ngrams_obj)

        rows.append({
            "problem": problem,
            "known_author": known_author,
            "unknown_author": unknown_author,
            "known_text": known_text,
            "unknown_text": unknown_text,
            "highest_common_count": count_val,
            "highest_common_ngram": ngram_str,
        })

        if print_progress and total:
            if (i % max(1, total // 50) == 0) or (i == total):
                pct = int(i * 100 / total)
                print(f"\rProcessed {i}/{total} ({pct}%)", end="")

    if print_progress:
        print()

    # Ensure column order as requested
    cols = [
        "problem", "known_author", "unknown_author",
        "known_text", "unknown_text",
        "highest_common_count", "highest_common_ngram",
    ]
    df_out = pd.DataFrame(rows)
    return df_out.reindex(columns=cols)

def keep_before_phrase(text: str, phrase: str, case_insensitive: bool = False) -> str:
    """
    Return everything in `text` before the first occurrence of `phrase`.
    If `phrase` isn’t found, returns the entire `text`.

    :param text:       The full string you want to trim.
    :param phrase:     The substring (phrase) you want to stop at.
    :param case_insensitive:  If True, match phrase ignoring case.
    :return:           The portion of `text` before `phrase`.
    """
    if case_insensitive:
        idx = text.lower().find(phrase.lower())
    else:
        idx = text.find(phrase)

    return text[:idx] if idx != -1 else text

def compute_log_probs_with_median(text: str, tokenizer, model):
    """
    For each token (including the first), returns:
      - tokens: list of tokenizer.convert_ids_to_tokens
      - log_probs: list of log-probs for each token
      - median_logprobs: median log-prob of the distribution at each step
    """
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]         # shape [1, seq_len]
    # --- ALIGN TOKENS CORRECTLY HERE ---
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits                 # [1, seq_len, vocab_size]

    log_probs = []
    median_logprobs = []
    # for each position i, look at logits from i-1 (or the BOS for i=0)
    for i in range(input_ids.size(1)):
        prev_idx = 0 if i == 0 else i - 1
        dist = torch.log_softmax(logits[0, prev_idx], dim=-1)
        log_prob = dist[input_ids[0, i]].item()
        median_lp = float(dist.median().item())
        log_probs.append(log_prob)
        median_logprobs.append(median_lp)

    return tokens, log_probs, median_logprobs

def score_phrases(
    base_text: str,
    ref_phrase: str,
    paraphrases: List[str],
    tokenizer,
    model
) -> pd.DataFrame:
    """
    1) Score base_text alone → base_total
    2) For each phrase (reference + paraphrases):
         a) Get its token count by scoring phrase alone
         b) Score base_text + phrase → full tokens & log_probs
         c) sum_before = sum(full log_probs)
         d) phrase_tokens    = last n_phrase tokens of full tokens
         e) phrase_log_probs = last n_phrase values of full log_probs
         f) phrase_total     = sum(phrase_log_probs)
         g) difference       = base_total - sum_before
         h) APPEND row
    3) Return DataFrame with columns:
       phrase_type, phrase, tokens, base_total, sum_before,
       log_probs, phrase_total, difference
    """
    # 1) score base_text
    print("→ Scoring base_text alone…")
    _, log_probs_base, _ = compute_log_probs_with_median(base_text.strip(), tokenizer, model)
    base_total = sum(log_probs_base)
    print(f"   base_total = {base_total:.4f}\n")

    items = [("reference", ref_phrase)] + [("paraphrase", p) for p in paraphrases]
    rows = []

    for idx, (ptype, phrase) in enumerate(items, start=1):
        print(f"→ [{idx}/{len(items)}] Processing {ptype}…")

        # a) phrase alone → get token count
        tokens_phrase, log_probs_phrase, _ = compute_log_probs_with_median(phrase, tokenizer, model)
        n_phrase_tokens = len(tokens_phrase)
        # b) full sequence
        full_text = base_text + phrase
        tokens_full, log_probs_full, _ = compute_log_probs_with_median(full_text, tokenizer, model)
        # c) full sum
        sum_before = sum(log_probs_full)
        # d/e) slice last n_phrase_tokens
        phrase_tokens    = tokens_full[-n_phrase_tokens:]
        phrase_log_probs = log_probs_full[-n_phrase_tokens:]
        # f/g) compute sums
        phrase_total = sum(phrase_log_probs)
        difference   = base_total - sum_before
        # h) collect
        rows.append({
            "phrase_type":  ptype,
            "phrase":       phrase,
            "tokens":       phrase_tokens,
            "sum_log_probs_base":   base_total,
            "sum_log_probs_inc_phrase":   sum_before,
            "difference":   difference,
            "phrase_log_probs":    phrase_log_probs,
            "sum_log_probs_phrase": phrase_total,
        })

    return pd.DataFrame(rows, columns=[
        "phrase_type", "phrase", "tokens",
        "sum_log_probs_base", "sum_log_probs_inc_phrase",
        "difference", "phrase_log_probs", "sum_log_probs_phrase",
    ])
    
def _logsumexp(xs: Sequence[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))

def add_pmf_column(
    df: pd.DataFrame,
    logprob_col: str,
    priors: Optional[Union[str, Sequence[float]]] = None,
    out_col: str = "pmf",
    keep_logZ: bool = False,
) -> pd.DataFrame:
    """
    Treat each row as a candidate. `logprob_col` contains a list of token log-probs per row.
    Computes P(row i) ∝ exp(sum(logprobs_i)) * prior_i and writes it to `out_col`.

    priors:
      - None: uniform
      - str: name of a column holding prior probabilities per row
      - Sequence[float]: prior probs aligned with df.index
    """
    # sequence log-likelihood per row
    L = df[logprob_col].apply(
        lambda xs: sum(xs) if isinstance(xs, (list, tuple)) and len(xs) > 0 else float("-inf")
    ).tolist()

    # apply priors (in probability space)
    if priors is None:
        L_adj = L
    else:
        if isinstance(priors, str):
            prior_vals = df[priors].tolist()
        else:
            prior_vals = list(priors)
            if len(prior_vals) != len(L):
                raise ValueError("Length of `priors` must match number of rows.")
        L_adj = [Li + (math.log(p) if (p is not None and p > 0) else float("-inf"))
                 for Li, p in zip(L, prior_vals)]

    # normalize across rows (stable)
    logZ = _logsumexp(L_adj)
    pmf = [math.exp(Li - logZ) if Li != float("-inf") else 0.0 for Li in L_adj]

    df = df.copy()
    df[out_col] = pmf
    if keep_logZ:
        df["_logZ"] = logZ  # same for all rows
    return df

def score_phrases_no_context(
    ref_phrase: str,
    paraphrases: List[str],
    tokenizer,
    model
) -> pd.DataFrame:
    """
    1) Score base_text alone → base_total
    2) For each phrase (reference + paraphrases):
         a) Get its token count by scoring phrase alone
         b) Score base_text + phrase → full tokens & log_probs
         c) sum_before = sum(full log_probs)
         d) phrase_tokens    = last n_phrase tokens of full tokens
         e) phrase_log_probs = last n_phrase values of full log_probs
         f) phrase_total     = sum(phrase_log_probs)
         g) difference       = base_total - sum_before
         h) APPEND row
    3) Return DataFrame with columns:
       phrase_type, phrase, tokens, base_total, sum_before,
       log_probs, phrase_total, difference
    """
    # 1) score base_text
    items = [("reference", ref_phrase)] + [("paraphrase", p) for p in paraphrases]
    rows = []

    for idx, (ptype, phrase) in enumerate(items, start=1):
        print(f"→ [{idx}/{len(items)}] Processing {ptype}…")

        # a) phrase alone → get token count
        tokens_phrase, log_probs_phrase, _ = compute_log_probs_with_median(phrase, tokenizer, model)
        # b) compute sum
        phrase_total = sum(log_probs_phrase)
        # h) collect
        rows.append({
            "phrase_type":  ptype,
            "phrase":       phrase,
            "tokens":       tokens_phrase,
            "log_probs":    log_probs_phrase,
            "sum_log_probs": phrase_total,
        })

    return pd.DataFrame(rows, columns=[
        "phrase_type", "phrase", "tokens",
        "log_probs", "sum_log_probs",
    ])