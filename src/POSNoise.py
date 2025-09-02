import pandas as pd
import spacy

# Load spaCy with an English model that provides Universal POS tags.
# (Using the small model for example; authors used 'en_core_web_lg' for higher accuracy:contentReference[oaicite:45]{index=45}.)
nlp = spacy.load("en_core_web_sm")  

# Define the POS category to placeholder symbol mapping (as per Halvani & Graner 2021, Table 2).
pos_to_symbol = {
    "NOUN": "#",   # common noun
    "PROPN": "§",  # proper noun
    "VERB": "Ø",   # verb
    "ADJ": "@",    # adjective
    "ADV": "©",    # adverb
    "NUM": "µ",    # numeral (digits, numbers) 
    "SYM": "$",    # symbol or emoji
    "X": "¥"       # other/unrecognized (foreign words, etc.)
}

# Define a set of contraction tokens to preserve (common clitics split by the tokenizer).
contraction_tokens = {"'m", "'d", "'s", "'re", "'ve", "'ll", "n't", "'t"}  
# (This covers tokens like "'s", "n't", "'m" etc. that spaCy may treat as separate tokens.)

# Optionally, define a list of function words to always keep (could use a curated list or spaCy stop words).
# Here we use spaCy's built-in list of stop words as a proxy for common function words.
# (In a full implementation, you'd use the exact list L of function words/phrases from the POSNoise method.)
function_words = set([w.lower() for w in nlp.Defaults.stop_words])  

def posnoise_transform(text: str) -> str:
    """Apply POSNoise content masking to a single text string."""
    doc = nlp(text)
    transformed_tokens = []
    for token in doc:
        # Preserve whitespace from the original text (to keep punctuation attachment).
        text_token = token.text  # original token text
        ws = token.whitespace_   # original trailing whitespace (empty if none)

        if token.is_space:
            # If the token is purely whitespace (e.g., spaCy might tokenize multiple spaces/newlines as a token).
            # We add it as is (though normally spaCy will handle spaces via whitespace_).
            transformed_tokens.append(text_token)
            continue

        # If token is a contraction or possessive clitic that should remain (e.g., "'s", "n't"),
        # or if it's in the function word list (content-independent word), **do not mask it**.
        if text_token in contraction_tokens or text_token.lower() in function_words:
            transformed_tokens.append(text_token + ws)
            continue

        # Determine the coarse POS tag for this token.
        pos_tag = token.pos_  # e.g., "NOUN", "VERB", "ADJ", etc. (Universal POS tag)
        if pos_tag in pos_to_symbol:
            # This token is a content word (open-class POS) that should be masked.
            # Special case: if it's a NUM but appears to be a written-out number (alphabetic), keep it.
            if pos_tag == "NUM":
                # Check if token.text contains any digit character (or is a Roman numeral).
                # If no digits and not a roman numeral, we treat it as a written number and do not mask.
                is_roman = text_token.isalpha() and text_token.upper() in ["I","V","X","L","C","D","M"]
                if text_token.isdigit() or is_roman:
                    transformed_tokens.append(pos_to_symbol["NUM"] + ws)  # mask numeric/roman numerals as µ
                else:
                    transformed_tokens.append(text_token + ws)  # keep spelled-out number (e.g., "ten")
            else:
                # Replace token with its POS placeholder symbol.
                transformed_tokens.append(pos_to_symbol[pos_tag] + ws)
        else:
            # If the POS tag is not in our mapping, it means the token is not a content word category.
            # This includes pronouns, determiners, conjunctions, prepositions, etc., as well as punctuation.
            # We retain such tokens as they are.
            transformed_tokens.append(text_token + ws)
    # Join all pieces into the transformed string. Using the preserved whitespace ensures 
    # punctuation and spacing remain exactly as in the original.
    return "".join(transformed_tokens)

def apply_posnoise(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Apply POSNoise transformation to all texts in a DataFrame column."""
    # Create a new column with POSNoise-transformed text
    df[text_col + "_POSNoise"] = df[text_col].apply(posnoise_transform)
    return df

# Example usage:
# df = pd.DataFrame({"text": ["Like before, further improvements to this section are welcome."]})
# df = apply_posnoise(df, "text")
# print(df["text_POSNoise"].iloc[0])
# Expected output: "Like before, further # to this # are @."
