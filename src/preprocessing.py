import spacy

import pandas as pd
import re

from nltk.tokenize import sent_tokenize

def load_ner_model(model="en_core_web_sm"):
    """Load a Named Entity Recognition model and add additional entities
    
    Parameters:
    - model (str): The NER model to load
    
    Returns:
    - An initialised NER model
    """
    nlp = spacy.load(model)

    # Add custom rule for abbreviations like Capt. and others
    # Add custom rules for abbreviations
    abbreviation_patterns = [
        # Personal titles
        {"label": "PERSON", "pattern": [{"LOWER": "capt."}]},
        {"label": "PERSON", "pattern": [{"LOWER": "dr."}]},
        {"label": "PERSON", "pattern": [{"LOWER": "mr."}]},
        {"label": "PERSON", "pattern": [{"LOWER": "mrs."}]},
        {"label": "PERSON", "pattern": [{"LOWER": "prof."}]},
        {"label": "PERSON", "pattern": [{"LOWER": "rev."}]},
        {"label": "PERSON", "pattern": [{"LOWER": "sr."}]},
        {"label": "PERSON", "pattern": [{"LOWER": "jr."}]},

        # Common Latin abbreviations
        {"label": "ABBREVIATION", "pattern": [{"LOWER": "e.g."}]},  # For example
        {"label": "ABBREVIATION", "pattern": [{"LOWER": "i.e."}]},  # That is
        {"label": "ABBREVIATION", "pattern": [{"LOWER": "etc."}]},  # Et cetera
        {"label": "ABBREVIATION", "pattern": [{"LOWER": "et al."}]},  # And others
        {"label": "ABBREVIATION", "pattern": [{"LOWER": "a.m."}]},  # Ante meridiem
        {"label": "ABBREVIATION", "pattern": [{"LOWER": "p.m."}]},  # Post meridiem
        {"label": "ABBREVIATION", "pattern": [{"LOWER": "vs."}]},   # Versus
        {"label": "ABBREVIATION", "pattern": [{"LOWER": "cf."}]},   # Compare
        {"label": "ABBREVIATION", "pattern": [{"LOWER": "viz."}]},  # Namely
        {"label": "ABBREVIATION", "pattern": [{"LOWER": "ca."}]},   # Circa

        # Additional abbreviations
        # {"label": "ABBREVIATION", "pattern": [{"LOWER": "no."}]},  # Number - Ignore 
        {"label": "ABBREVIATION", "pattern": [{"LOWER": "vol."}]},  # Volume
        {"label": "ABBREVIATION", "pattern": [{"LOWER": "pp."}]},   # Pages
        {"label": "ABBREVIATION", "pattern": [{"LOWER": "fig."}]},  # Figure
    ]

    # Adding the patterns to spaCy's pipeline
    ruler = nlp.add_pipe("entity_ruler", before="ner", name="abbreviation_ruler")
    ruler.add_patterns(abbreviation_patterns)
    
    return nlp

def process_text(text, model):
    """
    Process the input text by performing Named Entity Recognition (NER), masking the entities,
    splitting the text into sentences, and then restoring the entities with their original text.

    Parameters:
    - text (str): The input text to process, including named entities.
    - model (str): The model to use for NER

    Returns:
    - list: A list of sentences with named entities restored to their original form.
    """
    
    # Step 1: Perform Named Entity Recognition (NER)
    doc = model(text)
    
    # Create a dictionary where the keys are the original entities and the values are placeholders
    entities = {ent.text: f"__ENTITY{idx}__" for idx, ent in enumerate(doc.ents)}

    # Step 2: Mask named entities in the text
    masked_text = text
    
    # Sort entities by length to prevent conflicts (longer entities are replaced first)
    sorted_entities = sorted(entities.items(), key=lambda x: -len(x[0]))
    
    for entity, placeholder in sorted_entities:
        # Use regex to ensure we match whole words only (avoiding partial replacements)
        masked_text = re.sub(r'\b' + re.escape(entity) + r'\b', placeholder, masked_text)

    # Step 3: Perform Sentence Splitting using NLTK
    sentences = sent_tokenize(masked_text)

    # Step 3: Perform Sentence Splitting using Spacy
    # doc = nlp(masked_text)
    # sentences = [sent.text for sent in doc.sents]

    reversed_entities = {v: k for k, v in entities.items()}
    
    # Step 4: Restore Named Entities in sentences using regex for exact match
    restored_sentences = []
    for sentence in sentences:
        restored_sentence = sentence
        for placeholder, entity in reversed_entities.items():
            # Replace placeholders with original entities using \b for word boundaries
            restored_sentence = re.sub(r'\b' + re.escape(placeholder) + r'\b', entity, restored_sentence)
        restored_sentences.append(restored_sentence)

    return restored_sentences

def preprocess_and_process_text(text, model):
    """
    Preprocesses the text by replacing '\\' with a placeholder, processes with NER and sentence splitting,
    then restores the '\\' placeholders back to '\\'.

    Parameters:
    - text (str): The original text to process.

    Returns:
    - list: The list of sentences with entities restored, including correct handling of backslashes.
    """
    # Preprocess step: Replace '\\' with a placeholder to avoid issues during NER
    text = text.replace("\\", "__BACKSLASH_PLACEHOLDER__")
    text = text.replace("(??)", "__REF_ERROR__")

    # Now apply the process_text function
    processed_sentences = process_text(text, model)

    # Restore the '\\' back in the processed sentences
    processed_sentences_with_backslashes = [
        sentence.replace('__BACKSLASH_PLACEHOLDER__', r'\\') for sentence in processed_sentences
    ]
    processed_sentences_with_backslashes = [
        sentence.replace('__REF_ERROR__', "(??)") for sentence in processed_sentences
    ]
    
    return processed_sentences_with_backslashes

def vectorize_df(df, model="en_core_web_sm", impostors=False):

	# Step 1: Perform Named Entity Recognition (NER)
	nlp_model = load_ner_model(model)
	
	df_copy = df.copy()
        
	# Apply the function to the DataFrame
	df_copy["sentence"] = df_copy["text"].apply(preprocess_and_process_text, model=nlp_model)
    
	# Expand the processed sentences into separate rows if needed
	expanded_df = df_copy.explode("sentence").reset_index(drop=True)

	# Step 3: Add chunk_id column (sequential numbering grouped by doc_id)
	if impostors:
		expanded_df["chunk_id"] = expanded_df.groupby(["doc_id", "impostor_id"]).cumcount() + 1
	else:
		expanded_df["chunk_id"] = expanded_df.groupby("doc_id").cumcount() + 1
    
	# Rename the "processed_sentences" column for clarity
	expanded_df.rename(columns={"processed_sentences": "sentence"}, inplace=True)
	
	if impostors:
		expanded_df = expanded_df[['corpus', 'doc_id', 'impostor_id', 'chunk_id', 'author', 'texttype', 'sentence', 'parascore_free']]
	else:
		expanded_df = expanded_df[['corpus', 'doc_id', 'chunk_id', 'author', 'texttype', 'sentence']]
        
	return expanded_df