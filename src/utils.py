import re
import pandas as pd

def create_temp_doc_id(input_text):
    """Create a new doc id by preprocessing the current id"""
    
    # Extract everything between the brackets
    match = re.search(r'\[(.*?)\]', input_text)
    
    if match:
        extracted_text = match.group(1)
        # Replace all punctuation and spaces with "_"
        cleaned_text = re.sub(r'[^\w]', '_', extracted_text)
        # Replace multiple underscores with a single "_"
        final_text = re.sub(r'_{2,}', '_', cleaned_text)
        return final_text.lower()
        
    return None

def apply_temp_doc_id(df):
    """Apply the doc id function on the dataframe"""
    
    # Rename doc_id to orig_doc_id first    
    df.rename(columns={'doc_id': 'orig_doc_id'}, inplace=True)

    # Create the new doc_id column directly
    df['doc_id'] = df['orig_doc_id'].apply(create_temp_doc_id)

    # df.drop("orig_doc_id", axis=1, inplace=True)
    
    # Move the new doc_id column to the front
    cols = ['doc_id', 'orig_doc_id'] + [col for col in df.columns if col not in ['doc_id', 'orig_doc_id', 'text']] + ['text']

    df = df[cols]

    return df