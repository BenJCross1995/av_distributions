# -*- coding: utf-8 -*-
import json
import os
import uuid
import pyreadr
import pandas as pd

from datetime import datetime
from pathlib import Path
from typing import Union

def read_jsonl(file_path):
    """Reads a JSONL file and converts it into a pandas DataFrame."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parsed_line = json.loads(line)
            if isinstance(parsed_line, list) and len(parsed_line) == 1:
                data.append(parsed_line[0])
            else:
                data.append(parsed_line)
                
    return pd.DataFrame(data)

def read_rds(file_path):
    """Reads an RDS file and converts it into a pandas DataFrame."""
    objects = pyreadr.read_r(file_path)  # dict-like: {object_name: object}

    # If the file contains a single object, just return that object
    if len(objects) == 1:
        return next(iter(objects.values()))

    # Otherwise, return the whole dict (could be multiple data frames, models, etc.)
    return dict(objects)

def read_xml(location: Union[str, Path]) -> str:
    """Reads an XML file and stores as string"""
    path = Path(location)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    # Read and return the file contents as a single string
    return path.read_text(encoding='utf-8')

def write_jsonl(data, output_file_path):
    """Writes a pandas DataFrame to a JSONL file."""
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for _, row in data.iterrows():
            json.dump(row.to_dict(), file)
            file.write('\n')


def save_error_as_txt(data, folder_path):
    """
    Saves error data to a folder path.
    
    Parameters:
    - data: The pandas DataFrame to save.
    - folder_path: The folder path where the file should be saved.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]  # Generate a unique ID
    filename = f"error_{timestamp}_{unique_id}.txt"
    file_path = os.path.join(folder_path, filename)
    
    with open(file_path, 'w') as file:
        file.write(data)