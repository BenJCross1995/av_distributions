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
        
def read_completed_excel_result(
    excel_path: str | Path,
    *,
    required: tuple[str, ...] = ("docs", "known", "unknown", "metadata", "no_context", "LLR"),
    engine: str = "openpyxl",
) -> dict[str, pd.DataFrame | None]:
    """
    Read an Excel workbook and return a dict of DataFrames for the requested sheets.
    Matching is case-insensitive and ignores spaces/underscores. Missing sheets -> None.
    """
    p = Path(excel_path)
    if not p.exists():
        raise FileNotFoundError(p)

    # read all sheets once; sheet_name=None → dict of DataFrames keyed by sheet name
    all_sheets: dict[str, pd.DataFrame] = pd.read_excel(p, sheet_name=None, engine=engine)

    def norm(s: str) -> str:
        return s.strip().lower().replace(" ", "_")

    # normalized lookup of the actual workbook sheets
    lookup = {norm(name): df for name, df in all_sheets.items()}

    # return only the requested logical names (normalized)
    out: dict[str, pd.DataFrame | None] = {}
    for logical in required:
        out[logical] = lookup.get(norm(logical))
    return out
