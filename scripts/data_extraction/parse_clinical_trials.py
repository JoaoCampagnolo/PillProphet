#!/usr/bin/env python
"""
ClinicalTrials.gov XML Parser

This script parses all XML files in a given directory containing ClinicalTrials.gov data,
extracts relevant information from each file, and saves the data to a CSV file.

Usage:
    python parse_clinical_trials.py /path/to/xml/files

The script will:
- Recursively search for XML files in the provided directory.
- Use multiprocessing to process files in parallel.
- Display a progress bar during processing.
- Save the extracted data to 'clinical_trials_data.csv' in the current directory.

Dependencies:
- Python 3.x
- xml.etree.ElementTree (standard library)
- csv (standard library)
- multiprocessing (standard library)
- tqdm (install via 'pip install tqdm')

Example:
    python parse_clinical_trials.py C:\ClinicalTrials\XMLFiles

Author:
    João Campagnolo

Date:
    2024-10-27

"""

import os
from lxml import etree
import pandas as pd
from tqdm import tqdm

def flatten_xml(element, parent_key='', sep='_'):
    items = {}
    for child in element:
        if len(child):
            new_key = f"{parent_key}{sep}{child.tag}" if parent_key else child.tag
            items.update(flatten_xml(child, new_key, sep=sep))
        else:
            new_key = f"{parent_key}{sep}{child.tag}" if parent_key else child.tag
            text = child.text.strip() if child.text else ''
            if new_key in items:
                if isinstance(items[new_key], list):
                    items[new_key].append(text)
                else:
                    items[new_key] = [items[new_key], text]
            else:
                items[new_key] = text
    return items

def parse_xml_files(xml_dir):
    data = []
    fieldnames = set()
    xml_file_paths = []

    for root_dir, dirs, files in os.walk(xml_dir):
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(root_dir, file)
                xml_file_paths.append(file_path)

    print(f"Total XML files to process: {len(xml_file_paths)}")

    for file_path in tqdm(xml_file_paths, desc="Processing XML files", unit="file"):
        try:
            tree = etree.parse(file_path)
            root = tree.getroot()
            record = flatten_xml(root)
            data.append(record)
            fieldnames.update(record.keys())
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    return data, list(fieldnames)

def create_dataframe(data, fieldnames):
    df = pd.DataFrame(data)
    for field in fieldnames:
        if field not in df.columns:
            df[field] = None
    return df

def save_dataframe_to_csv(df, csv_file):
    df.to_csv(csv_file, index=False)
    print(f"Data has been saved to '{csv_file}'.")

def main():
    xml_dir = input('/path/to/your/xml/files\n')  # Replace with your XML directory path
    csv_file = 'clinical_trials_data.csv'  # Output CSV file

    data, fieldnames = parse_xml_files(xml_dir)
    df = create_dataframe(data, fieldnames)
    save_dataframe_to_csv(df, csv_file)

if __name__ == '__main__':
    main()

