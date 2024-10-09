import sys
sys.path.append('C:/Users/TomFi/Desktop/Cours/Projet IA/neuroforest-master')
from typing import List
from pathlib import Path
from tqdm import tqdm

import json
import re

def preprocess_json_string(json_string):
    # Regular expression to match the problematic items in "Positions"
    pattern = re.compile(r'\[({.*?}),\d+,\d+,\d+\.\d+\]')
    
    def replace_function(match):
        # Extract the JSON object and the last float value
        json_object = match.group(1)
        float_value = match.group(0).split(',')[-1].strip(']')
        return f'[{json_object},{float_value}]'
    
    # Replace the problematic items
    corrected_string = pattern.sub(replace_function, json_string)
    return corrected_string

def preprocess_json(file_path, output_path):
    with open(file_path, 'r') as file:
        json_string = file.read()
    
    # Preprocess the JSON string to fix the problematic items
    corrected_string = preprocess_json_string(json_string)
    
    try:
        # Attempt to load the corrected JSON string into a dictionary
        data = json.loads(corrected_string)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        data = None
    
    if data is not None:
        with open(output_path, 'w+') as file:
            json.dump(data, file, indent=4)
        # print("Successfully processed and saved JSON data")
    else:
        print(f"Failed to process JSON data for file: {file_path}")


def check(year : str, file_name : str):
    file_path = Path(f"C:/Users/TomFi/Desktop/Cours/Projet IA/data_{year}/trajectories/{file_name}.json")
    output_path = Path(f"{file_path.parent.parent}/trajectories_processed/{file_name}.json")
    preprocess_json(file_path, output_path)
    

folder = Path("C:/Users/TomFi/Desktop/Cours/Projet IA/data_2022/trajectories")
for file in tqdm(folder.glob("*.json")):
    output_path = Path(f"{file.parent.parent}/trajectories_processed/{file.stem}.json")
    preprocess_json(file, output_path)