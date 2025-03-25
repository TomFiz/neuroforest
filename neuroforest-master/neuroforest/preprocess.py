import sys
import os
sys.path.append('C:/Users/kupec/OneDrive/Desktop/neuroforest-main/neuroforest-master')
from typing import List
from pathlib import Path
from tqdm import tqdm

import json
import re

def preprocess_json_string(json_string):
    # Regular expression to match the problematic items in "Positions"
    pattern_1 = re.compile(r'\[({.*?}),\d+,\d+,\d+\.\d+\]')
    pattern_2 = re.compile(r'\[({.*?}),\d+,\d+,\d+]')
    pattern_3 = re.compile(r'\[({.*?}),\d+,\d+E-?\d+,\d+\.\d+]')

    def replace_function(match):
        # Extract the JSON object and the last float value
        json_object = match.group(1)
        float_value = match.group(0).split(',')[-1].strip(']')
        return f'[{json_object},{float_value}]'
    
    # Replace the problematic items
    corrected_string = pattern_1.sub(replace_function, json_string)
    corrected_string = pattern_2.sub(replace_function, corrected_string)
    corrected_string = pattern_3.sub(replace_function, corrected_string)

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
        print(f"\nError decoding JSON: {e}")
        char_index = int(str(e).split(' ')[-1][:-1])
        error_str = corrected_string[char_index-30:char_index+30]
        print(f"\nError occurred around there : {error_str}")
        data = None
    
    if data is not None:
        with open(output_path, 'w+') as file:
            json.dump(data, file, indent=4)
        # print("Successfully processed and saved JSON data")
    else:
        print(f"\nFailed to process JSON data for file: {file_path}")


def check(year : str, file_name : str):
    file_path = Path(f"C:/Users/kupec/OneDrive/Desktop/neuroforest-main/data_{year}/trajectories/{file_name}.json")
    output_path = Path(f"{file_path.parent.parent}/trajectories_processed/{file_name}.json")
    preprocess_json(file_path, output_path)


folder = Path("C:/Users/kupec/OneDrive/Desktop/neuroforest-main/data_2024/trajectories")
for file in tqdm(folder.glob("*.json")):
    output_path = Path(f"{file.parent.parent}/trajectories_processed/{file.stem}.json")
    if not os.path.exists(output_path):
        preprocess_json(file, output_path)

# check("2022", "Angela_patchy0")