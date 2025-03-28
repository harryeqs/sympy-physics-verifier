import json
import os
from camel.datasets import DataPoint

def reformat_dataset(code_file, raw_file, output_file):
    """
    Reformat the dataset to include only the necessary fields.
    """
    with open(code_file, 'r', encoding='utf-8') as f:
        code_data = json.load(f)

    with open(raw_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # change 'id' to 'sample_id' if running for SciBench
    raw_data_dict = {str(item['id']).strip(): item for item in raw_data}

    # Create a new list to hold the reformatted data
    reformatted_data = []
    for code_item in code_data:
        value = code_item['verification_result']['code_output']
        # Check if the value is a number
        try:
            # If it's a number, convert it to float
            numeric_value = float(value)
            value = f"{numeric_value:.2g}"  # Format to 2 significant figures
        except ValueError:
            # If it's not a number, skip this item
            pass
        reformatted_item = DataPoint(
            question=raw_data_dict[code_item['sample_id']]['question'],
            rationale=code_item['response']['code'],
            final_answer=f"{value} {code_item['response']['unit']}",    
            metadata=code_item['metadata']
        )

        # Add the reformatted item to the list
        reformatted_data.append(reformatted_item.model_dump())

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(reformatted_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Define file paths
    code_file = 'PhysicsDatasets/seed_dataset/OlympiadBench.json'
    raw_file = 'PhysicsDatasets/OlympiadBench/OE_TO_physics_en_COMP.json'
    output_file = 'reformatted_OlympianBench.json'

    # Reformat the dataset
    reformat_dataset(code_file, raw_file, output_file)