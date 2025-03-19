import os
import json
from typing import Dict, Literal

class DataProcessor():
    def __init__(
            self,
            dataset_origin: Literal["OlympiadBench", "SciBench"],
            base_dir: str,
        ):
        self.dataset_origin = dataset_origin
        self.base_dir = base_dir

    def preprocess_data(self, data: Dict):

        if self.dataset_origin == "OlympiadBench":
            # Create a new dictionary with the selected keys.
            processed = {
                "id": data.get("id"),
                "question": data.get("context") + '\n' + data.get("question"),
                "gt_answer": data.get("final_answer")[0],
                "unit": data.get("unit"),
                "metadata": {}
            }
            
            # Define keys that should be moved out of metadata.
            keys_to_exclude = {"id", "question", "context", "final_answer", \
                            "image_1", "image_2", "image_3", "image_4", "image_5"}
        elif self.dataset_origin == "SciBench":
            processed = {
                "id": data.get("problemid"),
                "question": data.get("problem_text"),
                "gt_answer": data.get("answer_number"),
                "unit": data.get("unit"),
                "metadata": {}
            }
            
            keys_to_exclude = {"problemid", "problem_text", "answer_number"}
        
        # Put the remaining keys into metadata.
        for key, value in data.items():
            if key not in keys_to_exclude:
                processed["metadata"][key] = value
                
        return processed

    def preprocess_dataset(self):

        if self.dataset_origin == "OlympiadBench":
            dataset_path = os.path.join(self.base_dir, "PhysicsDatasets/OlympiadBench/")
            files = ['OE_TO_physics_en_COMP.json']
        elif self.dataset_origin == "SciBench":
            dataset_path = os.path.join(self.base_dir, "PhysicsDatasets/SciBench/")
            # files = ['class_sol.json', 'fund_sol.json', 'thermo_sol.json']
            files = ['full_question.json']
        else:
            raise ValueError("Invalid dataset name. Please choose either 'OlympiadBench' or 'SciBench'.")

        dataset = []
        for file in files:
            with open(os.path.join(dataset_path, file)) as f:
                d = json.load(f)
                f.close()
            dataset += d

        processed_dataset = []
        for data in dataset:
            processed_data = self.preprocess_data(data)
            processed_dataset.append(processed_data)

        return processed_dataset