from pipeline import PhysicsCodeGenPipeline
from camel.models import DeepSeekModel, OpenAIModel
from camel.types import ModelType
from typing import Dict, Literal
import os
import json
import logging
import argparse

from data_processor import DataProcessor

DATASET_ORIGIN = 'SciBench'

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    current_path = os.path.dirname(os.path.abspath(__file__))

    physics_data_processor = DataProcessor(dataset_origin=DATASET_ORIGIN, base_dir=current_path)
    dataset = physics_data_processor.preprocess_dataset()

    reason_model = OpenAIModel(
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict={
            "temperature": 0.2,
        }
    )

    # reason_model = DeepSeekModel(
    #     model_type=ModelType.DEEPSEEK_REASONER,
    #     model_config_dict={
    #         "temperature": 0.2,
    #     }
    # )
    
    output_location = os.path.join(current_path, 'seed_dataset', DATASET_ORIGIN)

    all_problem_ids = [sample['id'] for sample in dataset]

    if not os.path.isfile(output_location):
        os.mknod(output_location)
        unsolved_problem_ids = all_problem_ids
    else:
        with open(output_location) as f:
            solved_dataset = json.load(f)

        solved_problem_ids = [sample['id'] for sample in solved_dataset]
        unsolved_problem_ids = [problem_id for problem_id in all_problem_ids if problem_id not in solved_problem_ids]

    pipeline = PhysicsCodeGenPipeline(
        reason_model=reason_model,
        dataset=dataset,
        output_location=output_location,
        problem_ids=unsolved_problem_ids,
    )

    pipeline.run()