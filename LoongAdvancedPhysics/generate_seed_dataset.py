from pipeline import PhysicsCodeGenPipeline
from camel.models import DeepSeekModel, OpenAIModel
from camel.types import ModelType
import os
import json
import logging

from data_processor import DataProcessor

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    DATASET_ORIGIN = input("Enter the dataset origin (SciBench or OlympiadBench): ")

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
    
    output_location = os.path.join(current_path, 'PhysicsDatasets', 'seed_dataset', f'{DATASET_ORIGIN}.json')

    all_problem_ids = [str(sample['id']).strip() for sample in dataset]

    if not os.path.isfile(output_location):
        # os.mknod(output_location)
        with open(output_location, 'w') as f:
            f.write("")
        unsolved_problem_ids = all_problem_ids
    else:
        with open(output_location) as f:
            solved_dataset = json.load(f)

        solved_problem_ids = [sample['sample_id'] for sample in solved_dataset]
        unsolved_problem_ids = [str(problem_id).strip() for problem_id in all_problem_ids if problem_id not in solved_problem_ids]
        print(f"Problems to solve in this run: {unsolved_problem_ids}")

    print(f'No. of problem to solve: {len(unsolved_problem_ids)}')

    pipeline = PhysicsCodeGenPipeline(
        reason_model=reason_model,
        dataset=dataset,
        output_location=output_location,
        problem_ids=unsolved_problem_ids,
        save_right_solution=True,
    )

    pipeline.run()