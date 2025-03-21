from pipeline import PhysicsCodeGenPipeline
from camel.models import DeepSeekModel, OpenAIModel
from camel.types import ModelType
import os
import json
import logging

from data_processor import DataProcessor

def extract_problem_ids(dataset_location):
    """
    extract problem ids from a given dataset stored with json. 
    If the provided path not exist, create an empty file and return []
    """
    if not os.path.isfile(dataset_location):
        with open(dataset_location, 'w') as f:
            f.write("")
        problem_ids = []
    else:
        try:
            with open(dataset_location) as f:
                dataset = json.load(f)
            problem_ids = [sample['sample_id'] for sample in dataset]
        except json.decoder.JSONDecodeError:
            problem_ids = []

    return problem_ids


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    DATASET_ORIGIN = input("Enter the dataset origin (SciBench or OlympiadBench): ")
    RERUN_FAILED_SAMPLES = input("Do you want to rerun failed samples? (y/n): ")

    if RERUN_FAILED_SAMPLES.lower().strip() == 'y' or not RERUN_FAILED_SAMPLES:
        RERUN_FAILED_SAMPLES = True
    elif RERUN_FAILED_SAMPLES.lower().strip() == 'n':
        RERUN_FAILED_SAMPLES = False
    else:
        raise ValueError('Invalid input, please enter "y" or "n".')

    current_path = os.path.dirname(os.path.abspath(__file__))

    physics_data_processor = DataProcessor(dataset_origin=DATASET_ORIGIN, base_dir=current_path)
    dataset = physics_data_processor.preprocess_dataset()

    reason_model = OpenAIModel(
        model_type=ModelType.GPT_4O_MINI,
        # model_config_dict={
        #     "temperature": 0.2,
        # }
    )

    # reason_model = DeepSeekModel(
    #     model_type=ModelType.DEEPSEEK_REASONER,
    #     model_config_dict={
    #         "temperature": 0.2,
    #     }
    # )
    
    output_location = os.path.join(current_path, 'PhysicsDatasets', 'seed_dataset', f'{DATASET_ORIGIN}.json')
    failed_output_location = os.path.join(current_path, 'PhysicsDatasets', 'seed_dataset', f'{DATASET_ORIGIN}_failed.json')

    all_problem_ids = [str(sample['id']).strip() for sample in dataset]

    solved_problem_ids = extract_problem_ids(output_location)
    failed_problem_ids = extract_problem_ids(failed_output_location)

    print(f'No. of solved problems: {len(solved_problem_ids)}')
    unsolved_problem_ids = [str(problem_id).strip() for problem_id in all_problem_ids if problem_id not in solved_problem_ids]

    if not RERUN_FAILED_SAMPLES:
        print(f'No. of failed problems: {len(failed_problem_ids)}')
        unsolved_problem_ids = [str(problem_id).strip() for problem_id in unsolved_problem_ids if problem_id not in failed_problem_ids]
    else:
        print('Rerun failed samples. Reset all failed problems.')
        with open(failed_output_location, 'w') as f:
            f.write("")
    
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