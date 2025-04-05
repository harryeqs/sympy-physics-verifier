from pipeline import PhysicsCodeGenPipeline
from cot_pipeline import CoT_PhysicsCodeGenPipeline
from camel.models import DeepSeekModel, OpenAIModel
from camel.types import ModelType
from typing import Dict, Literal
import os
import json
import logging
import argparse

from data_processor import DataProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Which dataset to use: OlympiadBench, SciBench or gendatasci/gendataoly")
parser.add_argument("--num", type=int, required=False, help="Number of samples to generate. If not provided, all samples will be generated. It will be ignored if you have specified problem_ids.")
parser.add_argument("--sample", action='store_true', help='randomly sample from the datasets instead of choosing the first few. It will be ignored if you have specified problem_ids.')
parser.add_argument("--problem_ids", nargs='+', default='', help='specify problem ids to solve. Pass in multiples using "--problem_ids id_1 id_2 ... "')
parser.add_argument("--out_location", type=str, required=False, help="Output directory for the generated samples.")
parser.add_argument("--cot_as_gt", action='store_true', help='Use answers generated from Chain of Thought to compare against.')
parser.add_argument("--llm_verifier", action='store_true', help='Use LLM to verify whether the output matches the ground truth')

args = parser.parse_args()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    current_path = os.path.dirname(os.path.abspath(__file__))

    physics_data_processor = DataProcessor(dataset_origin=args.dataset, base_dir=current_path)
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
    
    output_location = args.out_location if args.out_location else os.path.join(current_path, "output.json")

    # clean the output file if already exists
    with open(output_location, 'w') as f:
        f.write('')

    if args.problem_ids == '':
        problem_ids = None
    else:
        problem_ids = [problem_id.strip() for problem_id in args.problem_ids]

    if args.cot_as_gt:
        pipeline = CoT_PhysicsCodeGenPipeline(
            reason_model=reason_model,
            dataset=dataset,
            output_location = output_location,
            num=args.num,
            sample=args.sample,
            problem_ids=problem_ids,
            llm_verifier=args.llm_verifier
        )
    else:
        pipeline = PhysicsCodeGenPipeline(
            reason_model=reason_model,
            dataset=dataset,
            output_location = output_location,
            num=args.num,
            sample=args.sample,
            problem_ids=problem_ids,
            llm_verifier=args.llm_verifier
        )

    pipeline.run()