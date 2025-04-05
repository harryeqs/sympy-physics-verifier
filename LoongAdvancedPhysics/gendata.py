from pipeline import PhysicsCodeGenPipeline
from cot_pipeline import CoT_PhysicsCodeGenPipeline
from camel.models import DeepSeekModel, OpenAIModel
from camel.types import ModelType
from typing import Dict, Literal
import os
import json
import logging
import argparse
import json
import argparse
from tqdm import tqdm
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent
from camel.datagen.self_instruct import SelfInstructPipeline
from camel.datagen.self_instruct.filter import RougeSimilarityFilter
import os
from datasets import load_dataset
import random

from data_processor import DataProcessor


def gendata(input_path, target_num, expand_ratio, output_path, human_machine_ratio, agent, filter_config):
    # Load seed data
    with open(input_path, "r") as f:
        seed_data = [json.loads(line) for line in f]

    target_num_instructions = (
        target_num if target_num is not None 
        else int(len(seed_data) * expand_ratio)
    )

    if output_path is None:
        # Extract directory and filename from input_path
        input_dir = os.path.dirname(input_path)
        input_filename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(
            input_dir, 
            f"{input_filename}_self_{target_num_instructions}.json"
        )

    # Initialize the pipeline
    pipeline = SelfInstructPipeline(
        agent=agent,
        seed=input_path,
        num_machine_instructions=target_num_instructions,
        data_output_path=output_path,
        human_to_machine_ratio=human_machine_ratio,
        filter_config=filter_config,
    )

    # Generate machine instructions with progress tracking
    with tqdm(
        total=pipeline.num_machine_instructions,
        desc="Generating instructions"
    ) as pbar:
        while len(pipeline.machine_tasks) < pipeline.num_machine_instructions:
            prompt, instruction = pipeline.generate_machine_instruction()
            
            # Update RougeSimilarityFilter with existing instructions
            existings = [
                task["instruction"] 
                for task in pipeline.human_tasks + pipeline.machine_tasks
            ]
            for filter_instance in pipeline.instruction_filter.filters:
                if isinstance(filter_instance, RougeSimilarityFilter):
                    filter_instance.existing_instructions = existings
            
            # Validate instruction and add to the pipeline if accepted
            if pipeline.instruction_filter.filter(prompt, instruction):
                pipeline.machine_tasks.append({
                    "id": f"machine_task_{len(pipeline.machine_tasks) + 1}",
                    "instruction": instruction,
                })
                pbar.update(1)
    # Construct the final dataset
    pipeline.construct_data()


parser = argparse.ArgumentParser()
parser.add_argument("--dataname_or_path", type=str, required=True, help="Seed data path")
parser.add_argument("--output_path", type=str, required=False, help="Output generated data path")
parser.add_argument("--expand_ratio", type=float, default=1.0, help="Expand ratio of the seed data.")
parser.add_argument("--target_num", type=int, default=None, help="Fixed number of generated instructions (overrides expand_ratio).")
parser.add_argument("--HM_ratio", type=int, nargs=2, default=[6, 2], help="Human-to-machine sample ratio, e.g., 6 2.")

args = parser.parse_args()

if __name__ == "__main__":


    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        )
    agent = ChatAgent(model=model)


    filter_config = {
    "length": {},
    "keyword": {},
    "punctuation": {},
    "non_english": {},
    "rouge_similarity": {"threshold": 0.7},
    }


###############################################################################################################################################################################
    # Load seed data
    if args.dataname_or_path == "scibench": 
        dataset = load_dataset("xw27/scibench")["train"]
###############################################################################################################################################################################


###############################################################################################################################################################################
    # split data 
    instructions = [example["problem_text"] for example in dataset]
    random.shuffle(instructions)

    batch_size = 30
    folder = "/home/wiss/liu/loong/sdfs/sympy-physics-verifier/LoongAdvancedPhysics/PhysicsDatasets/gendata/seeddatasplit/"
    output_folder = folder + "{}_seed1".format(args.dataname_or_path)
    output_path_list = []
    # generate batches 
    num_batches = (len(instructions) + batch_size - 1) // batch_size  # 计算总批次数
    for i in range(num_batches):
        batch_data = instructions[i * batch_size: (i + 1) * batch_size]
        output_path = os.path.join(folder, f"{args.dataname_or_path}_seeddata{i + 1}.jsonl")
        output_path_list.append(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            for instruction in batch_data:
                json_obj = {"instruction": instruction}
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

        print(f"Saved {len(batch_data)} samples to {output_path}")

###############################################################################################################################################################################

###############################################################################################################################################################################
    # gen data 
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        )
    agent = ChatAgent(model=model)

    filter_config = {
        "length": {},
        "keyword": {},
        "punctuation": {},
        "non_english": {},
        "rouge_similarity": {"threshold": 0.7},
    }

    # agrs
    folder = "/home/wiss/liu/loong/sdfs/sympy-physics-verifier/LoongAdvancedPhysics/PhysicsDatasets/gendata/"
    human_machine_ratio = tuple([6, 2])

    gendata_path_list = []
    for i in range(num_batches): 
        output_path = folder + args.dataname_or_path + "_{}.json".format(i)
        input_path = output_path_list[i]
        gendata_path_list.append(output_path)
        gendata(input_path, args.target_num, args.expand_ratio, args.output_path, tuple(args.human_machine_ratio), agent, filter_config)
###############################################################################################################################################################################
    

###############################################################################################################################################################################
    # merge gen data 
    def load_json(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    datai = [] 
    for i in gendata_path_list: 
        datai = load_json(i) + datai
    merged_data = datai

    merged_output_path = "/home/wiss/liu/loong/sdfs/sympy-physics-verifier/LoongAdvancedPhysics/PhysicsDatasets/gendata/scibench_merge.json"  # 合并后的输出路径

    # renumber 
    for index, entry in enumerate(merged_data, start=1):
        entry["id"] = f"machine_task_{index}"
    # save 
    with open(merged_output_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"Merged data with unique IDs saved to {merged_output_path}")
###############################################################################################################################################################################

