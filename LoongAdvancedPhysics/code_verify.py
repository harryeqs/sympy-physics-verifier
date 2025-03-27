import json
import random
import time
import os
from typing import List, Dict, Union, Literal
from models import CodeVerificationResult
from camel.agents import ChatAgent
from camel.models import OpenAIModel
from camel.types import ModelType

import logging

# 1. Understand the Problem: Review the physics problem statement carefully. Identify the key physical principles, equations, and variables involved.
CODE_VERIFICATION_PROMPT="""
Task: Verify whether the given code correctly solves a physics problem. Your goal is to ensure the code authentically calculates the solution based on the problem’s requirements, without cheating (e.g., hardcoding the answer, directly assigning the solution to a variable based on a provided answer, or skipping essential steps). 
Follow these steps:
1. Analyze the Code: Examine the provided code line by line. Check if it:
    Uses appropriate physics formulas or methods to compute the solution.
    Avoids directly setting the answer variable to a predefined value or expression (e.g., answer = 42) unless explicitly justified by the problem.
    Implements the necessary calculations rather than relying on external data or shortcuts unrelated to the problem’s context.
2. Determine Validity: Decide if the code is valid. Output a JSON object with keys "is_valid" (A boolean, True if the code is valid, False if it is not) and "issue" (A string describing the specific problem if is_valid is false, or null if is_valid is true).
"""

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# rRad seed datasets
current_path = os.path.dirname(os.path.abspath(__file__))
output_location = os.path.join(current_path, 'PhysicsDatasets', 'seed_dataset', 'code_verification.json')
dataset_path = os.path.join(current_path, "PhysicsDatasets/seed_dataset/")
files = ['OlympiadBench.json', 'SciBench.json']
dataset=[]
for file in files:
   if not os.path.exists(os.path.join(dataset_path, file)):
         raise FileNotFoundError(f"Dataset file not found: {os.path.join(dataset_path, file)}")
   with open(os.path.join(dataset_path, file), 'r', encoding='UTF-8') as f:
         d = json.load(f)
         f.close()
   dataset += d

# Define LLM
reason_model = OpenAIModel(
   model_type=ModelType.O3_MINI,
   model_config_dict={
      "temperature": 0.2,
   }
)
reason_agent = ChatAgent(
   model=reason_model,
   system_message=CODE_VERIFICATION_PROMPT
)

verification_summary = {
   'total_samples': len(dataset),
   'successful': 0,
   'failed': 0
}
sample_count=0
failed_samples_ids=[]

# Start Verification
for sample in dataset:
   sample_count += 1
   sample_id = str(sample['sample_id']).strip()
   code = sample['response']
   code = code['code']

   reason_agent.reset()
   prompt_message = code
   logger.info(f'==========Verifying Code for Question {sample_id} ({sample_count}/{len(dataset)})==========')
   try:
      raw_response = reason_agent.step(prompt_message, response_format=CodeVerificationResult)
      structured_response = CodeVerificationResult.model_validate(raw_response.msgs[0].parsed)
   except Exception as e:
      logger.info(f'Failed to verify Question {sample_id} with error {e}')

   # Manage Output
   output = CodeVerificationResult(
         sample_id=sample_id,
         is_valid=structured_response.is_valid,
         issue=structured_response.issue
      )
   if output.is_valid:
      verification_summary['successful'] += 1
   else:
      verification_summary['failed'] += 1
      failed_samples_ids.append(sample_id)

logger.info(f"==========Seed Dataset Generation Summary==========")
logger.info(f"Total Samples: {verification_summary['total_samples']}")
logger.info(f"Successful Generations: {verification_summary['successful_generations']}")
logger.info(f"Failed Generations: {verification_summary['failed_generations']}")
logger.info(f"Failed Sample IDs: {failed_samples_ids}")
logger.info(f"======================================")