import json
import os
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

if __name__ == "__main__":
   # Set up logging
   logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
   logger = logging.getLogger(__name__)
   logger.addHandler(logging.StreamHandler())
   logger.setLevel(logging.INFO)
   
   file = 'reformatted_OlympianBench.json' # Change this to the path of your dataset file
   output_location = 'verified_OlympiadBench.json'
   with open(file, 'r', encoding='utf-8') as f:
      seed_dataset = json.load(f)
   logger.info(f"Loaded {len(seed_dataset)} solved samples from {file}")

   # Define LLM
   reason_model = OpenAIModel(
      model_type=ModelType.GPT_4O_MINI,
      model_config_dict={
         "temperature": 0.2,
      }
   )
   reason_agent = ChatAgent(
      model=reason_model,
      system_message=CODE_VERIFICATION_PROMPT
   )

   verification_summary = {
      'total_samples': len(seed_dataset),
      'successful': 0,
      'failed': 0
   }

   sample_count=0
   failed_samples_ids=[]
   outputs = []

   # Start Verification
   for i, sample in enumerate(seed_dataset):
      sample_count += 1

      reason_agent.reset()
      logger.info(f'Verifying code for Question {i} of {file}')
      prompt_message = f"The question is: {sample['question']}. The code is: {sample['rationale']}"
      try:
         raw_response = reason_agent.step(prompt_message, response_format=CodeVerificationResult)
         structured_response = CodeVerificationResult.model_validate(raw_response.msgs[0].parsed)
      except Exception as e:
         logger.info(f'Failed to verify Question {i} of {file} with error {e}')

      # Manage Output
      output = CodeVerificationResult(
            is_valid=structured_response.is_valid,
            issue=structured_response.issue
         )
      
      if output.is_valid:
         verification_summary['successful'] += 1
         outputs.append(output)
      else:
         verification_summary['failed'] += 1
         failed_samples_ids.append(i)

   logger.info(f"==========Seed Dataset Generation Summary==========")
   logger.info(f"Total Samples: {verification_summary['total_samples']}")
   logger.info(f"Successful Code Generations: {verification_summary['successful']}")
   logger.info(f"Failed Generations: {verification_summary['failed']}")
   logger.info(f"Failed Code Sample IDs: {failed_samples_ids}")
   logger.info(f"======================================")
   