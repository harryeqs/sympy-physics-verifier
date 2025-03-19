import json
import random
import time
import os
from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from typing import List, Dict, Union, Literal
from verifier import PhysicsVerifier, logger
from models import ResponseFormat, AnswerFormat, VerificationResult, OutputFormat

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  

REASON_AGENT_PROMPT = """
Task: Solve the given Physics problem using symbolic computation with Sympy and return the response in a structured JSON format defined by the following ResponseFormat:
    class ResponseFormat(BaseModel):
       reasoning: str
       code: str
       unit: Union[str, None] = None

Instructions:
1. The primary goal is to solve a Physics problem using symbolic computation. Your solution should involve setting up equations, solving them symbolically, and computing the desired physical quantity.
2. Begin by importing the necessary libraries from Sympy.
3. Clearly define all symbolic variables and physical constants required for the problem.
4. Write the Sympy code that sets up and solves the Physics problem.
5. Ensure that the very last line of the code assigns the final computed result to a variable with the format:
       result = <computed_value>
   This is mandatory since the result will be extracted and compared with the ground truth.
6. Prepare a plain text explanation of the solution steps and reasoning, assigning it to the "reasoning" field in the response.
7. If the Physics problem involves any units (e.g., meters, seconds, kilograms, etc.), specify the unit in the explanation or assign it to the "unit" field. If no unit is applicable, set the unit field to None.
8. Return the complete response in a JSON object with the keys:
       - "reasoning": containing the explanation as a string,
       - "code": containing the complete Sympy code as a string,
       - "unit": containing the appropriate unit as a string (or None if not applicable).

Example structure of the code output (as a string):
-----------------------------------------------------------
import sympy as sp

# Define symbols, physical constants, and variables
x, y, t = sp.symbols('x y t')
g = sp.symbols('g')  # gravitational constant, for instance

# [Your problem-specific code here to set up and solve the Physics problem]

# Compute the final result
final_result = ...  # your computation

# Explanation of the approach in plain text
reasoning = "Step 1: ... Step 2: ... (include details of the physics concepts used and any relevant unit information)"

# Final output assignment (must be the last line)
result = final_result
-----------------------------------------------------------

Ensure that your response strictly follows the ResponseFormat structure:
{
    "reasoning": <your explanation as a string>,
    "code": <your complete Sympy code as a string>,
    "unit": <unit as a string or None>
}
"""

class PhysicsCodeGenPipeline():
   """
   A pipeline for generating physics solutions using symbolic computation with Python's Sympy library.
   """
   def __init__(
         self, 
         reason_model: BaseModelBackend, 
         dataset: List[Dict], 
         output_location: str, 
         num: Union[int, None] = None,
         sample: bool = False,
         problem_ids: Union[int, None] = None,
         save_right_solution: bool = False,
         ):
      """
      Initialize the pipeline with the reason model and the dataset.

      Args:
          reason_model (BaseModelBackend): The model used for reasoning and code generation.
          dataset (dict): The dataset containing physics problems and solutions.
          output_location: The file path for the output.
          num: number of samples to generate.
          sample: wether or not randomly sample from the dataset.
          problem_ids: the problem ids for the problems to run.
          save_right_solution: wether or not to only save correct llm solutions.
      """
      # Set limit
      if problem_ids is not None:
         self.dataset = []
         for data in dataset:
            data_id = data['id'].strip()
            if data_id in problem_ids:
               self.dataset.append(data)
      else:
         if num is not None:
            if sample:
               self.dataset = random.sample(dataset, num)
            else:
               self.dataset = dataset[:num]
         else:
            self.dataset = dataset

      # Initialize the reasoning agent
      self.reason_agent = ChatAgent(
         model=reason_model,
         system_message=REASON_AGENT_PROMPT
      )
      
      self.verifier = PhysicsVerifier()
      self.output_location = output_location
      self.save_right_solution = save_right_solution

      self.generation_summary = {
         'total_samples': len(self.dataset),
         'successful_generations': 0,
         'failed_generations': 0
      }
      
   def verify(self, response: ResponseFormat, answer: AnswerFormat) -> VerificationResult:
      """
      A method to verify the correctness of the generated code using PythonVerifier.

      Args:
          response (ResponseFormat): The response format containing the reasoning and code sections.
          gt_answer (str): The ground truth answer to compare against.
      """
      return self.verifier.verify(response, answer)
      

   def run(self):
      """
      Run the pipeline on the dataset and sequentially update the JSON array output.
      """
      # Load existing output if it exists, otherwise initialize an empty list.
      if os.path.exists(self.output_location):
         try:
            with open(self.output_location, 'r') as f:
               outputs = json.load(f)
         except json.JSONDecodeError:
            outputs = []
      else:
         outputs = []

      print(len(self.dataset))
      for sample in self.dataset:
         sample_id = sample['id']
         question = sample['question']
         gt_answer = sample['gt_answer']
         unit = sample['unit']

         full_answer = AnswerFormat(gt_answer=gt_answer, unit=unit) # Create the full answer format including both the numerical answer and unit
         
         raw_response = self.reason_agent.step(question, response_format=ResponseFormat)
         structured_response = ResponseFormat.model_validate(raw_response.msgs[0].parsed)

         logger.info(f'=====Verifying Question {sample_id}=====')
         verification_outcome = self.verify(structured_response, full_answer)
         logger.info(f'Verification Outcome: Result Match: {verification_outcome.result_match}, Unit Match: {verification_outcome.unit_match}')
         
         if verification_outcome.result_match and verification_outcome.unit_match:
            self.generation_summary['successful_generations'] += 1
         else:
            self.generation_summary['failed_generations'] += 1

         output = OutputFormat(
            sample_id=str(sample_id),
            response=structured_response,
            answer=full_answer,
            verification_result=verification_outcome,
            metadata=sample['metadata']
         )

         if self.save_right_solution:
            if not verification_outcome.result_match and verification_outcome.unit_match:
               continue

         outputs.append(output.model_dump())

         with open(self.output_location, 'w') as f:
            json.dump(outputs, f, indent=4)
      
      logger.info(f"Generation Summary: {self.generation_summary}")