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
Task: Solve the given Physics problem using symbolic computation with Sympy and return the response in a JSON format following the specified ResponseFormat.

ResponseFormat Structure:
{
    "reasoning": <explanation as a string>,
    "code": <complete Sympy code as a string>,
    "unit": <unit as a string or None>
}

Instructions:
1. **Import Libraries:**
   - Begin by importing Sympy (e.g., `import sympy as sp`).

2. **Define Symbols and Constants:**
   - Define all the necessary symbolic variables (e.g., `x, y, t`) and any physical constants that are required for the problem.

3. **Set Up the Problem:**
   - Write the Sympy code to set up the equations that describe the Physics problem.
   - Include comments in your code that explain each step clearly.

4. **Solve the Problem:**
   - Use appropriate Sympy functions to solve the equations symbolically.
   - Ensure that all the steps necessary to reach the solution are included.

5. **Final Result Assignment:**
   - The very last line of your code must assign the final computed result to a variable named `result`:
     ```python
     result = <computed_value>
     ```
   - This is mandatory because the output will be extracted and compared with the ground truth.

6. **Reasoning Explanation:**
   - In a plain text explanation (assigned to the "reasoning" field in the JSON output), provide a clear and concise description of the solution steps.
   - Include any relevant details about the physics concepts and units used. If no specific unit applies, set `"unit"` to `None`.

7. **Output JSON Structure:**
   - Ensure your final answer is a valid JSON object with three keys: `"reasoning"`, `"code"`, and `"unit"`.
   - Follow the structure exactly, so that it can be automatically parsed and validated.

Example Code Template:
-----------------------------------------------------------
import sympy as sp

# Step 1: Define symbols and physical constants
x, y, t = sp.symbols('x y t')
g = sp.symbols('g')  # gravitational constant, if applicable

# Step 2: Set up the Physics problem (e.g., equations of motion)
# [Insert problem-specific equations and logic here]

# Step 3: Solve the equations symbolically
# final_result = sp.solve([...], ...)

# Step 4: Explanation of the approach:
reasoning = "Step 1: Imported sympy and defined symbols. Step 2: Set up the equation based on Newton's laws. Step 3: Solved the equation symbolically to compute the required physical quantity. Units (if any) are specified accordingly."

# Step 5: Compute the final result and assign it to 'result'
result = final_result  # final computed value

-----------------------------------------------------------
Return the output as a JSON object with keys:
   - "reasoning": Detailed explanation of the steps as a string.
   - "code": The complete Sympy code as a string.
   - "unit": A string representing the unit (e.g., "m", "s", "kg") if applicable, otherwise None.
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
         problem_ids: Union[List[int], None] = None,
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
            data_id = str(data['id']).strip()
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

      self.failed_samples_ids = []
      
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
         sample_id = str(sample['id']).strip()
         question = sample['question']
         gt_answer = sample['gt_answer']
         unit = sample['unit']

         full_answer = AnswerFormat(gt_answer=gt_answer, unit=unit) # Create the full answer format including both the numerical answer and unit
         
         self.reason_agent.reset()
         try:
            raw_response = self.reason_agent.step(question, response_format=ResponseFormat)
            structured_response = ResponseFormat.model_validate(raw_response.msgs[0].parsed)
         except Exception as e:
            logger.error(f'Error occurred while generating response for Question {sample_id}: {str(e)}')
            self.failed_samples_ids.append(sample_id)
            continue

         logger.info(f'==========Verifying Question {sample_id}==========')
         verification_outcome = self.verify(structured_response, full_answer)
         logger.info(f'Verification Outcome: Result Match: {verification_outcome.result_match}, Unit Match: {verification_outcome.unit_match}')
         
         if verification_outcome.result_match and verification_outcome.unit_match:
            self.generation_summary['successful_generations'] += 1
         else:
            self.generation_summary['failed_generations'] += 1

         output = OutputFormat(
            sample_id=sample_id,
            response=structured_response,
            answer=full_answer,
            verification_result=verification_outcome,
            metadata=sample['metadata']
         )

         
         if not verification_outcome.result_match or not verification_outcome.unit_match:
            self.failed_samples_ids.append(sample_id)
            if self.save_right_solution:
               continue

         outputs.append(output.model_dump())

         with open(self.output_location, 'w') as f:
            json.dump(outputs, f, indent=4)
      
      logger.info(f"==========Seed Dataset Generation Summary==========")
      logger.info(f"Total Samples: {self.generation_summary['total_samples']}")
      logger.info(f"Successful Generations: {self.generation_summary['successful_generations']}")
      logger.info(f"Failed Generations: {self.generation_summary['failed_generations']}")
      logger.info(f"Failed Sample IDs: {self.failed_samples_ids}")
      logger.info(f"======================================")
