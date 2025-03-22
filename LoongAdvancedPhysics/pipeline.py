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

# REASON_AGENT_PROMPT = """
# Task: Solve the given physics problem using symbolic computation with Sympy. The generated code must be self-contained, executable, and free of syntax errors. Do not print the result; instead, assign it to the variable `result`.

# **MANDATORY STRUCTURE (Follow this order strictly):**

# 1. **Imports**  
#    - Must include:
#      ```python
#      import sympy as sp
#      ```

# 2. **Symbol and Function Declaration**  
#    - For declaring symbols, use sp.symbols() as follows:
#      - Single symbol:
#        ```python
#        x = sp.symbols('x', positive=True)
#        ```
#      - Multiple symbols:
#        ```python
#        x, y = sp.symbols('x y', real=True)
#        ```
#    - For functions, use sp.Function(), for example:
#      ```python
#      f = sp.Function('f')(x)
#      ```

# 3. **Constant Definitions**  
#    - Use numerical values only when required (e.g., `g = 9.81`). Otherwise, keep expressions symbolic (e.g., `G = sp.symbols('G')`).

# 4. **Equation Setup**  
#    - Always use the sp.Eq(left, right) format for equations.
#    - Example:
#      ```python
#      eq = sp.Eq(F, m*a)
#      ```

# 5. **Solving the Equation**  
#    - For symbolic solutions, use:
#      ```python
#      sol = sp.solve(eq, variable)
#      ```
#    - For numerical solutions, use:
#      ```python
#      sol = sp.nsolve(eq, variable, initial_guess)
#      ```
#    - Ensure that the equation has a valid solution before solving.

# 6. **Result Assignment**  
#    - Assign the final solution to the variable `result` without using any print statements.
#    - Example:
#      ```python
#      result = sol[0] if isinstance(sol, list) else sol
#      ```

# **ERROR PREVENTION CHECKLIST:**
# ✓ All calls to sp.symbols() and sp.Function() must have properly closed string literals.  
# ✓ Ensure that all parentheses and brackets are correctly matched.  
# ✓ Separate keyword parameters with commas correctly.  
# ✓ No print statements should be present; output must be assigned to `result`.  
# ✓ The final output should be a JSON object with keys "code" (executable code as a string) and "unit" (physical unit as a string).

# **Example Template:**
# ```json
# {
#   "code": "import sympy as sp\nx, y = sp.symbols('x y', real=True)\neq = sp.Eq(x**2 - y, 0)\nsol = sp.solve(eq, x)\nresult = sol[0]",
#   "unit": "m"
# }
# ```
# """

REASON_AGENT_PROMPT = """
Task: Write a self-contained Sympy code snippet to solve a physics problem. The code must be executable and free of syntax errors.

Requirements:
- Import Sympy as: `import sympy as sp`
- Declare symbols one by one using sp.symbols (e.g., `x= sp.symbols('x', real=True)`) with properly closed string literals and correct keyword syntax.
- Construct equations using sp.Eq (e.g., `eq = sp.Eq(left, right)`).
- Solve the equation (using sp.solve for symbolic or sp.nsolve for numerical solutions).
- Assign the final answer to the variable `result` (do not use print statements).
- Output a JSON object with keys "code" (the complete code as a string) and "unit" (the physical unit as a string).

Let the model decide on the details while ensuring the code is correct.
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
         max_attempts: int = 3,
         save_right_solution: bool = False,
         ):
      """
      Initialize the pipeline with the reason model and the dataset.

      Args:
          reason_model (BaseModelBackend): The model used for code generation.
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
      self.max_attempts = max_attempts
      
   def verify(self, response: ResponseFormat, answer: AnswerFormat) -> VerificationResult:
      """
      A method to verify the correctness of the generated code using PythonVerifier.

      Args:
          response (ResponseFormat): The response format containing the code and the units.
          gt_answer (str): The ground truth answer to compare against.
      """
      return self.verifier.verify(response, answer)
      
   @staticmethod
   def initialize_output_list(output_location):
      """
      Load existing output if it exists, otherwise initialize an empty list.
      """
      if os.path.exists(output_location):
         try:
            with open(output_location, 'r') as f:
               outputs = json.load(f)
         except json.JSONDecodeError:
            outputs = []
      else:
         outputs = []
      return outputs

   def run(self):
      """
      Run the pipeline on the dataset and sequentially update the JSON array output.
      """
      dataset_origin = os.path.split(self.output_location)[-1].replace('.json', '')
      failed_output_location = os.path.join(*os.path.split(self.output_location)[:-1], f'{dataset_origin}_failed.json')

      outputs = self.initialize_output_list(self.output_location)
      failed_outputs = self.initialize_output_list(failed_output_location)

      sample_count = 0

      for sample in self.dataset:
         sample_count += 1
         sample_id = str(sample['id']).strip()
         question = sample['question']
         gt_answer = sample['gt_answer']
         unit = sample['unit']

         full_answer = AnswerFormat(gt_answer=gt_answer, unit=unit) # Create the full answer format including both the numerical answer and unit
         
         # Reset the reasoning agent on each new question
         self.reason_agent.reset()
         attempts = 0
         feedback = ""
   
         while attempts < self.max_attempts:

            prompt_message = question + (f"\n\nPlease improve the solution based on the following feedback:\n{feedback}" if feedback else "")
            logger.info(f'==========Generating Code for Question {sample_id} ({sample_count}/{len(self.dataset)}): Attempt {attempts + 1}==========')

            try:
               raw_response = self.reason_agent.step(prompt_message, response_format=ResponseFormat)
               structured_response = ResponseFormat.model_validate(raw_response.msgs[0].parsed)
            except Exception as e:
               logger.info(f'Failed to generate or verify Question {sample_id} with error {e}, retrying with attempt {attempts} out of {self.max_attempts}')
               attempts += 1
               if attempts == self.max_attempts:
                  verification_outcome = None
                  break

            logger.info(f'==========Verifying Question {sample_id} ({sample_count}/{len(self.dataset)})==========')
            verification_outcome = self.verify(structured_response, full_answer)
            logger.info(f'Verification Outcome: Result Match: {verification_outcome.result_match}, Unit Match: {verification_outcome.unit_match}')

            if verification_outcome.result_match and verification_outcome.unit_match:
               break # Exit the loop if the response is correct
            else:
               attempts += 1
               feedback = f"""
The code output does not match the expected answer or have errors. Please review the code and try again.
Expected answer:
   - Answer: {gt_answer}
   - Unit: {unit}
Your previous answer: 
   - Code: {structured_response.code}
   - Code Output: {verification_outcome.code_output}
   - Unit: {structured_response.unit}
Verification details: 
   - Result match: {verification_outcome.result_match},
   - Unit match: {verification_outcome.unit_match}
Error message: {verification_outcome.error}
               """
               if sample["solution"]:
                  feedback += f"""
Please refer to this solution when generating the code:
{sample['solution']}
         """   
         
         # Post-attempts handling
         output = OutputFormat(
               sample_id=sample_id,
               response=structured_response,
               answer=full_answer,
               verification_result=verification_outcome,
               metadata=sample['metadata']
            )
         
         if verification_outcome.result_match and verification_outcome.unit_match:
               self.generation_summary['successful_generations'] += 1
         else:
            self.generation_summary['failed_generations'] += 1
            self.failed_samples_ids.append(sample_id)
            failed_outputs.append(output.model_dump())

            with open(failed_output_location, 'w') as f:
               json.dump(failed_outputs, f, indent=4)
            
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
