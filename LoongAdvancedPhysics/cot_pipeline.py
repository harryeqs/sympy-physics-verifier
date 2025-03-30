import json
import random
import time
import os
from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from typing import List, Dict, Union, Literal
from verifier import PhysicsVerifier, logger
from llm_verifier import LLMVerifier
from models import ResponseFormat, AnswerFormat, VerificationResult, OutputFormat

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

COT_PROMPT="""
Task: Solve the given physics problem by reasoning step-by-step using Chain of Thought (CoT) to derive the final answer. The focus is on the accuracy and clarity of your reasoning.

Output Format (JSON):
{
    "cot": "<step-by-step reasoning as a string>",
    "gt_answer": "<final answer as a string; should be parsed by sympy or latex; the unit is not includes>",
    "unit": "<unit as a string or None>"
}

Instructions:
1. Chain of Thought Reasoning:
   - Break down the problem into clear, logical steps.
   - Identify known values, variables, and relevant physics formulas.
   - Use LaTeX for symbolic computations throughout the process.
   - Derive the final answer and specify its unit (if applicable).

2. Unit Handling:
   - Maintain unit consistency throughout the reasoning.
   - Specify the unit of the `reasoning_answer` in the `unit` field (e.g., "m/s", "kg*m/s^2", or None for dimensionless).
   - For composite units, explicitly use "*" between components. Use "^(-1)", "^(-2)" for negative exponents. e.g. "J*K^(-1)*mol^(-1)" instead of "J K^-1 mol^-1".

Example:
Problem: "A one horsepower propeller powered by a battery and is used to propel a small boat initially at rest. You have two options:\n\n1. Put the propeller on top of the boat and push on the air with an initial force $F_{1}$\n2. Put the propeller underwater and push on the water with an initial force $F_{2}$.\n\nThe density of water is $997 \\mathrm{~kg} / \\mathrm{m}^{3}$ while the density of air is $1.23 \\mathrm{~kg} / \\mathrm{m}^{3}$. Assume that the force is both cases is dependent upon only the density of the medium, the surface area of the propeller, and the power delivered by the battery. What is $F_{2} / F_{1}$ ? You may assume (unrealistically) the efficiency of the propeller does not change. Round to the nearest tenths."
{
    "cot": "The force exerted on the fluid is roughly proportional to the change in momentum with respect to time:\n$$\nF=\\frac{d p}{d t}=v \\frac{d m}{d t}=v \\frac{d}{d t}(\\rho A x)=\\rho A v^{2}\n$$\n\nIt is kept at a constant power $P=F v$, which can allow us to solve for the speed $v$ of the propellers.\n\n$$\nP=\\rho A v^{3} \\Longrightarrow v=\\left(\\frac{P}{\\rho A}\\right)^{1 / 3}\n$$\n\nso the force is given by:\n\n$$\nF=\\rho A\\left(\\frac{P}{\\rho A}\\right)^{2 / 3} \\Longrightarrow F \\propto \\rho^{1 / 3}\n$$\n\nTherefore:\n\n$$\nF_{2} / F_{1}=(997 / 1.23)^{1 / 3}=9.26 \\text { times }\n$$",
    "cot_answer": "9.26",
    "cot_unit": "J"
}
For a problem yielding a symbolic frequency expression, your response might look like:
{
    "cot": "Step 1: Start with the equation. Step 2: Substitute a = \\frac{i \\hbar}{2m}. Step 3: Derive \\omega = \\frac{\\hbar k^2}{2m}.",
    "cot_answer": "\\omega = \\frac{\\hbar k^2}{2m}",
    "cot_unit": "rad/s"
}
"""

class CoT_PhysicsCodeGenPipeline():
   """
   A pipeline for generating physics solutions using both CoT and symbolic computation with Python's Sympy library.
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
         save_right_solution: bool = True,
         llm_verifier: bool = False
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
      
      self.cot_agent = ChatAgent(
         model=reason_model,
         system_message=COT_PROMPT
      )
      
      if llm_verifier == True:
         self.verifier = LLMVerifier(reason_model=reason_model)
      else:
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
   
   def generate_cot(self, question: str, double: bool=False) -> Union[AnswerFormat, None]:
      """
      Generate CoT response in JSON format for the given question.

      Args:
         question: The physics problem to solve.
         double: Double-check the CoT.
      """
      try:
         raw_response = self.cot_agent.step(question, response_format=AnswerFormat)
         structured_response = AnswerFormat.model_validate(raw_response.msgs[0].parsed)
         if double == True:
            prompt_message = question + (f"\n\nCheck the validity of this solution and give the right solution based on it.\n{structured_response.cot}")
            raw_response = self.reason_agent.step(prompt_message, response_format=AnswerFormat)
            structured_response = AnswerFormat.model_validate(raw_response.msgs[0].parsed)
            print(structured_response.gt_answer)
         return structured_response
      except Exception as e:
         logger.error(f"Failed to parse CoT response: {e}")
         return None

   def verify(self, response: ResponseFormat, answer: AnswerFormat) -> VerificationResult:
      """
      A method to verify the correctness of the generated code using PythonVerifier.

      Args:
          response (ResponseFormat): The response format containing the code and the units.
          gt_answer (AnswerFormat): The CoT generated answer to compare against.
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
      print("Using CoT as Ground Truth")
      dataset_origin = os.path.split(self.output_location)[-1].replace('.json', '')
      failed_output_location = os.path.join(*os.path.split(self.output_location)[:-1], f'{dataset_origin}_failed.json')

      outputs = self.initialize_output_list(self.output_location)
      failed_outputs = self.initialize_output_list(failed_output_location)

      sample_count = 0

      for sample in self.dataset:
         sample_count += 1
         sample_id = str(sample['id']).strip()
         question = sample['question']
         cot_response=self.generate_cot(question)
         if cot_response is None:
                  logger.error(f"CoT response is None")
         else:
            cot = cot_response.cot
            gt_answer = cot_response.gt_answer
            unit = cot_response.unit

         full_answer = AnswerFormat(cot=cot, gt_answer=gt_answer, unit=unit) # Create the full answer format including both the numerical answer and unit
         
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
