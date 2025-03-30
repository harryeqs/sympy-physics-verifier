from models import ResponseFormat, AnswerFormat, VerificationResult, LLMVerificationResult
from typing import Any, Tuple
import sympy as sp
import concurrent.futures
import re
import math
from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from sympy.physics import units
from sympy.physics.units.prefixes import PREFIXES
from verifier import UnitParser

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

VERIFIER_PROMPT="""
Task: Compare a physics solver’s response to a ground truth answer and check if the response is valid.
Both the response and the answer are provided in LaTeX format, each consisting of a numerical or symbolic answer and a unit.

Requirements:
- Focus on the mathematical and physical equivalence of the expressions. 
- Ignoring minor formatting differences such as subscripts (e.g., epsilon_0 vs. epsilon_{0}), trivial notation variations (e.g., Eq(a,1200) vs. 1200) or difference from approximation (e.g. 7.1763 vs. 7.2). 
- Output a JSON object with the following keys:
    result_match: Boolean (True if the numerical or symbolic parts match, False otherwise)
    unit_match: Boolean (True if the units match, False otherwise)
"""

def clean_python_code(raw_code: str) -> str:
    """
    Extracts and cleans Python code from an LLM response.
    
    Parameters:
        llm_response (str): The raw response from an LLM containing Python code
        
    Returns:
        str: Cleaned Python code ready for execution
    """
    # Handle different code formats
    code = raw_code.strip()
    
    # Case 1: Code in markdown blocks with ```python
    if "```" in code:
        parts = code.split("```")
        for i, part in enumerate(parts):
            if part.strip().lower().startswith("python"):
                if i+1 < len(parts):
                    code = parts[i+1].strip()
                    break
    
    # Case 2: Code starts with 'python' without code blocks
    elif code.lower().startswith("python"):
        code = code[len("python"):].strip()
    
    # Case 3: Handle code with escaped newlines \n
    code = code.replace("\\n", "\n")
    
    # Case 4: Remove any surrounding quotes
    if (code.startswith("'") and code.endswith("'")) or (code.startswith('"') and code.endswith('"')):
        code = code[1:-1]

    
    return code

def clean_answer(raw_answer: str) -> str:
    """
    Clean a raw answer string by removing LaTeX formatting.
    
    Parameters:
        raw_answer (str): The raw answer string potentially containing LaTeX formatting
        
    Returns:
        str: The cleaned answer string without LaTeX formatting
    """
    # Remove whitespace
    answer = raw_answer.strip()
    
    # Remove dollar signs that indicate LaTeX math mode
    if answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1].strip()
    
    # Replace LaTeX scientific notation format (e.g., 1 \times 10^{14})
    answer = re.sub(r'([\d.]+)\s*\\times\s*10\^\{(\d+)\}', r'\1e\2', answer)
    
    # Remove \mathrm commands
    answer = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', answer)
    
    # Remove other common LaTeX formatting
    answer = answer.replace('\\', '')
    
    return answer

class LLMVerifier:
    def __init__(self, reason_model: BaseModelBackend, tolerance: float = 1e-2):
        # Set the tolerance for float comparisons.
        self.tolerance = tolerance
        self.unit_parser = UnitParser()
        self.error_msg = None
        self.reason_model=reason_model
        # Define LLM
        self.verifier_agent = ChatAgent(
            model=self.reason_model,
            system_message=VERIFIER_PROMPT
        )
    
    def safe_exec(self, code):
        namespace = {}
        try:
            pattern = r'([\w\d_]+)\.evalf\(\)'
            evalf_vars = set(re.findall(pattern, code))

            for var in evalf_vars:
                # Replace `XXX.evalf()` with a conditional check
                safe_evalf = f'{var} if isinstance({var}, float) else {var}.evalf()'
                code = code.replace(f'{var}.evalf()', safe_evalf)

            try:
                # Compile the code first to catch syntax errors early.
                compiled_code = compile(code, "<string>", "exec")
            except SyntaxError as se:
                logger.error(f"Syntax error during compilation: {se}")
                self.error_msg = f"Syntax error: {se}"
                return None
            
            exec(compiled_code, namespace, namespace)
        except Exception as e:
            logger.info(f"Failed to execute llm generated code. Error: {e}")
            self.error_msg = f"Failed to execute code: {e}"
            return None

        if "result" not in namespace:
            logger.error("The executed code did not define a variable called 'result'.")
        
        return namespace.get("result", None)
    
    @staticmethod
    def verify_unit(response_unit_expr, answer_unit_expr) -> bool:
        """
        Verifies that the unit provided in the response is equivalent
        to the ground truth unit by comparing their simplified sympy expressions.
        """
        try: 
            logger.info(f'Comparing response unit ({response_unit_expr}) with answer unit ({answer_unit_expr})')
            diff = sp.simplify(response_unit_expr - answer_unit_expr)
            return diff == 0
        except Exception as e:
            logger.error("Failed to compare units:", e)
            return False
        
    def execute_code(self, code: str, timeout: int = 120):
        """
        Executes the generated code from the response in an isolated namespace.
        The environment includes sympy and the necessary physics units.
        The code is expected to define a variable called 'result'.
        """

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.safe_exec, code)
        try:
            result = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.error(f"Execution timed out after {timeout} seconds")
            self.error_msg = f"Execution timed out after {timeout} seconds"
        
        return result
        
    def convert_units(self, output, response_unit_expr, answer_unit_expr):
        """
        Convert a value from one unit to another.
        
        Parameters:
            output: The value to convert
            response_unit_expr: The original unit
            answer_unit_expr: The target unit
            
        Returns:
            (converted_output, converted_unit): The converted value and unit
        """
        try:
            output_with_unit = output * response_unit_expr

            # Get scaling factor and base answer units
            scaling_factor, base_unit = self.unit_parser.detect_scaling_factor(answer_unit_expr)

            answer_unit_args = self.unit_parser.detect_unit_args(base_unit)

            if len(answer_unit_args) > 1:
                logger.info(f'Answer unit is a composite unit with: {answer_unit_args}')
            
            # Perform the unit conversion
            converted_output_expr = units.convert_to(output_with_unit, answer_unit_args)
            logger.info(f'Converted response expr: {converted_output_expr}')

            output, response_unit_expr = self.unit_parser.extract_value_and_unit(converted_output_expr)
            
            if not isinstance(output, (int, float, sp.Number)):
                raise ValueError(f"Failed to extract value from converted output: {output}")
            
            output = float(output)

            # Apply scaling factor if needed
            if scaling_factor != 1:
                logger.info(f'Applying scaling factor {scaling_factor} for answer units')
                output /= scaling_factor
                response_unit_expr *= scaling_factor
            
            logger.info(f'Converted output: {output}')
            logger.info(f'Converted unit: {response_unit_expr}')
            
            return output, response_unit_expr
        
        except Exception as e:
            logger.error(f'Unit conversion failed: {e}')
            return output, response_unit_expr

    def verify(self, response: ResponseFormat, answer: AnswerFormat) -> Tuple[Any, bool, bool]:
        """
        Use LLM to verify whether the code's output and the unit matches the ground truth.
        """
        if answer.unit == "dimensionless":
            answer.unit = None

        self.response = response
        self.answer = answer

        # Clean and execute the code provided in the response
        cleaned_code = clean_python_code(self.response.code)
        output = self.execute_code(cleaned_code)

        if output is None:
            return VerificationResult(code_output=None, result_match=False, unit_match=False, error=self.error_msg)
        
        if self.unit_parser.unit_is_none(answer.unit) and self.unit_parser.unit_is_none(response.unit):
            response_unit_expr = None
            answer_unit_expr = None
        else:
            response_unit_expr = self.unit_parser.parse_unit(response.unit)
            answer_unit_expr = (self.unit_parser.parse_unit(answer.unit) if answer.unit else None)

        logger.info(f'Response unit: {response_unit_expr}')
        logger.info(f'Ground truth unit: {answer_unit_expr}')
        cleaned_answer = clean_answer(answer.gt_answer)
        logger.info(f'Response expression: {output}')
        gt_expr = cleaned_answer.lstrip("$").rstrip("$")
        logger.info(f'Ground truth expression: {gt_expr}')
        # Use LLM Verifier
        try:
            prompt_message = f"Response: Result={output}, Unit={response_unit_expr}\nGround Truth: Result={gt_expr}, Unit={answer_unit_expr}"
            raw_response = self.verifier_agent.step(prompt_message, response_format=LLMVerificationResult)
            structured_response = LLMVerificationResult.model_validate(raw_response.msgs[0].parsed)
        except Exception as e:
            logger.info(f'Failed to verify with error {e}')
            self.error_msg=f'Failed to verify with error {e}'
        return VerificationResult(code_output=str(output), result_match=structured_response.result_match, unit_match=structured_response.unit_match, error=self.error_msg)