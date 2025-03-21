from models import ResponseFormat, AnswerFormat, VerificationResult
from typing import Any, Tuple
import sympy as sp
import concurrent.futures
import re
import math
import multiprocessing
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from sympy.physics import units
from sympy.physics.units.prefixes import PREFIXES

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class UnitParser:
    """
    Class for handling unit parsing and manipulation operations.
    """

    def __init__(self):
        # Base unit dictionary
        extra_allowed_units = {
            'hrs': units.hour, 
            'min': units.minute, 
            'Joule': units.joule, 
            'Joules': units.joule, 
            'circ': units.degree,
            "Omega": units.ohm,
            '%': units.Unit('percent'),
        }

        self.allowed_units = self._load_sympy_units()
        self.allowed_units.update(extra_allowed_units)
        
        # Add SI prefixed units
        self._add_si_prefixes()

    @staticmethod
    def _load_sympy_units():
        sympy_units = {}

        for attr_name in dir(units):
            unit_obj = getattr(units, attr_name)
            if isinstance(unit_obj, units.Quantity):
                sympy_units[attr_name] = unit_obj

        return sympy_units
    
    def _add_si_prefixes(self):
        """Add SI prefixed units (like km, MHz, etc.) to the allowed units."""
        prefixed_units = {}
        for prefix, prefix_obj in PREFIXES.items():
            for unit_name, base_unit in self.allowed_units.copy().items():
                prefixed_unit_name = f"{prefix}{unit_name}"  # Example: "MJ", "kN"
                prefixed_units[prefixed_unit_name] = prefix_obj.scale_factor * base_unit
        
        # Add only new prefixed units that don't conflict with existing ones
        prefixed_units = {k: v for k, v in prefixed_units.items() if k not in self.allowed_units}
        self.allowed_units.update(prefixed_units)

    def parse_unit(self, unit_str: str):
        """
        Parse a unit string into a SymPy expression using the appropriate method.
        
        Parameters:
            unit_str (str): The unit string to parse
            
        Returns:
            SymPy expression representing the unit
        """
        if not unit_str or unit_str == "dimensionless":
            return None
            
        if "$" in unit_str or "\\" in unit_str:
            # Likely a LaTeX formatted string
            return self.parse_unit_with_latex(unit_str)
        
        # Standard unit string
        processed_str = self.preprocess_unit_string(unit_str)
        
        try:
            expr = parse_expr(processed_str, local_dict=self.allowed_units, evaluate=True)
            return sp.simplify(expr)
        except Exception as e:
            logger.info(f"Failed to parse unit '{unit_str}' (processed as '{processed_str}'): {e}")
            return None
    
    def parse_unit_with_latex(self, unit_str: str):
        """
        Parse a unit string using SymPy's LaTeX parser.
        
        Parameters:
            unit_str (str): The unit string in LaTeX format
            
        Returns:
            SymPy expression representing the unit
        """
        # Clean the LaTeX string
        unit_str = unit_str.strip().lstrip("$").rstrip("$").lstrip("^")
        unit_str = re.sub(r'\\mathrm\{([^}]*)\}', r'{\\\1}', unit_str)
        unit_str = unit_str.replace('~', '')
        
        try:
            expr = parse_latex(unit_str)
            logger.info(f"Parsed LaTeX unit: {expr}.")
        except Exception as e:
            logger.info(f"Failed to parse LaTeX unit '{unit_str}': {e}")
            return unit_str
        
        # Substitute allowed unit symbols
        for key, unit_obj in self.allowed_units.items():
            sym = sp.symbols(key)
            expr = expr.subs(sym, unit_obj)
        
        simplified_expr = sp.simplify(expr)
        logger.info(f"Simplified LaTeX unit: {simplified_expr}")
        return simplified_expr
    
    def detect_scaling_factor(self, unit_expr):
        """
        Detect a scaling factor in the unit expression.
        
        Parameters:
            unit_expr (SymPy expression): The unit expression
            
        Returns:
            (scale_factor, base_unit): Tuple of scale factor and base unit
        """
        value, base_unit = self.extract_value_and_unit(unit_expr)
        
        if isinstance(value, (int, float, sp.Number)):
            return value, base_unit
        return 1, unit_expr
    
    @staticmethod
    def preprocess_unit_string(unit_str: str) -> str:
        """
        Preprocess a unit string to replace '^' with '**' for exponentiation.
        
        Parameters:
            unit_str (str): The unit string to preprocess
            
        Returns:
            Preprocessed unit string
        """
        superscript_map = {
            "\u00b2": "2",  # Superscript ²
            "\u00b3": "3",  # Superscript ³
            "\u2070": "0", "\u2071": "1", "\u2074": "4", "\u2075": "5",
            "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9"
        }

        for unicode_char, normal_char in superscript_map.items():
            unit_str = unit_str.replace(unicode_char, "**" + normal_char)

        unit_str = unit_str.replace('^', '**').strip()
        return unit_str
    
    @staticmethod
    def unit_is_none(unit_str: str) -> bool:
        if unit_str is None:
            return True
        
        if isinstance(unit_str, str):
            unit_str = unit_str.strip().lower()

            if unit_str == 'none' or unit_str == '':
                return True
            
        return False
    
    @staticmethod
    def extract_value_and_unit(expr):
        """
        Extract the numerical value and unit from a SymPy expression.
        
        Parameters:
            expr: SymPy expression with units
            
        Returns:
            (value, unit): Numerical value and unit as separate expressions
        """
        # Flatten the expression into multiplicative terms
        factors = sp.Mul.make_args(expr)
        
        # Separate numerical values and unit terms
        numeric_terms = []
        unit_terms = []
        
        for term in factors:
            if term.is_number:  
                numeric_terms.append(term)
            elif isinstance(term, sp.Symbol):  
                unit_terms.append(term)
            elif any(term.has(u) for u in units.__dict__.values()):  
                unit_terms.append(term)
            else:
                # Handle unknown symbols as part of the value
                numeric_terms.append(term)
        
        # Construct the final numerical value and unit
        value = sp.Mul(*numeric_terms) if numeric_terms else 1
        unit_expr = sp.Mul(*unit_terms) if unit_terms else 1
        
        return value, unit_expr
    
    @staticmethod
    def detect_unit_args(unit_expr):
        """
        Extract the base units from a composite SymPy unit expression.
        
        Parameters:
            unit_expr: SymPy expression representing a composite unit
            
        Returns:
            List of SymPy base unit components
        """
        factors = sp.Mul.make_args(unit_expr)
        base_units = [factor.base if hasattr(factor, 'is_Pow') and factor.is_Pow else factor for factor in factors]
        return base_units
    

def is_number(s):
    try:
        float(s)
    except ValueError:
        return False
    return True

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
    
    # Remove any stray single quotes around individual code lines
    code = re.sub(r"^'(.*)'$", r"\1", code, flags=re.MULTILINE)
    
    # Handle cases where entire code is wrapped in quotes with commas
    if "'," in code:
        parts = code.split("',")
        cleaned_parts = []
        for part in parts:
            cleaned_part = part.strip()
            if cleaned_part.startswith("'"):
                cleaned_part = cleaned_part[1:]
            cleaned_parts.append(cleaned_part)
        code = "\n".join(cleaned_parts)
    
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

class PhysicsVerifier:
    def __init__(self, tolerance: float = 1e-2):
        # Set the tolerance for float comparisons.
        self.tolerance = tolerance
        self.unit_parser = UnitParser()
        self.error_msg = None

    def safe_exec(self, code):
        namespace = {}
        try:
            pattern = r'([\w\d_]+)\.evalf\(\)'
            evalf_vars = set(re.findall(pattern, code))

            for var in evalf_vars:
                # Replace `XXX.evalf()` with a conditional check
                safe_evalf = f"{var} if isinstance({var}, float) else {var}.evalf()"
                code = code.replace(f"{var}.evalf()", safe_evalf)

            exec(code, namespace, namespace)
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

        with concurrent.futures.ProcessPoolExecutor() as executor:
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
        Verifies that:
        - The executed code's output (variable 'result') matches the ground truth answer.
        - The unit provided in the response is equivalent to the ground truth unit.
        
        Returns:
            A tuple (result_match: bool, unit_match: bool)
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
            response_unit_expr = self.unit_parser.parse_unit(response.unit.strip())
            answer_unit_expr = self.unit_parser.parse_unit(answer.unit.strip())

        logger.info(f'Response unit: {response_unit_expr}')
        logger.info(f'Ground truth unit: {answer_unit_expr}')

        cleaned_answer = clean_answer(answer.gt_answer)

        if is_number(cleaned_answer):    
            gt_value = float(cleaned_answer)

            if not isinstance(output, (int, float, sp.Number)):
                logger.info(f'Convert output expr {output} into numerical.')
                try:
                    output = output.evalf()
                except Exception as e:
                    logger.error(f"Failed to evaluate output {output}: {e}")
                    result_match = False
           
            if (answer_unit_expr is not None and 
                response_unit_expr is not None and 
                not self.verify_unit(response_unit_expr, answer_unit_expr)):

                logger.info(f'Units do not match directly. Attempting conversion...')
                output, response_unit_expr = self.convert_units(
                    output, response_unit_expr, answer_unit_expr
                ) 
            
            try:
                #Compare numerical values
                logger.info(f'Response value: {output}')
                logger.info(f'Ground truth value: {gt_value}')
                result_match = math.isclose(output, gt_value, rel_tol=self.tolerance)
            except Exception as e:
                logger.error(f"Failed to compare values: {e}")
                result_match = False

        else:
            # Compare symbolic expressions
            try:
                logger.info(f'Response expression: {output}')
                gt_expr = parse_latex(answer.gt_answer.lstrip("$").rstrip("$"))
                gt_expr = gt_expr.subs({
                    sp.Symbol('pi'): sp.pi,
                    sp.Symbol('e'): sp.E
                })
                logger.info(f'Ground truth expression: {gt_expr}')

                if isinstance(gt_expr, (int, float, sp.Number)):
                    output = output.evalf()

                result_match = sp.simplify(output - gt_expr) == 0
            except Exception as e:
                logger.error(f"Failed to compare symbolic expressions: {e}")
                self.error_msg = f"Failed to compare symbolic expressions: {e}"
                result_match = False

        if self.unit_parser.unit_is_none(answer.unit):
            # If the answer is dimensionless, the reponse should also be dimensionless
            unit_match = self.unit_parser.unit_is_none(answer.unit)
        elif response_unit_expr is None or answer_unit_expr is None:
            unit_match = False
        else:
            unit_match = self.verify_unit(response_unit_expr, answer_unit_expr)

        print("self.error_msg:", self.error_msg)
        return VerificationResult(code_output=str(output), result_match=result_match, unit_match=unit_match, error=self.error_msg)


# Example usage:
if __name__ == "__main__":
    # Set up logging for testing
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    # Example response with a LaTeX formatted unit string.
    response = ResponseFormat(
        reasoning="Example with LaTeX formatted unit.",
        code="import sympy as sp\nresult = 3*sp.pi",
        unit="meter^2"
    )
    # Ground truth with equivalent unit in a Python-friendly format.
    answer = AnswerFormat(
        gt_answer='$3\\pi$',
        unit="$m^2$"
    )
 
    verifier = PhysicsVerifier()
    try:
        outcome = verifier.verify(response, answer)
        logger.info(f"Code execution output: {outcome.code_output}")
        logger.info(f"Result Match: {outcome.result_match}")
        logger.info(f"Unit Match: {outcome.unit_match}")
    except Exception as e:
        logger.error("Verification failed:", e)

