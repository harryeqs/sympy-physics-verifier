from models import ResponseFormat, AnswerFormat
from pydantic import BaseModel
import sympy as sp
import re
import math
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from sympy.physics import units
from sympy.physics.units.prefixes import PREFIXES

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# A dictionary to allow parsing of common unit expressions.
allowed_units = {
    "m": units.meter,
    "meter": units.meter,
    "meters": units.meter,
    "s": units.second,
    "second": units.second,
    "seconds": units.second,
    "kg": units.kilogram,
    "kilogram": units.kilogram,
    "kilograms": units.kilogram,
    "N": units.newton,
    "newton": units.newton,
    "J": units.joule,
    "joule": units.joule,
    "circ": units.degree,
    "degree": units.degree,
    "degrees": units.degree,
    "K": units.kelvin,
    "kelvin": units.kelvin,
    "g": units.gram,
    "gram": units.gram,
    "grams": units.gram,
    "cm": units.centimeter,
    "km": units.kilometer,
    "kilometer": units.kilometer,
    "centimeter": units.centimeter,
    # extend as needed
}

allowed_prefixed_units = {}

# ✅ Add SI Prefix Support to Allowed Units
for prefix, prefix_obj in PREFIXES.items():
    for unit_name, base_unit in allowed_units.copy().items():
        prefixed_unit_name = f"{prefix}{unit_name}"  # Example: "MJ", "kN"
        allowed_prefixed_units[prefixed_unit_name] = prefix_obj.scale_factor * base_unit

#avoid repetition
allowed_prefixed_units = {k: v for k, v in allowed_prefixed_units.items() if k not in allowed_units}

allowed_units.update(allowed_prefixed_units)

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

def detect_unit_args(unit_expr):
    """
    Extracts the base units from a composite SymPy unit expression.

    Parameters:
        unit_expr: SymPy expression representing a composite unit (e.g., kg/m^3)

    Returns:
        List of SymPy base unit components (e.g., [kg, m])
    """
    factors = sp.Mul.make_args(unit_expr)  # Decompose into factors
    
    base_units = [factor.base if factor.is_Pow else factor for factor in factors]

    return base_units

def detect_scaling_factor(answer_unit_expr):
    """
    Detects a scaling factor in the answer unit expression.
    
    Parameters:
        answer_unit_expr (SymPy expression): The unit expression from the correct answer.

    Returns:
        (scale_factor, base_unit): Tuple of scale factor (if found) and the base unit.
    """
    # Extract value and unit from the answer's unit
    value, base_unit = extract_value_and_unit(answer_unit_expr)

    # If the extracted value is purely numeric, it's a scale factor
    if isinstance(value, (int, float, sp.Number)):
        return value, base_unit  # scale_factor, unit
    return 1, answer_unit_expr  # No scale factor found, return 1


def extract_value_and_unit(expr):
    """
    Extracts the numerical value and unit from a SymPy expression.
    Correctly handles compound units (e.g., m/s, AU, N*m, kg*m/s^2).
    
    Parameters:
        expr: SymPy expression with units (e.g., 3604.36 * meter / second, 0.592092647418689 * AU)
    
    Returns:
        (value, unit): Numerical value and unit as separate expressions.
    """
    # Flatten the expression into multiplicative terms
    factors = sp.Mul.make_args(expr)

    # Separate numerical values and unit terms
    numeric_terms = []
    unit_terms = []

    for term in factors:
        if term.is_number:  # If it's a pure number, store in numeric_terms
            numeric_terms.append(term)
        elif isinstance(term, sp.Symbol):  # Ensure AU or other units are handled
            unit_terms.append(term)
        elif any(term.has(u) for u in units.__dict__.values()):  # If term contains known units
            unit_terms.append(term)
        else:
            # Handle unknown symbols as part of the value (e.g., h, g in symbolic cases)
            numeric_terms.append(term)

    # Construct the final numerical value and unit
    value = sp.Mul(*numeric_terms) if numeric_terms else 1  # If no value found, assume 1
    unit_expr = sp.Mul(*unit_terms) if unit_terms else 1  # If no unit found, assume dimensionless

    return value, unit_expr


def parse_unit_with_latex(unit_str: str):
    """
    Parse a unit string using sympy's LaTeX parser.
    After parsing, substitute any symbols with their corresponding
    unit objects defined in allowed_units. This function also attempts
    to reassemble composite units that the LaTeX parser splits into separate symbols.
    
    Parameters:
        unit_str (str): The unit string in LaTeX format, e.g. "$\\frac{\\mathrm{kg}}{\\mathrm{m}^{3}}$"
    
    Returns:
        A simplified sympy expression representing the unit.
    """
    # Remove any leading/trailing whitespace and dollar signs or ^ characters.
    unit_str = unit_str.strip().lstrip("$").rstrip("$").lstrip("^")
    
    # Preprocess the string: Replace \mathrm{...} with its inner content.
    unit_str = re.sub(r'\\mathrm\{([^}]*)\}', r'{\\\1}', unit_str)

    # remove random symbols like tilde
    unit_str = unit_str.replace('~', '')
    
    try:
        expr = parse_latex(unit_str)
        logger.info(f"Parsed LaTeX unit: {expr}.")
    except Exception as e:
        raise ValueError(f"Failed to parse LaTeX unit '{unit_str}': {e}")
    
    # Substitute allowed unit symbols with their corresponding objects.
    for key, unit_obj in allowed_units.items():
        sym = sp.symbols(key)
        expr = expr.subs(sym, unit_obj)
    
    simplified_expr = sp.simplify(expr)
    logger.info(f"Simplified LaTex unit: {simplified_expr}")
    return simplified_expr

def preprocess_unit_string(unit_str: str) -> str:
    """
    Preprocess a unit string to replace '^' with '**' for exponentiation.
    This is used for non-LaTeX strings.
    """
    return unit_str.replace('^', '**').strip()

class PhysicsVerifier:
    def __init__(self, torlerance: float =1e-2):
        # Set the tolerance for float comparisons.
        self.torlerance = torlerance

    def execute_code(self, code: str):
        """
        Executes the generated code from the response in an isolated namespace.
        The environment includes sympy and the necessary physics units.
        The code is expected to define a variable called 'result'.
        """
        namespace = {}
        try:
            exec(code, namespace, namespace)
            if "result" not in namespace:
                logger.info("The executed code did not define a variable called 'result'.")
        except Exception as e:
            logger.info(f"Failed to execute code: {e}")
        
        return namespace.get("result", None)

    def parse_unit(self, unit_str: str):
        """
        Parses a unit string into a sympy expression.
        If the string looks like LaTeX (e.g. enclosed in $...$ or contains LaTeX commands),
        it uses sympy's LaTeX parser; otherwise, it falls back to using parse_expr.
        """
        if "$" in unit_str or "\\" in unit_str:
            # Likely a LaTeX formatted string.
            return parse_unit_with_latex(unit_str)
        
        processed_str = preprocess_unit_string(unit_str)
        
        try:
            expr = parse_expr(processed_str, local_dict=allowed_units, evaluate=True)
            return sp.simplify(expr)
        except Exception as e:
            raise ValueError(f"Failed to parse unit '{unit_str}' (processed as '{processed_str}'): {e}")
            
    def parse_answer_and_response_units(self):
        try: 
            response_unit_expr = self.parse_unit(self.response.unit)
            logger.info(f'Response unit: {response_unit_expr}')
            answer_unit_expr = self.parse_unit(self.answer.unit)
            logger.info(f'Answer unit: {answer_unit_expr}')
            return response_unit_expr, answer_unit_expr
        except Exception as e:
            logger.error("Failed to parse units:", e)
            return None, None

    def verify_unit(self, response_unit_expr, answer_unit_expr) -> bool:
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

    def verify(self, response: ResponseFormat, answer: AnswerFormat) -> tuple[bool, bool]:
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
            return False, False
        
        if answer.unit:
            response_unit_expr, answer_unit_expr = self.parse_answer_and_response_units()
        
        if isinstance(output, (int, float, sp.Number)):
            if answer.unit and (response_unit_expr is not None) and (answer_unit_expr is not None):
                raw_unit_match = self.verify_unit(response_unit_expr, answer_unit_expr)

                if not raw_unit_match:
                    logger.info(f'Response unit ({response_unit_expr}) does not match the answer unit ({answer_unit_expr}). Attempting to convert...')

                    output_with_unit = output * response_unit_expr

                    try:
                        scaling_factor, base_unit = detect_scaling_factor(answer_unit_expr)
                    except Exception as e:
                        logger.error(f'Failed to detect scaling factor for answer unit ({answer_unit_expr}): {e}')

                    try:
                        answer_unit_args = detect_unit_args(base_unit)

                        if len(answer_unit_args) > 1:
                            logger.info(f'Answer unit is a composite unit with: {answer_unit_args}')

                        converted_output_expr = units.convert_to(output_with_unit, answer_unit_args)
                        logger.info(f'Converted response expr: {converted_output_expr}')
                    except Exception as e:
                        logger.error(f'Failed to convert output to the target unit: {e}')

                    try:
                        output, response_unit_expr = extract_value_and_unit(converted_output_expr)

                        if not isinstance(output, (int, float, sp.Number)):
                            raise ValueError(f"Failed to extract value from converted output: {output}")
                        
                        output = float(output)

                        if scaling_factor != 1:
                            logger.info(f'Apply scaling factor {scaling_factor} for answer units to response.')
                            output /= scaling_factor
                            response_unit_expr *= scaling_factor

                        logger.info(f'Converted response output: {output}')
                        logger.info(f'Converted response unit: {response_unit_expr}')
                    except Exception as e:
                        logger.error(f'Failed to exparate output value and unit: {e}')
            
            try:
                gt_value = float(clean_answer(answer.gt_answer))
                logger.info(f'Response value: {output}')
                logger.info(f'Ground truth value: {gt_value}')
            except Exception as e:
                logger.error("Failed to convert ground truth answer to float. Error:", e)
                return False, False
            
            tolerance = self.torlerance
            result_match = math.isclose(output, gt_value, rel_tol=tolerance)

        else:
            # For symbolic expressions, convert both the output and the ground truth to LaTeX.
            # (Assumes that answer.gt_answer is provided as a LaTeX string)
            logger.info(f'Output expression: {output}')
            gt_expr = parse_latex(answer.gt_answer.lstrip("$").rstrip("$"))
            logger.info(f'Ground truth expression: {output}')
            
            try:
                result_match = sp.simplify(output - gt_expr) == 0
            except Exception as e:
                logger.error("Failed to compare symbolic expressions. Error:", e)
                return False, False

        if not answer.unit:
            # If the ground truth unit is not provided, only check the result.
            return result_match, True
        elif response_unit_expr is None or answer_unit_expr is None:
            return result_match, False
        
        unit_match = self.verify_unit(response_unit_expr, answer_unit_expr)
        return result_match, unit_match


# Example usage:
if __name__ == "__main__":
    # Example response with a LaTeX formatted unit string.
    response = ResponseFormat(
        reasoning="Example with LaTeX formatted unit.",
        code="result = 1.00",
        unit="^{\circ}"
    )
    # Ground truth with equivalent unit in a Python-friendly format.
    answer = AnswerFormat(
        gt_answer='$1$',
        unit="degree"
    )
 
    verifier = PhysicsVerifier()
    try:
        result_match, unit_match = verifier.verify(response, answer)
        logger.info("Result Match:", result_match)
        logger.info("Unit Match:", unit_match)
    except Exception as e:
        logger.error("Verification failed:", e)

