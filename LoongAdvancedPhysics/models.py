from pydantic import BaseModel
from typing import Union, Any, Dict

class ResponseFormat(BaseModel):
   code: str
   unit: Union[str, None] = None

class AnswerFormat(BaseModel):
   cot: Union[str, None] = None
   gt_answer: str
   unit: Union[str, None] = None

class VerificationResult(BaseModel):
   code_output: Union[str, None]
   result_match: bool
   unit_match: bool
   error: Union[str, None]

class OutputFormat(BaseModel):
   sample_id: str
   response: ResponseFormat
   answer: AnswerFormat
   verification_result: Union[VerificationResult, None]
   metadata: Dict

class CodeVerificationResult(BaseModel):
   sample_id: str
   is_valid: bool
   issue: str         