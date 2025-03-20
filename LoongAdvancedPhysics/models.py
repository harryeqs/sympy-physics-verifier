from pydantic import BaseModel
from typing import Union, Any, Dict

class ResponseFormat(BaseModel):
   code: str
   unit: Union[str, None] = None

class AnswerFormat(BaseModel):
   gt_answer: str
   unit: Union[str, None] = None

class VerificationResult(BaseModel):
   code_output: Union[str, None]
   result_match: bool
   unit_match: bool

class OutputFormat(BaseModel):
   sample_id: str
   response: ResponseFormat
   answer: AnswerFormat
   verification_result: VerificationResult
   metadata: Dict
   