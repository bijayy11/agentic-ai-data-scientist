import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.output_parsers import PydanticOutputParser,RetryWithErrorOutputParser

from pydantic import BaseModel,Field
from typing import List,Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from models.local_llama3 import get_local_llama3
from components.pinecone_config import *
import json
class ObjectiveResponseSchema(BaseModel):
    Objective: str = Field(..., description="The objective or goal of the machine learning task, e.g., 'Predict customer churn', 'Classify loan applications as risky or not', etc.")

class ProblemResponseSchema(BaseModel):
    problem_type: str = Field(..., description="The type of problem identified by the model, e.g., Binary Classification, Regression")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model's confidence score between 0 and 1")

class TargetVariableResponseSchema(BaseModel):
    target_variable: str = Field(..., description="The target variable to be predicted or classified")
    evaluation_metrics: List[str] = Field(..., description="List of evaluation metrics suitable for the problem type, e.g., ['accuracy', 'precision', 'recall']")