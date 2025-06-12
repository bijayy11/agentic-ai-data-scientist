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

class ResponseSchema(BaseModel):
    problem_type: str = Field(..., description="The type of problem identified by the model, e.g., Binary Classification, Regression")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model's confidence score between 0 and 1")

parser = PydanticOutputParser(pydantic_object=ResponseSchema)


def classify_problem_type(user_prompt):
    cached = search_similar_prompt(f"problem_type::{user_prompt}")
    # if cached:
    #     return cached.split("=>")[1].strip()

    prompt_template_str = """
    You are an expert in Machine Learning tasked with classifying the problem described below.

    Task: {input}

    Return a JSON object in the following **exact format**:

    {{
    "problem_type": "<eg.,Classification or Regression>",
    "confidence": <A float value between 0 and 1 indicating how confident you are>
    }}

    Respond ONLY with valid JSON. No comments or explanations.
    """

    prompt = PromptTemplate.from_template(prompt_template_str)
    llm = get_local_llama3()
    
    retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=llm)
    
    # Run the prompt and model separately
    raw_output = (prompt | llm).invoke({"input": user_prompt})
    
    # Use the retry parser explicitly with the prompt
    result = retry_parser.parse_with_prompt(raw_output, prompt.format(input=user_prompt))

    # Store the result
    message = store_data(DataInput(text=f"problem_type::{user_prompt} => {result}", metadata={"source": "test"}))
    if message:
        print(f"Stored result: {message}")
    else:
        print("Failed to store the result.")
    return result


if __name__ == "__main__":
    user_input = input("Enter a user prompt: ")
    problem_type = classify_problem_type(user_input)
    print(f"Classified problem type: {problem_type}")