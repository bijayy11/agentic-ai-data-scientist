from agents.config import *

parser = PydanticOutputParser(pydantic_object=ProblemResponseSchema)


def classify_problem_type(user_prompt):
    # cached = search_similar_prompt(f"problem_type::{user_prompt}")
    # if cached:
    #     return cached.split("=>")[1].strip()
    print(f"Classifying problem type for user prompt: {user_prompt}")
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
    # message = store_data(DataInput(text=f"problem_type::{user_prompt} => {result}", metadata={"source": "test"}))
    # if message:
    #     print(f"Stored result: {message}")
    # else:
    #     print("Failed to store the result.")
    return result


if __name__ == "__main__":
    user_input = input("Enter a user prompt: ")
    problem_type = classify_problem_type(user_input)
    print(f"Classified problem type: {problem_type}")