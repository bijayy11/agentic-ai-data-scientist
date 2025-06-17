from agents.config import *

parser = PydanticOutputParser(pydantic_object=ObjectiveResponseSchema)

def generate_objective(user_prompt):
    # cached = search_similar_prompt(f"objective::{user_prompt}")
     # if cached:
    #     return cached.split("=>")[1].strip()

    prompt_template_str = """
    You are an expert in Machine Learning tasked with parsing the natural language problem into machine learning and/or objective.

    Task: {input}

    Return a JSON object in the following **exact format**:
    {{
    "Objective": "<The objective or goal of the machine learning task, e.g., 'Predict customer churn', 'Classify loan applications as risky or not', etc.>"
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
    sample_data = input("Type something to generate objective...")
    result = generate_objective(sample_data)
    print(f"Generated Objective: {result}")