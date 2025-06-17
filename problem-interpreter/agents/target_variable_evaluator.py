from agents.config import *

parser = PydanticOutputParser(pydantic_object=TargetVariableResponseSchema)

def get_target_variable_and_metrics(user_prompt, metadata):
    key = f"target_metrics::{user_prompt}::{str(metadata)}"

    prompt_template_str = """
    Given the user requirement: "{input}" and dataset metadata: {metadata},
    identify the target variable and appropriate evaluation metrics.

    Return a JSON object in this exact format:
    {{
        "target_variable": "<name of the target variable>",
        "evaluation_metrics": ["<metric1>", "<metric2>", ...]
    }}

    Respond ONLY with valid JSON. No comments or explanations.
    """

    prompt = PromptTemplate.from_template(prompt_template_str)
    llm = get_local_llama3()

    retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=llm)

    raw_output = (prompt | llm).invoke({"input": user_prompt, "metadata": metadata})

    formatted_prompt = prompt.format(input=user_prompt, metadata=metadata)
    result = retry_parser.parse_with_prompt(raw_output, formatted_prompt)

    message = store_data(DataInput(text=f"target_metrics::{user_prompt} => {result}", metadata={"source": "test"}))
    if message:
        print(f"Stored result: {message}")
    else:
        print("Failed to store the result.")
    return result

if __name__ == "__main__":
    user_input = input("Enter the user requirement: ")
    metadata = input("Enter dataset metadata (e.g., column names, types): ")

    result = get_target_variable_and_metrics(user_input, metadata)

    print(f"Target Variable: {result.target_variable}")
    print(f"Evaluation Metrics: {result.evaluation_metrics}")

