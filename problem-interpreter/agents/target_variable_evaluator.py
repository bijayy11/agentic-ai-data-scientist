from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from models.local_llama3 import get_local_llama3
from components.pinecone_store import search_similar_prompt, store_prompt_result

def get_target_variable_and_metrics(user_prompt, metadata):
    key = f"target_metrics::{user_prompt}::{str(metadata)}"
    cached = search_similar_prompt(key)
    if cached:
        response = cached.split("=>")[1].strip()
        return response.split("|TARGET|")[0], response.split("|TARGET|")[1]

    prompt = PromptTemplate.from_template("""
    Given the user requirement: "{input}" and dataset metadata: {metadata},
    identify the target variable and appropriate evaluation metrics.
    Output as: <target>|TARGET|<metrics>
    """)
    chain = LLMChain(llm=get_local_llama3(), prompt=prompt)
    result = chain.run(input=user_prompt, metadata=metadata)
    store_prompt_result(key, result)
    return result.split("|TARGET|")[0], result.split("|TARGET|")[1]
