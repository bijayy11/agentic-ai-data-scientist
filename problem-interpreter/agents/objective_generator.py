from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from models.local_llama3 import get_local_llama3
from components.pinecone_store import search_similar_prompt, store_prompt_result

def generate_objective(user_prompt):
    cached = search_similar_prompt(f"objective::{user_prompt}")
    if cached:
        return cached.split("=>")[1].strip()

    prompt = PromptTemplate.from_template("""
    Given the user requirement: "{input}", extract the modeling objective in one sentence.
    """)
    chain = LLMChain(llm=get_local_llama3(), prompt=prompt)
    result = chain.run(input=user_prompt)
    store_prompt_result(f"objective::{user_prompt}", result)
    return result