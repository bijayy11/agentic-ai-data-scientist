from langchain_ollama import OllamaLLM

def get_local_llama3():
    return OllamaLLM(model="llama3", temperature=0.4)
