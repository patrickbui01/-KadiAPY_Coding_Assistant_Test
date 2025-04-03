from langchain_groq import ChatGroq

def get_groq_llm(model_name, temperature, api_key):
    llm = ChatGroq(model=model_name, temperature=temperature, api_key=api_key)
    return llm
