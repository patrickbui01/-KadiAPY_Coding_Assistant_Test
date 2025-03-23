from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def setup_vectorstore(docs, embedding_model,  persist_directory,):
    print("Start setup_vectorstore_function")
    vectorstore = get_chroma_vectorstore(embedding_model, persist_directory)
    vectorstore.add_documents(docs)
    return vectorstore

def get_chroma_vectorstore(embedding_model, vectorstore_path):
    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_model)
    
    return vectorstore