from langchain_community.vectorstores import FAISS

def init_faiss(chunks, embeddings_model):
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings_model)
    return vectorstore