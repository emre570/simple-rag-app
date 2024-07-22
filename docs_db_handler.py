import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import TextSplitter
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

def load_docs(data_folder):
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    doc_loader = PyPDFDirectoryLoader(data_folder)
    
    docs = doc_loader.load()
    return docs

def init_db(chunks, embeddings_model, folder_path, embeddings):
    """
    Initialize FAISS with given chunks and embedding model, save it to the folder path, or load from the folder if it exists.
    """
    faiss_path = os.path.join(folder_path, "index.faiss")
    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings_model)
        os.makedirs(folder_path, exist_ok=True)
        vectorstore.save_local(folder_path)
    return vectorstore

def add_db_docs(vectorstore, folder_path, embeddings_model):
    """
    Load documents from the folder, check if they exist in the vectorstore, and add them if they don't.
    """
    documents = load_docs(folder_path)
    for document in documents:
        embedding = embeddings_model.encode(document)
        result = vectorstore.similarity_search_by_vector(embedding, k=1)
        if not (result and result[0].score > 0.9):  # Adjust the threshold as needed
            chunks = TextSplitter().split_text(document)
            vectorstore.add_documents(chunks, embeddings_model)
    vectorstore.save_local(folder_path)  # Save the updated vectorstore