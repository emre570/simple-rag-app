from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
import os

def init_faiss(chunks, embeddings_model, db_path):
    """
    Initialize the FAISS vector store. If a path is provided, load the existing database, else create a new one.
    """
    index_file = os.path.join(db_path, "index.faiss")
    
    if os.path.exists(index_file):
        vectorstore = FAISS.load_local(db_path, embeddings=embeddings_model, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings_model)
        os.makedirs(db_path, exist_ok=True)
        vectorstore.save_local(db_path)
    
    return vectorstore

def save_faiss(vectorstore, db_path):
    """
    Save the FAISS vector store to the specified path.
    """
    vectorstore.save_local(db_path)

def add_document(vectorstore, document, embeddings_model, threshold=0.99):
    """
    Add a document to the FAISS vector store if it is not already present.

    Parameters:
    - vectorstore: The FAISS vector store instance.
    - document: The document text to add.
    - embeddings_model: The model used to generate embeddings.
    - threshold: Similarity threshold for considering documents as duplicates.
    """
    doc_embedding = embeddings_model.embed_documents([document])[0]
    search_results = vectorstore.similarity_search_by_vector(doc_embedding, k=1)

    # Debug: Print search results to understand their structure
    print(f"Search results: {search_results}")

    if not search_results:
        vectorstore.add_documents([document])
        return True

    # Extract similarity score based on the structure of the search results
    if isinstance(search_results[0], tuple):
        similarity_score = search_results[0][1]
    elif hasattr(search_results[0], 'score'):
        similarity_score = search_results[0].score
    else:
        raise TypeError("Unexpected search result structure")

    if similarity_score < threshold:
        vectorstore.add_documents([document])
        return True

    return False

def load_docs(data_folder):
    """
    Load documents from the specified folder using PyPDFDirectoryLoader.
    """
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    doc_loader = PyPDFDirectoryLoader(data_folder)
    docs = doc_loader.load()
    return docs

def process_documents_from_folder(folder_path, vectorstore, embeddings_model):
    """
    Process all documents in the specified folder, embed them, and add them to the vector store.
    """
    docs = load_docs(folder_path)
    for doc in docs:
        document_text = doc.page_content  # Assuming `page_content` contains the text of the document
        added = add_document(vectorstore, document_text, embeddings_model)
        if added:
            print(f"Added document: {doc.metadata['source']}")
        else:
            print(f"Document already exists: {doc.metadata['source']}")
