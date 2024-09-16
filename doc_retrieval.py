import os
from indexer import split_docs
from embedder import call_embed_model
from retriever import retrieve_docs
from docs_db_handler import init_db, add_db_docs, load_docs
from langchain_core.tools import tool

@tool
def make_doc_retrieval(question: str):
    """
    Tool Name: make_doc_retrieval

    Description:
    The make_doc_retrieval tool is designed to retrieve relevant documents based on a given question. It leverages a pre-existing database of documents and utilizes a sentence transformer model to generate embeddings for efficient similarity search. The tool performs the following steps:

    - Data Loading: Loads all documents from a specified data folder.
    - Document Splitting: Splits the loaded documents into manageable chunks for more granular embedding.
    - Embedding Model Setup: Initializes the specified embedding model (sentence-transformers/all-MiniLM-L12-v2) to create vector representations of the document chunks.
    - Vector Store Initialization: Initializes a vector store to manage and query the document embeddings.
    - Database Management: Adds new documents to the vector store while avoiding duplicates.
    - Document Retrieval: Searches the vector store for the top 5 most similar document chunks related to the input question, without displaying the full content.
    
    Inputs:
    question (str): The user's question or query that specifies the information they are seeking.
    
    Output:
    A list of document chunks that are most relevant to the user's question. The tool returns these chunks based on their similarity to the input query.
    
    Usage Context:
    Use this tool when you need to fetch information from a set of documents based on a user's question. This tool is ideal for providing detailed answers or sourcing relevant content from a document database.
    """
    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_directory, "data")
    db_path = os.path.join(current_directory, "db")

    docs = load_docs(data_folder)

    chunks = split_docs(docs)

    embed_model_name = "sentence-transformers/all-MiniLM-L12-v2"
    embeddings_model = call_embed_model(embed_model_name)

    vectorstore = init_db(chunks, embeddings_model, db_path, embeddings_model)

    add_db_docs(vectorstore, data_folder, db_path, embeddings_model)
    
    retrieved_content = retrieve_docs(question, vectorstore, similar_docs_count = 5, see_content=False)
    
    print(retrieved_content)
    print(type(retrieved_content))
    
    return retrieved_content