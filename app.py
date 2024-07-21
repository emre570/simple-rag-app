import os
from document_handler import load_docs
from indexer import split_docs
from embedder import call_embed_model
from vector_db_handler import init_faiss, process_documents_from_folder, save_faiss
from retriever import retrieve_docs
from chain_handler import setup_chain

# Define the folder paths
current_directory = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_directory, "data")
db_path = os.path.join(current_directory, "db")

# Load documents from the data folder
docs = load_docs(data_folder)

# Debug: Check loaded documents
print(f"Loaded {len(docs)} documents")

# Split documents into chunks
chunks = split_docs(docs)

# Debug: Check document chunks
print(f"Split into {len(chunks)} chunks")

# Initialize the embeddings model
embed_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings_model = call_embed_model(embed_model_name)

# Debug: Check embeddings model
print(f"Embedding model: {embed_model_name}")

# Initialize or load the vector store
vectorstore = init_faiss(chunks, embeddings_model, db_path)

# Process documents from the folder and add to vector store
process_documents_from_folder(data_folder, vectorstore, embeddings_model)

# Save the updated vector store
save_faiss(vectorstore, db_path)

# Query parameters
similar_docs_count = 8
question = "Who are the first and last ruler of Roman Empire?"

# Retrieve documents based on the question
retriever = retrieve_docs(question, vectorstore, similar_docs_count, see_content=False)

# Set up the RAG chain
rag_chain = setup_chain("llama3", retriever)

# Stream and print the response from the RAG chain
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)
