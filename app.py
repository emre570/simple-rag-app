import os
import getpass
from indexer import split_docs
from embedder import call_embed_model
from retriever import retrieve_docs
from chain_handler import setup_chain
from docs_db_handler import init_db, add_db_docs, load_docs

current_directory = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_directory, "data")
db_path = os.path.join(current_directory, "db")

docs = load_docs(data_folder)

chunks = split_docs(docs)

embed_model_name = "sentence-transformers/all-MiniLM-L12-v2"
embeddings_model = call_embed_model(embed_model_name)

vectorstore = init_db(chunks, embeddings_model, db_path, embeddings_model)

add_db_docs(vectorstore, 'data', embeddings_model)

similar_docs_count = 5
question = "Who were the first and last rulers of Roman Empire?"

retriever = retrieve_docs(question, vectorstore, similar_docs_count, see_content=False)

rag_chain = setup_chain("llama3", retriever)

for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)