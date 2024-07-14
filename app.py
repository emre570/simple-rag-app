import os
import getpass
from document_handler import load_docs
from indexer import split_docs
from embedder import call_embed_model
from vector_db_handler import init_faiss
from retriever import retrieve_docs
from chain_handler import setup_chain

os.environ["LANGCHAIN_API_KEY"] = "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"

data_folder = "data"
docs = load_docs(data_folder)

chunks = split_docs(docs)

embed_model_name = "sentence-transformers/all-MiniLM-L12-v2"
embeddings_model = call_embed_model(embed_model_name)

vectorstore = init_faiss(chunks, embeddings_model)

similar_docs_count = 5
question = "When did Roman Empire split to two and who were their rulers?"

retriever = retrieve_docs(question, vectorstore, similar_docs_count, see_content=False)

rag_chain = setup_chain("llama3", retriever)

for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)