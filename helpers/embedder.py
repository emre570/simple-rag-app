from langchain_huggingface import HuggingFaceEmbeddings

def call_embed_model(model_name):
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings_model