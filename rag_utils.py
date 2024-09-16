import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

def load_docs(data_folder):
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    doc_loader = PyPDFDirectoryLoader(data_folder)
    
    docs = doc_loader.load()
    return docs

def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=80,
                                                   length_function=len,
                                                   is_separator_regex=False)
    return text_splitter.split_text(docs)

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

def add_db_docs(vectorstore, data_path, db_path, embeddings_model):
    """
    Load documents from the folder, check if they exist in the vectorstore, and add them if they don't.
    """
    documents = load_docs(data_path)
    #chunks = split_docs(documents)
    for document in documents:
        content = document.page_content
        embedding = embeddings_model.embed_query(content)
        result = vectorstore.similarity_search_by_vector(embedding, k=3)
        if not result:
            print("This content does not exist in vector database. Adding the content.")
            chunks = split_docs(content)
            vectorstore.add_texts(chunks)
    vectorstore.save_local(db_path)
    

def call_embed_model(model_name):
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings_model

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# CALL RETRIEVER
#If you want to see retrieved document content, set 'see_content' to True.
def retrieve_docs(question, vector_store, similar_docs_count, see_content:False):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": similar_docs_count})
    retrieved_docs = retriever.invoke(question)
    
    if(see_content):
        for i in range(similar_docs_count):
            print(retrieved_docs[i].page_content)
            
    return retriever

def setup_chain(model_name, retriever):
    llm = ChatOllama(model=model_name, base_url="http://127.0.0.1:11434", keep_alive=-1)
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    ### Answer question ###
    system_prompt = (
        "You are an assistant named Benedict. Your task is question-answering."
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say the user that you "
        "don't know and user must search the web."
        "Keep the answer concise and short."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain