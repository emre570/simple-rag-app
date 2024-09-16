import os
import uuid
from indexer import split_docs
from embedder import call_embed_model
from retriever import retrieve_docs
from chain_handler import setup_chain
from session_handler import get_session_history, save_session_history
from docs_db_handler import init_db, add_db_docs, load_docs
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

session_id = str(uuid.uuid4())

current_directory = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_directory, "data")
db_path = os.path.join(current_directory, "db")

docs = load_docs(data_folder)

chunks = split_docs(docs)

embed_model_name = "sentence-transformers/all-MiniLM-L12-v2"
embeddings_model = call_embed_model(embed_model_name)

vectorstore = init_db(chunks, embeddings_model, db_path, embeddings_model)

add_db_docs(vectorstore, data_folder, db_path, embeddings_model)

chat_history = get_session_history(session_id)

while True:
    question = input("\n Enter your question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
        
    retriever = retrieve_docs(question, vectorstore, similar_docs_count = 5, see_content=False)
    rag_chain = setup_chain("llama3", retriever)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda _: chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    answer = ""
    for chunk in conversational_rag_chain.stream(
        {"input": question},
        config={
            "configurable": {"session_id": session_id}
        },
    ):
        if 'answer' in chunk:
            print(chunk['answer'], end="", flush=True)
            answer += chunk['answer']
            
    chat_history.add_user_message(question)
    chat_history.add_ai_message(answer)
    
    save_session_history(session_id)
    
for message in st.session_state.conversation:
        if message['role'] == 'human':
            with st.chat_message("human"):
                st.write(message['content'])
        elif message['role'] == 'ai':
            with st.chat_message("assistant"):
                st.write(message['content'])