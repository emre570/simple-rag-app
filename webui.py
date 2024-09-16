import os
import time
import json
import uuid
import streamlit as st
from indexer import split_docs
from embedder import call_embed_model
from retriever import retrieve_docs
from chain_handler import setup_chain
from session_handler import get_session_history, save_session_history
from docs_db_handler import init_db, add_db_docs, load_docs
from langchain_core.runnables.history import RunnableWithMessageHistory

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
session_id = st.session_state.session_id

current_directory = os.path.dirname(os.path.abspath(__file__))
sessions_folder = os.path.join(current_directory, "sessions")
data_folder = os.path.join(current_directory, "data")
db_path = os.path.join(current_directory, "db")

embed_model_name = "sentence-transformers/all-MiniLM-L12-v2"

st.title("RAG App Web UI")

#-----  SIDEBAR - FILE UPLOAD - PREVIOUS CONVERSATIONS  -----#
jsons = [f for f in os.listdir(sessions_folder) if f.endswith('.json')]
json_datas = []

for file in jsons:
    with open(os.path.join(sessions_folder, file), 'r') as json_file:
        data = json.load(json_file)
        json_datas.append(data) 

with st.sidebar:
    with st.container():
        st.header("File Upload")

        # Check if 'uploaded_files' is already in session state
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = None
        
        uploaded_files = st.file_uploader(
            "Upload your documents from here:", accept_multiple_files=True, key="file_uploader"
        )
        
        if uploaded_files is not None and uploaded_files != st.session_state.uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            with st.spinner("Loading..."):
                # Iterate over the uploaded files and save them to the data folder
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(data_folder, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                time.sleep(3)
            
            st.success("Files uploaded and saved successfully!")

        # Add a button to start a new conversation
        if st.button("New Conversation"):
            st.session_state.session_id = str(uuid.uuid4())

            # Generate a new session_id
            session_id = st.session_state.session_id

            # Clear the conversation history in session state
            st.session_state.conversation = []

            # Display a message confirming the reset
            st.success("New conversation started!")

#-----  MAIN APP LOGIC  -----#
docs = load_docs(data_folder)
chunks = split_docs(docs)
print("docs vector")
embeddings_model = call_embed_model(embed_model_name)
vectorstore = init_db(chunks, embeddings_model, db_path, embeddings_model)
add_db_docs(vectorstore, data_folder, db_path, embeddings_model)
                
#-----  CHAT INTERFACE  -----#
# Use session state to store conversation
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Display the conversation history
for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.write(message["message"])

chat_history = get_session_history(session_id)

print("prompt go")
prompt = st.chat_input("Say something")

if prompt:
    print("prompt given")
    
    with st.chat_message("human"):
        st.write(prompt)
        st.session_state.conversation.append({"role": "human", "message": prompt})
    
    retriever = retrieve_docs(prompt, vectorstore, similar_docs_count=5, see_content=False)
    rag_chain = setup_chain("llama3", retriever)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda _: chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    with st.chat_message("ai"):
        answer = ""
        placeholder = st.empty()
              
        for response_chunk in conversational_rag_chain.stream(
            {"input": prompt},
            config={
                "configurable": {"session_id": session_id}
            },
        ):
            if 'answer' in response_chunk:
                answer += response_chunk["answer"]
                placeholder.write(answer)
        st.session_state.conversation.append({"role": "ai", "message": answer})
        save_session_history(session_id)
