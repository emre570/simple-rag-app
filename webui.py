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

session_id = str(uuid.uuid4())

current_directory = os.path.dirname(os.path.abspath(__file__))
sessions_folder = os.path.join(current_directory, "sessions")
data_folder = os.path.join(current_directory, "data")
db_path = os.path.join(current_directory, "db")

st.title("RAG App Web UI")

#-----  CHAT INTERFACE  -----#
# Use session state to store conversation
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

prompt = st.chat_input("Say something")
if prompt:
    st.session_state.conversation.append({'role': 'human', 'content': prompt})
    st.session_state.conversation.append({'role': 'ai', 'content': "Hey!"})  # simulate AI response

for message in st.session_state.conversation:
    if message['role'] == 'human':
        with st.chat_message("human"):
            st.write(message['content'])
    elif message['role'] == 'ai':
        with st.chat_message("assistant"):
            st.write(message['content'])

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
                
            docs = load_docs(data_folder)
            chunks = split_docs(docs)
            embed_model_name = "sentence-transformers/all-MiniLM-L12-v2"
            embeddings_model = call_embed_model(embed_model_name)
            vectorstore = init_db(chunks, embeddings_model, db_path, embeddings_model)
            add_db_docs(vectorstore, data_folder, db_path, embeddings_model)
            
            st.success("Files uploaded and saved successfully!")

    with st.container():
        st.header("Previous Conversations")
        for i in range(len(json_datas)):
            if st.button(f"Show Conversation {i+1}"):
                st.session_state.conversation = json_datas[i]

    # Move this to the bottom of the sidebar
    with st.container():
        st.header("Start a New Conversation")
        if st.button("Start New Conversation"):
            st.session_state.conversation = []  # Clear the current conversation

# Display conversation
if 'selected_conversation_index' in st.session_state:
    st.subheader(f"Conversation {st.session_state.selected_conversation_index+1}")
    for j, msg in enumerate(json_datas[st.session_state.selected_conversation_index]):
        role = msg["role"]
        content = msg["content"]
        if role == "human":
            with st.chat_message("user"):
                st.write(content)
        elif role == "ai":
            with st.chat_message("ai"):
                st.write(content)