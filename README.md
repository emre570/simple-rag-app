# Simple RAG App

## Announcements

* The LLM now remembers your questions!
* Sessions will be saved including your conversations. See "sessions" folder after exiting app.

## Overview

This project is a part of my self-development Retrieval-Augmented Generation (RAG) application that allows users to ask questions about the content of a PDF files placed in folder. The app uses advanced NLP models and techniques to provide accurate answers based on the document's content. The application leverages Ollama, Llama 3-8B, LangChain, and FAISS for its operations.

## Features

- **Ask Questions About PDFs:** Simply place a PDF file in the `data` folder and start asking questions about its content.
- **Advanced NLP Models:** Utilizes Ollama and Llama 3-8B for generating responses.
- **Efficient Document Retrieval:** Uses LangChain and FAISS for efficient document retrieval and processing.
- **Duplicate Handling:** The app checks the vector database for duplicates and avoids adding them if they already exist.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages (see `requirements.txt`)
- Ollama Installation with Llama 3 installed

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/emre570/simple-rag-app.git
   cd simple-rag-app
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Install LLM using Ollama:
   ```bash
   ollama pull llama3
   ```

### Usage

1. Place your PDF file(s) in the `data` folder.
2. Run the application:

   ```bash
   python [path_to_app_folder]/app.py
   ```
3. Enter your questions when prompted. Type 'exit' to quit the application.

## WIP Features

- [ ] **Web UI:** A web-based user interface for easier interaction.
- [X] **Conversation Memory:** The app will remember previous interactions during runtime for better context handling.
- [ ] **More Memory:** You will be able to navigate between sessions and continue the conversation.
- [ ] **Model Selection:** Ability to select different LLMs based on user preference.
- [ ] **Support for Multiple Document Types:** Extend functionality to work with Powerpoint slides, markdown files, text files, and more.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. For changes or see any mistakes, please open an issue first to discuss what you would like to change.

## Contact

For any questions or suggestions, please open an issue in the repository.
