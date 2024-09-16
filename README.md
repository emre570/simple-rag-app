# Simple RAG App

## Overview

This project is a part of my self-development Retrieval-Augmented Generation (RAG) application that allows users to ask questions about the content of a PDF files placed in folder. The app uses techniques to provide accurate answers based on the document's content. The application leverages Ollama, Llama 3-8B, LangChain, and FAISS for its operations.

## Features

- **Ask Questions About PDFs:** Simply place a PDF file in the `data` folder or upload from Web UI and start asking questions about its content.
- **LLM Models:** Utilizes Ollama and Llama 3-8B for generating responses.
- **Efficient Document Retrieval:** Uses LangChain and FAISS for efficient document retrieval and processing (PDF Only).
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
3. Download Ollama and install LLM using Ollama:
   ```bash
   ollama pull llama3
   ```

### Usage

1. Place your PDF file(s) in the `data` folder or upload it from Web UI.
2. Run the application:

   ```bash
   streamlit run [path_to_app_folder]/webui.py
   ```

   Alternatively, you can open `webui.bat` file.
3. Enter your questions when prompted. Type 'exit' to quit the application.

## WIP Features

- [X] **Web UI:** A web-based user interface for easier interaction.
- [X] **Conversation Memory:** The app will remember previous interactions during runtime for better context handling.
- [ ] **Support for Multiple Document Types:** Extend functionality to work with Powerpoint slides, markdown files, text files, and more.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. For changes or see any mistakes, please open an issue first to discuss what you would like to change.

## Contact

For any questions or suggestions, please open an issue in the repository.
