# Simple RAG App

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

### Installation

1. Clone the repository:

   `git clone https://github.com/yourusername/rag-conversational-app.git cd rag-conversational-app`
2. Install the required packages:

   `pip install -r requirements.txt`

### Usage

1. Place your PDF file(s) in the `data` folder.
2. Run the application:

   `python [path_to_app_folder]/app.py`
3. Enter your questions when prompted. Type 'exit' to quit the application.

## WIP Features

- **Conversation Memory:** The app will remember previous interactions during runtime for better context handling.
- **Web UI:** A web-based user interface for easier interaction.
- **Model and Database Selection:** Ability to select different LLMs and vector databases based on user preference.
- **Support for Multiple Document Types:** Extend functionality to work with Powerpoint slides, markdown files, text files, and more.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## Contact

For any questions or suggestions, please open an issue in the repository.