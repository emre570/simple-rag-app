# LOAD THE DOCS FROM FOLDER
import os
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

def load_docs(data_folder):
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    doc_loader = PyPDFDirectoryLoader(data_folder)
    return doc_loader.load()

