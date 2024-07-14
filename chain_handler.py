from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def setup_chain(model_name, retriever):
    llm = ChatOllama(model=model_name)
    
    prompt = hub.pull("rlm/rag-prompt")
    
    #example_messages = prompt.invoke({"context": "filler context", "question": "filler question"}).to_messages()
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain