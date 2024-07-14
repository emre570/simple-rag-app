# CALL RETRIEVER
def retrieve_docs(question, vector_store, similar_docs_count, see_content:False):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": similar_docs_count})
    retrieved_docs = retriever.invoke(question)

    print(f"{len(retrieved_docs)} similar documents found. If you want to see content, set 'see_content' to True.")
    
    if(see_content):
        for i in range(similar_docs_count):
            print(retrieved_docs[i].page_content)
            
    return retriever