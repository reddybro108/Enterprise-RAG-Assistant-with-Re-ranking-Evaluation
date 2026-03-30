from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def split_documents(corpus):
    documents = []

    # Convert BEIR corpus → LangChain Document format
    for doc_id in corpus:
        text = corpus[doc_id]["text"]
        title = corpus[doc_id].get("title", "")

        documents.append(
            Document(
                page_content=text,
                metadata={"doc_id": doc_id, "title": title}
            )
        )

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.split_documents(documents)