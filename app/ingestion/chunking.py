from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_documents(corpus):
    documents = []

    # Convert BEIR corpus → LangChain Document format
    for doc_id in corpus:
        text = corpus[doc_id]["text"]
        title = corpus[doc_id].get("title", "")
        source_path = corpus[doc_id].get("source_path", "")

        documents.append(
            Document(
                page_content=text,
                metadata={"doc_id": doc_id, "title": title, "source_path": source_path}
            )
        )

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks
