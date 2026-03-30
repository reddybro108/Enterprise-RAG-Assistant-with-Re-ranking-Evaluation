from app.ingestion.chunking import split_documents

documents = split_documents(corpus)

texts = []
doc_ids = []

for doc in documents:
    texts.append(doc.page_content)
    doc_ids.append(doc.metadata["doc_id"])