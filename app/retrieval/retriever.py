from app.embeddings.embedder import embed_texts


def retrieve(query, vectorstore, k=5):
    query_embedding = embed_texts([query])
    return vectorstore.search(query_embedding, k)
