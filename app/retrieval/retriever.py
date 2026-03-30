from app.embeddings.embedder import embed_texts
from app.vectorstore.faiss_store import FAISSStore


def retrieve(query, vectorstore, k=5):
    query_embedding = embed_texts([query])
    return vectorstore.search(query_embedding, k)


if __name__ == "__main__":
    query = "What is RAG?"

    # Step 1: Get embedding dimension dynamically
    sample_embedding = embed_texts(["test"])
    dim = sample_embedding.shape[1]

    # Step 2: Initialize FAISS
    vs = FAISSStore(dim=dim)

    # Step 3: Sample corpus (replace with your dataset later)
    texts = [
        "RAG improves LLM accuracy using retrieval",
        "FAISS is a library for efficient similarity search",
        "Transformers are used in modern NLP models"
    ]

    embeddings = embed_texts(texts)

    # Step 4: Build index
    vs.add(texts, embeddings)

    # Step 5: Retrieve
    results = retrieve(query, vs, k=3)

    # Step 6: Output
    for i, r in enumerate(results):
        print(f"Rank {i+1}")
        print(f"Text: {r['text']}")
        print(f"Score: {r['score']}")
        print("-" * 40)