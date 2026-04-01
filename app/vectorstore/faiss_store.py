import faiss
import numpy as np

class FAISSStore:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.texts = []
        self.metadatas = []

    def add(self, texts, embeddings, metadatas=None):
        vectors = np.array(embeddings).astype("float32")

        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        if metadatas is None:
            metadatas = [{} for _ in texts]

        self.index.add(vectors)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def search(self, query_embedding, k=5):
        if self.index.ntotal == 0:
            return []

        query_embedding = np.array(query_embedding).astype("float32")

        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        similarities, indices = self.index.search(query_embedding, k)

        results = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if 0 <= idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "score": float(similarity),
                    "metadata": self.metadatas[idx]
                })

        return results
