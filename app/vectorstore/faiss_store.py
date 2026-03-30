import faiss
import numpy as np

class FAISSStore:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, texts, embeddings):
        vectors = np.array(embeddings).astype("float32")

        # Ensure 2D
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query_embedding, k=5):
        query_embedding = np.array(query_embedding).astype("float32")

        # Ensure 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "score": float(dist)
                })

        return results