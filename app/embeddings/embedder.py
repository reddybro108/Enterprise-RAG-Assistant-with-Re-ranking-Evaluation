from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    embeddings = model.encode(texts)

    # Ensure numpy + float32
    embeddings = np.array(embeddings).astype("float32")

    # Ensure 2D shape
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)

    print("Embedding shape:", embeddings.shape)

    return embeddings