import os

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "false").lower() == "true"
_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            local_files_only=HF_LOCAL_FILES_ONLY,
        )
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    return embeddings


def get_embedding_dimension(embeddings: np.ndarray) -> int:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")
    return embeddings.shape[1]
