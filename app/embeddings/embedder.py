import os
from inspect import signature

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "false").lower() == "true"
_model = None


def _embedding_model_kwargs() -> dict:
    init_params = signature(SentenceTransformer.__init__).parameters
    if "local_files_only" in init_params:
        return {"local_files_only": HF_LOCAL_FILES_ONLY}
    return {}


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            **_embedding_model_kwargs(),
        )
    return _model


def embed_texts(texts):
    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        batch_size=32,   # 🔥 improves throughput
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings.astype("float32")


def get_embedding_dimension(embeddings: np.ndarray) -> int:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")
    return embeddings.shape[1]

if __name__ == "__main__":
    texts = ["Test embedding"]
    embeddings = embed_texts(texts)

    print("Shape:", embeddings.shape)
    print("Sample:", embeddings[0][:5])    
