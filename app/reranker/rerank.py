import os

from sentence_transformers import CrossEncoder


RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "false").lower() == "true"
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(
            RERANKER_MODEL_NAME,
            local_files_only=HF_LOCAL_FILES_ONLY,
        )
    return _reranker


def rerank_results(query: str, results: list[dict], top_n: int = 3):
    if not results:
        return []

    reranker = get_reranker()
    pairs = [[query, item["text"]] for item in results]
    scores = reranker.predict(pairs)

    reranked = []
    for score, item in zip(scores, results):
        updated = dict(item)
        updated["rerank_score"] = float(score)
        reranked.append(updated)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked[:top_n]
