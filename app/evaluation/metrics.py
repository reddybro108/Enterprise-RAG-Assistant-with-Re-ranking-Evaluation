def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int = 5):
    recall_scores = []
    rr_scores = []

    for query_id, retrieved_ids in run.items():
        relevant_ids = {doc_id for doc_id, score in qrels.get(query_id, {}).items() if score > 0}
        recall_scores.append(recall_at_k(retrieved_ids, relevant_ids, k))
        rr_scores.append(reciprocal_rank(retrieved_ids[:k], relevant_ids))

    count = max(len(run), 1)
    return {
        f"recall@{k}": sum(recall_scores) / count,
        "mrr": sum(rr_scores) / count,
    }
