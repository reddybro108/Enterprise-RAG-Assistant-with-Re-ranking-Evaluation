from app.evaluation.metrics import evaluate_retrieval
from app.vectorstore.faiss_store import FAISSStore


def test_faiss_store_returns_ranked_results():
    store = FAISSStore(dim=3)
    store.add(
        texts=["hr leave policy", "finance reimbursement form"],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        metadatas=[
            {"doc_id": "d1"},
            {"doc_id": "d2"},
        ],
    )

    results = store.search([[1.0, 0.0, 0.0]], k=2)

    assert len(results) == 2
    assert results[0]["metadata"]["doc_id"] == "d1"
    assert results[0]["score"] >= results[1]["score"]


def test_evaluate_retrieval_returns_expected_metrics():
    run = {"q1": ["d1", "d3", "d2"]}
    qrels = {"q1": {"d2": 1, "d1": 1}}

    metrics = evaluate_retrieval(run, qrels, k=2)

    assert metrics["recall@2"] == 0.5
    assert metrics["mrr"] == 1.0
