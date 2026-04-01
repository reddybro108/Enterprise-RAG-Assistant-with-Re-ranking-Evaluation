import os

from app.embeddings.embedder import embed_texts, get_embedding_dimension
from app.ingestion.chunking import split_documents
from app.ingestion.loader import load_fiqa_corpus, load_pdf_corpus
from app.reranker.rerank import rerank_results
from app.retrieval.retriever import retrieve
from app.vectorstore.faiss_store import FAISSStore


class RAGPipeline:
    def __init__(self):
        self.vectorstore = None
        self.is_ready = False

    def initialize(self):
        if self.is_ready:
            return

        dataset_type = os.getenv("DATASET_TYPE", "pdf").strip().lower()

        if dataset_type == "pdf":
            corpus = load_pdf_corpus()
        elif dataset_type == "fiqa":
            corpus = load_fiqa_corpus()
        else:
            raise ValueError(
                "Unsupported DATASET_TYPE. Use 'pdf' for local PDFs or 'fiqa' for the FIQA dataset."
            )

        documents = split_documents(corpus)
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        embeddings = embed_texts(texts)
        dim = get_embedding_dimension(embeddings)

        self.vectorstore = FAISSStore(dim=dim)
        self.vectorstore.add(texts, embeddings, metadatas)
        self.is_ready = True

    def query(self, query: str, top_k: int = 5, rerank_top_n: int = 3):
        if not self.is_ready or self.vectorstore is None:
            raise RuntimeError("Pipeline is not initialized")

        retrieved = retrieve(query, self.vectorstore, top_k)
        reranked = rerank_results(query, retrieved, top_n=rerank_top_n)
        context = "\n\n".join(item["text"] for item in reranked)

        return {"results": reranked, "context": context}
