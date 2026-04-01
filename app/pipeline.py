import os
from dataclasses import dataclass, field
from time import perf_counter

from app.embeddings.embedder import embed_texts, get_embedding_dimension
from app.ingestion.chunking import split_documents
from app.ingestion.loader import load_fiqa_corpus, load_pdf_corpus
from app.reranker.rerank import rerank_results
from app.retrieval.retriever import retrieve
from app.vectorstore.faiss_store import FAISSStore


@dataclass
class PipelineStatus:
    is_ready: bool = False
    is_initializing: bool = False
    error: str | None = None
    dataset_type: str | None = None
    document_count: int = 0
    chunk_count: int = 0
    init_duration_seconds: float | None = None
    warnings: list[str] = field(default_factory=list)


class RAGPipeline:
    def __init__(self):
        self.vectorstore = None
        self.status = PipelineStatus()

    @property
    def is_ready(self) -> bool:
        return self.status.is_ready

    @property
    def error(self) -> str | None:
        return self.status.error

    @property
    def warnings(self) -> list[str]:
        return list(self.status.warnings)

    def get_status(self) -> dict[str, object]:
        return {
            "is_ready": self.status.is_ready,
            "is_initializing": self.status.is_initializing,
            "error": self.status.error,
            "dataset_type": self.status.dataset_type,
            "document_count": self.status.document_count,
            "chunk_count": self.status.chunk_count,
            "init_duration_seconds": self.status.init_duration_seconds,
            "warnings": list(self.status.warnings),
        }

    def initialize(self):
        if self.is_ready:
            return

        if self.status.is_initializing:
            raise RuntimeError("Pipeline initialization is already in progress.")

        dataset_type = os.getenv("DATASET_TYPE", "pdf").strip().lower()
        self.status.is_initializing = True
        self.status.dataset_type = dataset_type
        self.status.error = None
        self.status.warnings = []
        started_at = perf_counter()

        try:
            if dataset_type == "pdf":
                corpus, warnings = load_pdf_corpus()
            elif dataset_type == "fiqa":
                corpus = load_fiqa_corpus()
                warnings = []
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
            self.status.is_ready = True
            self.status.document_count = len(corpus)
            self.status.chunk_count = len(texts)
            self.status.warnings = warnings
        except Exception as exc:
            self.vectorstore = None
            self.status.is_ready = False
            self.status.error = str(exc)
            raise
        finally:
            self.status.is_initializing = False
            self.status.init_duration_seconds = round(perf_counter() - started_at, 3)

    def query(self, query: str, top_k: int = 5, rerank_top_n: int = 3):
        if not self.is_ready or self.vectorstore is None:
            if self.error:
                raise RuntimeError(f"Pipeline is not initialized: {self.error}")
            raise RuntimeError("Pipeline is not initialized")

        retrieved = retrieve(query, self.vectorstore, top_k)
        reranked = rerank_results(query, retrieved, top_n=rerank_top_n)
        context = "\n\n".join(item["text"] for item in reranked)

        return {"results": reranked, "context": context}
