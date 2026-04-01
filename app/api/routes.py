from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.generation.llm import generate_response
from app.generation.prompt import build_prompt
from app.pipeline import RAGPipeline

router = APIRouter(prefix="/api", tags=["rag"])


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User question")
    top_k: int = Field(default=5, ge=1, le=20)
    rerank_top_n: int = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    query: str
    answer: str
    contexts: list[dict[str, Any]]


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def ensure_pipeline_ready() -> RAGPipeline:
    pipeline = get_pipeline()
    if not pipeline.is_ready:
        pipeline.initialize()
    return pipeline


@router.get("/health")
def api_health():
    pipeline = get_pipeline()
    status: dict[str, Any] = {
        "status": "ok",
        "pipeline_ready": pipeline.is_ready,
        "pipeline": pipeline.get_status(),
    }

    return status


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    try:
        pipeline = ensure_pipeline_ready()
        result = pipeline.query(request.query, request.top_k, request.rerank_top_n)
        prompt = build_prompt(request.query, result["context"])
        answer = generate_response(prompt)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Pipeline is unavailable. Check /api/health for initialization details. "
                f"Reason: {exc}"
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    return QueryResponse(
        query=request.query,
        answer=answer,
        contexts=result["results"],
    )
