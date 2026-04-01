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
    pipeline = RAGPipeline()
    pipeline.initialize()
    return pipeline


@router.get("/health")
def api_health():
    pipeline_ready = False
    try:
        pipeline_ready = get_pipeline().is_ready
    except Exception:
        pipeline_ready = False

    return {"status": "ok", "pipeline_ready": pipeline_ready}


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    try:
        pipeline = get_pipeline()
        result = pipeline.query(request.query, request.top_k, request.rerank_top_n)
        prompt = build_prompt(request.query, result["context"])
        answer = generate_response(prompt)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    return QueryResponse(
        query=request.query,
        answer=answer,
        contexts=result["results"],
    )
