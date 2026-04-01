import os

from fastapi import FastAPI

from app.api.routes import get_pipeline, router


app = FastAPI(
    title="Enterprise RAG Assistant",
    description="Open-source RAG assistant with reranking and evaluation.",
    version="1.0.0",
)

app.include_router(router)


@app.on_event("startup")
def warm_up_pipeline():
    if os.getenv("RAG_WARMUP_ON_START", "true").lower() != "true":
        return

    try:
        get_pipeline()
    except Exception:
        # Keep the API responsive so /api/health can report the startup issue.
        pass


@app.get("/")
def healthcheck():
    return {"status": "ok", "message": "Enterprise RAG Assistant is running"}
