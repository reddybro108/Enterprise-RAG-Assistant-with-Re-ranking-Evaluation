import os
from threading import Thread

from fastapi import FastAPI

from app.api.routes import ensure_pipeline_ready, router


app = FastAPI(
    title="Enterprise RAG Assistant",
    description="Open-source RAG assistant with reranking and evaluation.",
    version="1.0.0",
)

app.include_router(router)


@app.on_event("startup")
def warm_up_pipeline():
    if os.getenv("RAG_WARMUP_ON_START", "false").lower() != "true":
        return

    def _warm_up():
        try:
            ensure_pipeline_ready()
        except Exception:
            # Keep the API responsive so /api/health and /api/query can report the issue.
            pass

    Thread(target=_warm_up, daemon=True).start()


@app.get("/")
def healthcheck():
    return {"status": "ok", "message": "Enterprise RAG Assistant is running"}
