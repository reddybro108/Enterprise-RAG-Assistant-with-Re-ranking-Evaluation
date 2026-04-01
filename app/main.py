from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="Enterprise RAG Assistant",
    description="Open-source RAG assistant with reranking and evaluation.",
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
def healthcheck():
    return {"status": "ok", "message": "Enterprise RAG Assistant is running"}
