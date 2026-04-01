# Enterprise RAG Assistant with Reranking and Evaluation

An enterprise-oriented Retrieval-Augmented Generation (RAG) application built with FastAPI and Streamlit. The project ingests document corpora, retrieves relevant chunks with dense embeddings and FAISS, reranks results with a cross-encoder, and generates grounded answers using a local LLM served through Ollama.

This repository is designed for practical document Q&A workflows where reliability, traceability, and operational visibility matter as much as raw model output quality.

## Overview

The system provides:

- A FastAPI backend for health checks and query execution
- A Streamlit frontend for interactive question answering
- PDF ingestion with optional OCR fallback for scanned pages
- FAISS-based vector retrieval
- Cross-encoder reranking for better relevance
- Retrieval evaluation utilities and tests
- Startup health visibility for pipeline readiness, warnings, and initialization failures

## Key Features

- Enterprise-friendly API layer with clear health and query endpoints
- Retrieval pipeline initialization tracking with readiness and warning metadata
- Graceful handling of problematic PDFs during corpus loading
- Support for two dataset modes:
  - `pdf` for local PDF document folders
  - `fiqa` for FIQA benchmark-style local datasets
- Local LLM generation via Ollama
- Streamlit UI with backend health awareness

## Architecture

The application follows a layered structure:

1. `ui.py`
   Streamlit frontend that calls the FastAPI backend.
2. `app/main.py`
   FastAPI application entry point and startup warm-up.
3. `app/api/routes.py`
   REST endpoints for health checks and user queries.
4. `app/pipeline.py`
   End-to-end RAG orchestration for ingestion, embedding, indexing, retrieval, and reranking.
5. `app/ingestion/*`
   Corpus loading, PDF parsing, OCR fallback, and chunking.
6. `app/embeddings/*`
   Sentence-transformer embedding model loading and text embedding.
7. `app/vectorstore/*`
   FAISS index creation and similarity search.
8. `app/retrieval/*`
   Query embedding and nearest-neighbor retrieval.
9. `app/reranker/*`
   Cross-encoder reranking of retrieved contexts.
10. `app/generation/*`
    Prompt construction and LLM answer generation.
11. `app/evaluation/*`
    Retrieval evaluation metrics.

## Project Structure

```text
.
|-- app/
|   |-- api/
|   |-- embeddings/
|   |-- evaluation/
|   |-- generation/
|   |-- ingestion/
|   |-- reranker/
|   |-- retrieval/
|   |-- vectorstore/
|   |-- main.py
|   `-- pipeline.py
|-- data/
|   `-- rag_data/
|-- datasets/
|   `-- fiqa/
|-- tests/
|-- ui.py
|-- requirements.txt
`-- README.md
```

## Technology Stack

- Python 3.11+
- FastAPI
- Uvicorn
- Streamlit
- Sentence Transformers
- FAISS
- PyPDF
- pypdfium2
- RapidOCR
- Ollama with `qwen2.5:7b` as the default local model
- Pytest
- Docker and Docker Compose
- GitHub Actions

## Prerequisites

Before running the project, make sure you have:

- Python installed
- A virtual environment available at `.\eraenv`
- Required Python dependencies installed
- Ollama installed and running locally if you want answer generation
- At least one supported dataset prepared:
  - PDF files under `data/rag_data`
  - Or FIQA files under `datasets/fiqa`

## Installation

### 1. Create and activate a virtual environment

```powershell
python -m venv eraenv
.\eraenv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Prepare Ollama

Make sure Ollama is running and the default model is available locally.

Example:

```powershell
ollama pull qwen2.5:7b
ollama serve
```

By default, the app expects:

- `OLLAMA_URL=http://localhost:11434/api/generate`
- `OLLAMA_MODEL=qwen2.5:7b`

## Dataset Setup

### Option 1: Local PDF dataset

Place PDF files anywhere under:

```text
data/rag_data/
```

Notes:

- The loader scans subfolders recursively
- Hidden paths are skipped
- If a PDF page has no extractable text, OCR can be used when enabled
- Corrupt or unsupported PDFs are skipped and reported as warnings

### Option 2: FIQA dataset

Place the FIQA files under:

```text
datasets/fiqa/
```

Expected files include:

- `corpus.jsonl`
- `queries.jsonl`
- `qrels/test.tsv`

## Configuration

The application is configured through environment variables.

| Variable | Default | Description |
|---|---|---|
| `RAG_API_URL` | `http://127.0.0.1:8001` | Base URL used by Streamlit to call FastAPI |
| `DATASET_TYPE` | `pdf` | Dataset mode: `pdf` or `fiqa` |
| `ENABLE_PDF_OCR` | `true` | Enables OCR fallback for scanned PDF pages |
| `PDF_OCR_RENDER_SCALE` | `2.0` | Rendering scale used before OCR |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `HF_LOCAL_FILES_ONLY` | `false` | Restricts Hugging Face loading to local files only |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama generation endpoint |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ollama model name |
| `RAG_WARMUP_ON_START` | `false` | Enables optional background pipeline warm-up during FastAPI startup |

### Example PowerShell session

```powershell
$env:DATASET_TYPE="pdf"
$env:OLLAMA_MODEL="qwen2.5:7b"
$env:RAG_WARMUP_ON_START="false"
$env:RAG_API_URL="http://127.0.0.1:8001"
```

## How to Run

Run the services in this exact sequence.

### 1. Start the FastAPI backend

```powershell
.\eraenv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

FastAPI will start on:

```text
http://127.0.0.1:8001
```

### 2. Verify backend health

Open:

```text
http://127.0.0.1:8001/api/health
```

You should see a payload that includes:

- `status`
- `pipeline_ready`
- `pipeline`

The `pipeline` block reports:

- readiness
- initialization errors
- dataset type
- document and chunk counts
- skipped file warnings

By default, the health endpoint returns quickly and does not force heavy model or PDF initialization.

### 3. Start the Streamlit UI

```powershell
.\eraenv\Scripts\python.exe -m streamlit run ui.py
```

Streamlit usually opens at:

```text
http://localhost:8501
```

## Running with Docker

The repository includes a shared [Dockerfile](D:\Enterprise%20RAG%20Assistant%20with%20Reranking%20and%20Evaluation\Dockerfile) and a [docker-compose.yml](D:\Enterprise%20RAG%20Assistant%20with%20Reranking%20and%20Evaluation\docker-compose.yml) for the API and UI.

### 1. Make sure Ollama is running on the host

```powershell
ollama serve
ollama pull qwen2.5:7b
```

### 2. Start the containers

```powershell
docker compose up --build
```

This starts:

- FastAPI on `http://localhost:8001`
- Streamlit on `http://localhost:8501`

Notes:

- The API container connects to Ollama through `http://host.docker.internal:11434/api/generate`
- `data/` and `datasets/` are mounted into the API container
- If you are on Linux, `host.docker.internal` may need Docker host gateway support enabled

## API Endpoints

### `GET /`

Basic service status endpoint.

### `GET /api/health`

Backend and pipeline health endpoint.

Example response:

```json
{
  "status": "ok",
  "pipeline_ready": true,
  "pipeline": {
    "is_ready": true,
    "is_initializing": false,
    "error": null,
    "dataset_type": "pdf",
    "document_count": 12,
    "chunk_count": 148,
    "init_duration_seconds": 9.437,
    "warnings": []
  }
}
```

### `POST /api/query`

Submit a user question to the RAG pipeline.

Request body:

```json
{
  "query": "What does the reimbursement policy say?",
  "top_k": 5,
  "rerank_top_n": 3
}
```

Response body:

```json
{
  "query": "What does the reimbursement policy say?",
  "answer": "Generated answer here",
  "contexts": [
    {
      "text": "Relevant chunk text",
      "score": 0.91,
      "rerank_score": 7.42,
      "metadata": {
        "title": "policy.pdf",
        "source_path": "C:/path/to/policy.pdf"
      }
    }
  ]
}
```

## Streamlit Usage

The UI provides:

- Backend reachability status
- Pipeline readiness and warning visibility
- Query input box
- Retrieval and reranking controls
- Generated answer display
- Source traceability with file paths
- Expandable retrieved context inspection

## Testing

Run the test suite with:

```powershell
.\eraenv\Scripts\python.exe -m pytest
```

Current tests cover:

- FAISS result ranking behavior
- Retrieval evaluation metrics

## CI/CD

The repository includes a GitHub Actions workflow at [.github/workflows/ci.yml](D:\Enterprise%20RAG%20Assistant%20with%20Reranking%20and%20Evaluation\.github\workflows\ci.yml).

The workflow currently:

- Checks out the repository
- Sets up Python 3.11
- Installs dependencies
- Runs the test suite
- Builds the Docker image

This gives you a solid CI baseline for pull requests and branch pushes. You can extend it later with:

- image publishing to Docker Hub or GHCR
- deployment to a VM, container service, or Kubernetes
- security scanning
- linting and formatting checks

## Troubleshooting

### FastAPI starts but `/api/health` shows `pipeline_ready: false`

Likely causes:

- No PDFs found under `data/rag_data`
- FIQA dataset files missing under `datasets/fiqa`
- Hugging Face models not available or still downloading
- OCR dependencies missing
- A startup exception occurred during pipeline warm-up

What to do:

- Check the `pipeline.error` field in `/api/health`
- Confirm the dataset folder contains valid files
- Confirm required dependencies were installed successfully

### Streamlit shows "FastAPI is not reachable"

Likely causes:

- FastAPI is not running
- `RAG_API_URL` points to the wrong host or port

What to do:

- Start FastAPI first
- Verify `http://127.0.0.1:8001/api/health`
- Ensure `RAG_API_URL` matches the backend URL

### Query returns a 503 error

This usually means the pipeline is unavailable rather than the UI being broken.

What to do:

- Open `/api/health`
- Inspect the `pipeline.error` message
- Fix the underlying initialization issue before retrying

### PDF loading issues

Symptoms:

- Warnings about skipped files
- Missing content from scanned or corrupt PDFs

What to do:

- Check whether the PDF contains selectable text
- Keep `ENABLE_PDF_OCR=true` for scanned documents
- Remove or replace damaged PDFs

### LLM generation fails

Likely causes:

- Ollama is not running
- The configured model is not installed
- `OLLAMA_URL` or `OLLAMA_MODEL` is incorrect

What to do:

- Start Ollama
- Pull the configured model
- Verify the configured endpoint is reachable

## Operational Notes

- The pipeline is cached inside the FastAPI process
- With `RAG_WARMUP_ON_START=true`, initialization happens in a background thread during app startup
- The first startup may take time because models and embeddings need to load
- PDF ingestion can be expensive for large corpora or OCR-heavy documents

## Security and Production Considerations

For enterprise deployment, consider adding:

- Authentication and authorization
- Rate limiting
- Structured logging and monitoring
- Persistent vector index storage
- Background indexing jobs
- Model and dataset versioning
- CI/CD validation and deployment automation
- Containerization with Docker
- Reverse proxy and TLS termination

## Roadmap Ideas

- Persistent FAISS index serialization
- Incremental document ingestion
- Async background warm-up and indexing
- Better observability with logs and metrics
- User authentication and tenant isolation
- Evaluation dashboards
- Admin controls for corpus management

## License

Add your preferred license here before external distribution.

## Author Notes

This project is a solid base for an enterprise document intelligence assistant. The backend, UI, and pipeline are intentionally separated so the system can evolve into a larger internal platform with better indexing, monitoring, and access control over time.
