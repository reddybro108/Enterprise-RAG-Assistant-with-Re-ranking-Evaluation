import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st


API_BASE_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
QUERY_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/query"
HEALTH_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/health"


def check_api_health() -> tuple[bool, dict[str, Any] | None, str | None]:
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=15)
        response.raise_for_status()
        return True, response.json(), None
    except requests.RequestException as exc:
        return False, None, str(exc)


def query_rag_api(query: str, top_k: int, rerank_top_n: int) -> dict[str, Any]:
    response = requests.post(
        QUERY_ENDPOINT,
        json={
            "query": query,
            "top_k": top_k,
            "rerank_top_n": rerank_top_n,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def get_error_detail(response: requests.Response | None) -> str:
    if response is None:
        return "No response body returned."

    try:
        payload = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"

    if isinstance(payload, dict):
        return str(payload.get("detail") or payload)

    return str(payload)


def format_source_label(metadata: dict[str, Any]) -> str:
    source_path = metadata.get("source_path", "")
    if source_path:
        return Path(source_path).name
    return metadata.get("title") or metadata.get("doc_id") or "Unknown source"


st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon=":mag:",
    layout="wide",
)

st.title("Enterprise RAG Assistant")
st.caption("Streamlit UI for your FastAPI RAG pipeline with retrieval, reranking, and grounded answer generation.")

with st.sidebar:
    st.subheader("Backend")
    st.code(API_BASE_URL, language="text")

    is_healthy, health_payload, health_error = check_api_health()
    if is_healthy:
        st.success("FastAPI is reachable")
        st.json(health_payload)
    else:
        st.error("FastAPI is not reachable")
        st.caption(health_error)

    pipeline_status = (health_payload or {}).get("pipeline", {})
    if pipeline_status:
        if pipeline_status.get("is_ready"):
            st.caption(
                "Pipeline ready"
                f" | docs: {pipeline_status.get('document_count', 0)}"
                f" | chunks: {pipeline_status.get('chunk_count', 0)}"
            )
        else:
            st.warning("Pipeline is not ready yet")
            if pipeline_status.get("error"):
                st.caption(pipeline_status["error"])

        warnings = pipeline_status.get("warnings") or []
        if warnings:
            st.info(f"{len(warnings)} file(s) were skipped during PDF loading.")

    top_k = st.slider("Top K Retrieval", min_value=1, max_value=20, value=5)
    rerank_top_n = st.slider("Top N After Rerank", min_value=1, max_value=10, value=3)

query = st.text_area(
    "Ask a question",
    placeholder="Example: What does the document say about reimbursement policy?",
    height=120,
)

submit = st.button("Run Query", type="primary", use_container_width=True)

if submit:
    if not query.strip():
        st.warning("Enter a question before running the query.")
    elif not is_healthy:
        st.error("FastAPI is not reachable. Start the backend before running a query.")
    elif pipeline_status and not pipeline_status.get("is_ready"):
        st.error(
            "The backend is running, but the RAG pipeline is not ready yet. "
            "Check the sidebar health details first."
        )
    else:
        with st.spinner("Querying the RAG pipeline..."):
            try:
                result = query_rag_api(query.strip(), top_k, rerank_top_n)
            except requests.HTTPError as exc:
                detail = get_error_detail(exc.response)
                st.error(f"Request failed: {detail}")
            except requests.RequestException as exc:
                st.error(f"Could not reach FastAPI backend: {exc}")
            else:
                st.subheader("Answer")
                st.write(result.get("answer", ""))

                contexts = result.get("contexts", [])
                if contexts:
                    st.subheader("Sources")
                    for item in contexts:
                        metadata = item.get("metadata", {})
                        st.markdown(f"- **{format_source_label(metadata)}**")
                        st.caption(metadata.get("source_path", "Path not available"))

                st.subheader("Retrieved Contexts")

                if not contexts:
                    st.info("No contexts were returned.")
                else:
                    for index, item in enumerate(contexts, start=1):
                        metadata = item.get("metadata", {})
                        with st.expander(f"Context {index}: {format_source_label(metadata)}"):
                            st.markdown(f"**PDF file:** {format_source_label(metadata)}")
                            st.code(metadata.get("source_path", "Path not available"), language="text")
                            st.write(item.get("text", ""))
                            st.caption(
                                f"Similarity score: {item.get('score', 0.0):.4f} | "
                                f"Rerank score: {item.get('rerank_score', 0.0):.4f}"
                            )
                            if metadata:
                                st.json(metadata)

st.markdown("---")
st.markdown(
    """
    Run order:
    1. Start FastAPI: `.\eraenv\Scripts\python.exe -m uvicorn app.main:app --reload`
    2. Start Streamlit: `.\eraenv\Scripts\python.exe -m streamlit run ui.py`
    """
)
