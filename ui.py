import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st


API_BASE_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8001")
QUERY_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/query"
HEALTH_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/health"
EXAMPLE_QUERIES = [
    "What does the reimbursement policy say?",
    "Summarize the key points from the uploaded documents.",
    "Which document mentions travel reimbursement?",
    "What are the eligibility conditions mentioned in the policy?",
]


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


def status_badge(label: str, tone: str) -> str:
    tone_map = {
        "good": "#dcfce7",
        "warn": "#fef3c7",
        "bad": "#fee2e2",
        "info": "#dbeafe",
    }
    text_map = {
        "good": "#166534",
        "warn": "#92400e",
        "bad": "#991b1b",
        "info": "#1d4ed8",
    }
    return (
        "<span style="
        f"'display:inline-block;padding:0.3rem 0.7rem;border-radius:999px;"
        f"background:{tone_map[tone]};color:{text_map[tone]};font-weight:600;font-size:0.85rem;'>"
        f"{label}</span>"
    )


st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(59, 130, 246, 0.12), transparent 30%),
            radial-gradient(circle at top left, rgba(16, 185, 129, 0.10), transparent 28%),
            linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
    }
    .hero-card, .panel-card {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 20px;
        padding: 1.25rem 1.3rem;
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }
    .hero-subtitle {
        color: #334155;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 0;
    }
    .section-title {
        color: #0f172a;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    .answer-box {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #dbeafe;
        border-radius: 18px;
        padding: 1rem 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "query_text" not in st.session_state:
    st.session_state.query_text = ""

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


with st.sidebar:
    st.markdown("### Control Center")
    st.code(API_BASE_URL, language="text")

    is_healthy, health_payload, health_error = check_api_health()
    pipeline_status = (health_payload or {}).get("pipeline", {})

    if is_healthy:
        st.markdown(status_badge("FastAPI Reachable", "good"), unsafe_allow_html=True)
    else:
        st.markdown(status_badge("Backend Offline", "bad"), unsafe_allow_html=True)
        st.caption(health_error)

    if pipeline_status.get("is_ready"):
        st.markdown(status_badge("Pipeline Ready", "good"), unsafe_allow_html=True)
    elif pipeline_status.get("is_initializing"):
        st.markdown(status_badge("Pipeline Initializing", "warn"), unsafe_allow_html=True)
    else:
        st.markdown(status_badge("Pipeline Idle", "info"), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Retrieval Settings")
    top_k = st.slider("Top K Retrieval", min_value=1, max_value=20, value=5)
    rerank_top_n = st.slider("Top N After Rerank", min_value=1, max_value=10, value=3)

    st.markdown("---")
    st.subheader("Quick Start")
    for example in EXAMPLE_QUERIES:
        if st.button(example, use_container_width=True):
            st.session_state.query_text = example

    if health_payload:
        st.markdown("---")
        st.subheader("Backend Snapshot")
        col_a, col_b = st.columns(2)
        col_a.metric("Docs", pipeline_status.get("document_count", 0))
        col_b.metric("Chunks", pipeline_status.get("chunk_count", 0))
        st.caption(
            f"Dataset: {pipeline_status.get('dataset_type') or 'not loaded'} | "
            f"Init time: {pipeline_status.get('init_duration_seconds') or 'n/a'}"
        )
        warnings = pipeline_status.get("warnings") or []
        if warnings:
            st.warning(f"{len(warnings)} file(s) were skipped during loading.")
        if pipeline_status.get("error"):
            st.error(pipeline_status["error"])


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Enterprise RAG Assistant</div>
        <p class="hero-subtitle">
            Ask grounded questions across your document corpus, review retrieved evidence,
            and monitor backend readiness from one workspace.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
metric_col_1.metric("Backend", "Online" if is_healthy else "Offline")
metric_col_2.metric("Pipeline", "Ready" if pipeline_status.get("is_ready") else "Not Ready")
metric_col_3.metric("Documents", pipeline_status.get("document_count", 0))
metric_col_4.metric("Chunks", pipeline_status.get("chunk_count", 0))

left_col, right_col = st.columns([1.8, 1], gap="large")

with left_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Ask a Question</div>', unsafe_allow_html=True)
    st.caption("Use a specific question for better retrieval and reranking quality.")

    query = st.text_area(
        "Question",
        value=st.session_state.query_text,
        placeholder="Example: What does the reimbursement policy say about travel claims?",
        height=150,
        label_visibility="collapsed",
    )
    st.session_state.query_text = query

    action_col_1, action_col_2 = st.columns([1, 4])
    submit = action_col_1.button("Run Query", type="primary", use_container_width=True)
    clear = action_col_2.button("Clear", use_container_width=True)

    if clear:
        st.session_state.query_text = ""
        st.session_state.latest_result = None
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">System Health</div>', unsafe_allow_html=True)

    if is_healthy:
        st.success("FastAPI is reachable and responding.")
    else:
        st.error("FastAPI is not reachable.")
        st.caption(health_error or "Check whether the backend is running.")

    if pipeline_status:
        if pipeline_status.get("is_ready"):
            st.info("The retrieval pipeline is ready for queries.")
        elif pipeline_status.get("is_initializing"):
            st.warning("The pipeline is still initializing.")
        else:
            st.warning("The pipeline has not been initialized yet.")

        st.json(pipeline_status)
    elif health_payload:
        st.json(health_payload)
    else:
        st.caption("Health details will appear here after the backend becomes reachable.")

    st.markdown("</div>", unsafe_allow_html=True)

if submit:
    if not query.strip():
        st.warning("Enter a question before running the query.")
    elif not is_healthy:
        st.error("FastAPI is not reachable. Start the backend before running a query.")
    else:
        with st.spinner("Retrieving relevant context and generating answer..."):
            try:
                result = query_rag_api(query.strip(), top_k, rerank_top_n)
            except requests.HTTPError as exc:
                detail = get_error_detail(exc.response)
                st.error(f"Request failed: {detail}")
            except requests.RequestException as exc:
                st.error(f"Could not reach FastAPI backend: {exc}")
            else:
                st.session_state.latest_result = result

result = st.session_state.latest_result

if result:
    st.write("")
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    tab_answer, tab_sources, tab_contexts = st.tabs(["Answer", "Sources", "Retrieved Contexts"])

    with tab_answer:
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.subheader("Generated Answer")
        st.write(result.get("answer", ""))
        st.markdown("</div>", unsafe_allow_html=True)

    contexts = result.get("contexts", [])

    with tab_sources:
        if not contexts:
            st.info("No sources were returned.")
        else:
            for item in contexts:
                metadata = item.get("metadata", {})
                st.markdown(f"**{format_source_label(metadata)}**")
                st.caption(metadata.get("source_path", "Path not available"))
                st.caption(
                    f"Similarity: {item.get('score', 0.0):.4f} | "
                    f"Rerank: {item.get('rerank_score', 0.0):.4f}"
                )
                st.markdown("---")

    with tab_contexts:
        if not contexts:
            st.info("No retrieved contexts were returned.")
        else:
            for index, item in enumerate(contexts, start=1):
                metadata = item.get("metadata", {})
                title = format_source_label(metadata)
                with st.expander(f"Context {index}: {title}", expanded=index == 1):
                    st.markdown(f"**Source:** {title}")
                    st.code(metadata.get("source_path", "Path not available"), language="text")
                    st.write(item.get("text", ""))
                    st.caption(
                        f"Similarity score: {item.get('score', 0.0):.4f} | "
                        f"Rerank score: {item.get('rerank_score', 0.0):.4f}"
                    )
                    if metadata:
                        st.json(metadata)

    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Run a query to view the generated answer, sources, and retrieved contexts.")

st.markdown("---")
st.caption(
    "Run order: start FastAPI with "
    "`python -m uvicorn app.main:app --host 127.0.0.1 --port 8001`, "
    "then start Streamlit with `python -m streamlit run ui.py`."
)
