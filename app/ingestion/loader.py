import json
import os
from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader


DATASET_DIR = Path("datasets") / "fiqa"
PDF_DATASET_DIR = Path("data") / "rag_data"
ENABLE_PDF_OCR = os.getenv("ENABLE_PDF_OCR", "true").lower() == "true"
PDF_OCR_RENDER_SCALE = float(os.getenv("PDF_OCR_RENDER_SCALE", "2.0"))


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_fiqa_corpus(dataset_dir: str | Path = DATASET_DIR):
    dataset_dir = Path(dataset_dir)
    corpus_path = dataset_dir / "corpus.jsonl"

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Local FIQA corpus not found at '{corpus_path}'. "
            "Place the dataset under datasets/fiqa before starting the app."
        )

    corpus = {}
    for row in _read_jsonl(corpus_path):
        doc_id = row["_id"]
        corpus[doc_id] = {
            "title": row.get("title", ""),
            "text": row.get("text", ""),
        }

    return corpus


def load_pdf_corpus(dataset_dir: str | Path = PDF_DATASET_DIR):
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"PDF dataset folder not found at '{dataset_dir}'. "
            "Create the folder and place your PDF files inside it."
        )

    pdf_paths = sorted(
        pdf_path
        for pdf_path in dataset_dir.rglob("*.pdf")
        if not any(part.startswith(".") for part in pdf_path.parts)
    )
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files were found under '{dataset_dir}'. "
            "Add one or more PDF files before starting the app."
        )

    corpus = {}
    for pdf_path in pdf_paths:
        pages = _extract_pdf_text(pdf_path)

        full_text = "\n\n".join(pages).strip()
        if not full_text:
            continue

        relative_path = pdf_path.relative_to(dataset_dir)
        doc_id = str(relative_path.with_suffix("")).replace("\\", "/")
        corpus[doc_id] = {
            "title": pdf_path.name,
            "text": full_text,
            "source_path": str(pdf_path.resolve()),
        }

    if not corpus:
        raise ValueError(
            f"PDF files were found under '{dataset_dir}', but no extractable text was detected."
        )

    return corpus


def _extract_pdf_text(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(text)
            continue

        if not ENABLE_PDF_OCR:
            continue

        ocr_text = _extract_page_text_with_ocr(pdf_path, page_index)
        if ocr_text:
            pages.append(ocr_text)

    return pages


@lru_cache(maxsize=1)
def _get_ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "OCR dependencies are missing. Install requirements and ensure "
            "'rapidocr-onnxruntime' is available."
        ) from exc

    return RapidOCR()


def _extract_page_text_with_ocr(pdf_path: Path, page_index: int) -> str:
    try:
        import numpy as np
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "OCR dependencies are missing. Install requirements and ensure "
            "'pypdfium2' and 'rapidocr-onnxruntime' are available."
        ) from exc

    pdf = pdfium.PdfDocument(str(pdf_path))
    page = None
    bitmap = None
    image = None

    try:
        page = pdf[page_index]
        bitmap = page.render(scale=PDF_OCR_RENDER_SCALE)
        image = bitmap.to_pil()
        result, _ = _get_ocr_engine()(np.array(image))
    finally:
        if image is not None and hasattr(image, "close"):
            image.close()
        if bitmap is not None and hasattr(bitmap, "close"):
            bitmap.close()
        if page is not None and hasattr(page, "close"):
            page.close()
        if hasattr(pdf, "close"):
            pdf.close()

    if not result:
        return ""

    lines = []
    for item in result:
        if len(item) >= 2:
            text = str(item[1]).strip()
            if text:
                lines.append(text)

    return "\n".join(lines)


def load_fiqa_queries_and_qrels(dataset_dir: str | Path = DATASET_DIR, split: str = "test"):
    dataset_dir = Path(dataset_dir)
    queries_path = dataset_dir / "queries.jsonl"
    qrels_path = dataset_dir / "qrels" / f"{split}.tsv"

    if not queries_path.exists() or not qrels_path.exists():
        raise FileNotFoundError(f"Queries or qrels not found under '{dataset_dir}'.")

    queries = {}
    for row in _read_jsonl(queries_path):
        queries[row["_id"]] = row["text"]

    qrels = {}
    with qrels_path.open("r", encoding="utf-8") as file:
        next(file, None)
        for line in file:
            query_id, corpus_id, score = line.strip().split("\t")
            qrels.setdefault(query_id, {})[corpus_id] = int(score)

    return queries, qrels
