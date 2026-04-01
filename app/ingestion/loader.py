import json
from pathlib import Path


DATASET_DIR = Path("datasets") / "fiqa"


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
