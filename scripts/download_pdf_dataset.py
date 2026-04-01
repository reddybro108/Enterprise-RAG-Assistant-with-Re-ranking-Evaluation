import os
import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_REPO_ID = "pdfqa/pdfQA-Benchmark"
TARGET_DIR = Path("data") / "rag_data"
MAX_FILES = 100


def is_valid_windows_filename(name: str) -> bool:
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
    return not any(char in name for char in invalid_chars)


def download_limited_pdfs(repo_id: str, repo_type: str):
    print("📥 Fetching file list from HuggingFace...")

    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

    # ✅ FIX: validate only filename, not full path
    pdf_files = []
    for f in files:
        if f.lower().endswith(".pdf"):
            filename = os.path.basename(f)   # 🔥 key fix
            if is_valid_windows_filename(filename):
                pdf_files.append(f)

    if not pdf_files:
        raise ValueError("No valid PDF files found after filtering.")

    # limit to 100
    pdf_files = pdf_files[:MAX_FILES]

    destination_root = TARGET_DIR / repo_id.replace("/", "__")
    destination_root.mkdir(parents=True, exist_ok=True)

    print(f"📁 Downloading {len(pdf_files)} PDFs...\n")

    downloaded = 0

    for file in pdf_files:
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=file,
                local_dir=destination_root,
            )

            print(f"✅ {file}")
            downloaded += 1

        except Exception as e:
            print(f"⚠️ Skipped: {file}")
            print(f"   Reason: {e}")

    print(f"\n🎯 Download complete: {downloaded} files")
    print(f"📍 Location: {destination_root.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-type", default="dataset")

    args = parser.parse_args()

    download_limited_pdfs(args.repo_id, args.repo_type)


if __name__ == "__main__":
    main()
