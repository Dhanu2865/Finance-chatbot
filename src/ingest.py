import os
import uuid
from pathlib import Path
from typing import List
from tqdm import tqdm
from dotenv import load_dotenv
import re
import PyPDF2

from .vectorstore import create_or_reset_collection, add_documents_to_collection
from .embeddings import embed_texts

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")

def read_pdf(path: str) -> str:
    text = ""
    with open(path, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for p in reader.pages:
            t = p.extract_text() or ""
            text += t + "\n"
    return text


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf8") as f:
        return f.read()


def clean_text(s: str) -> str:
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def chunk_text(text: str, chunk_size_words: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size_words]
        chunks.append(" ".join(chunk))
        i += chunk_size_words - overlap
    return chunks

def load_documents(data_dir=DATA_DIR):
    files = list(Path(data_dir).glob("*"))
    docs = []
    for f in files:
        if f.suffix.lower() == ".pdf":
            txt = read_pdf(str(f))
        elif f.suffix.lower() in [".txt", ".md"]:
            txt = read_text_file(str(f))
        else:
            continue
        txt = clean_text(txt)
        chunks = chunk_text(txt)
        for idx, c in enumerate(chunks):
            docs.append({"text": c, "meta": {"source": f.name, "chunk": idx}})
    return docs


def ingest_all(data_dir=DATA_DIR):
    docs = load_documents(data_dir)
    if not docs:
        print(f"No documents found in {data_dir}")
        return 0

    texts = [d["text"] for d in docs]

    print(f"Embedding {len(texts)} chunks using Gemini (or fallback model)...")
    # Optional: Pre-generate embeddings to test consistency
    _ = embed_texts(texts)

    print("Creating new collection and adding documents to Chroma...")
    create_or_reset_collection()
    add_documents_to_collection(texts)  # ✅ Pass only texts (embeddings are handled inside)

    print("✅ Ingestion complete. Persisted to Chroma.")
    return len(texts)


if __name__ == "__main__":
    n = ingest_all()
    print("Ingested chunks:", n)
