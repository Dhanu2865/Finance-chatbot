# src/vectorstore.py
"""
Vectorstore management using ChromaDB + Gemini embeddings.
Supports persistence and querying for RAG applications.
"""

import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY is missing in .env")

genai.configure(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")

def get_chroma_client():
    """Initialize Chroma persistent client."""
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client


def create_or_reset_collection(collection_name="finbot_docs"):
    """Delete and recreate a fresh Chroma collection."""
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # ignore if not exists
    return client.create_collection(name=collection_name)


def get_collection(collection_name="finbot_docs"):
    """Fetch an existing Chroma collection if available."""
    client = get_chroma_client()
    try:
        return client.get_collection(collection_name)
    except Exception:
        return None

def get_embedding(text: str):
    """
    Generate embeddings using Gemini text-embedding-004 model.
    Falls back to local embeddings if Gemini fails.
    """
    try:
        result = genai.embed_content(model=EMBEDDING_MODEL, content=text)
        return result["embedding"]
    except Exception as e:
        print(f"⚠️ Gemini embedding failed: {e}")
        try:
            # fallback: sentence-transformers (local)
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            return model.encode(text).tolist()
        except Exception as e2:
            raise RuntimeError(f"Both Gemini and local embedding failed: {e2}")


def add_documents_to_collection(docs, collection_name="finbot_docs"):
    """
    Add multiple documents to the Chroma collection.
    Each document gets an auto-generated embedding.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)

    ids = [str(i) for i in range(len(docs))]
    metadatas = [{"source": f"chunk_{i}"} for i in range(len(docs))]
    embeddings = [get_embedding(doc) for doc in docs]

    collection.add(
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"✅ Added {len(docs)} documents to collection '{collection_name}'.")
    return len(docs)


def query_collection(query_text=None, n_results=4, collection_name="finbot_docs"):
    """
    Retrieve similar document chunks from Chroma given a query.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)

    if not query_text:
        return {"documents": [], "ids": [], "metadatas": []}

    query_embedding = get_embedding(query_text)
    res = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return res
