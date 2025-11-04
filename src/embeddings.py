import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "gemini")  # default: gemini
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")  # ✅ Use from .env

# ---------------------------
# GEMINI Embeddings
# ---------------------------
def embed_with_gemini(texts: List[str], model=EMBEDDING_MODEL):
    """
    Generates embeddings using the Gemini API.
    Requires `google-generativeai` package and GEMINI_API_KEY in .env
    """
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError("google-generativeai package required for Gemini embeddings") from e

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set in env")

    genai.configure(api_key=key)

    embeddings = []
    for text in texts:
        result = genai.embed_content(model=model, content=text)
        embeddings.append(result["embedding"])
    return embeddings


# ---------------------------
# OPENAI Embeddings
# ---------------------------
def embed_with_openai(texts: List[str], model="text-embedding-3-large"):
    try:
        import openai
    except Exception as e:
        raise RuntimeError("openai package required for OpenAI embeddings") from e

    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")  # ✅ Flexible lookup
    if not key:
        raise RuntimeError("OPENAI_API_KEY or OPEN_AI_KEY not set in env")

    openai.api_key = key
    resp = openai.Embedding.create(model=model, input=texts)
    return [r["embedding"] for r in resp["data"]]


# ---------------------------
# LOCAL Fallback
# ---------------------------
def embed_with_local(texts: List[str], model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=False)
    return embs.tolist()


# ---------------------------
# MASTER FUNCTION
# ---------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Default embedding function.
    Uses Gemini if PROVIDER=gemini,
    OpenAI if PROVIDER=openai,
    otherwise falls back to local SentenceTransformer.
    """
    if PROVIDER == "gemini":
        try:
            return embed_with_gemini(texts)
        except Exception as e:
            print("Gemini embedding failed; falling back to local model:", e)
            return embed_with_local(texts)
    elif PROVIDER == "openai":
        try:
            return embed_with_openai(texts)
        except Exception as e:
            print("OpenAI embedding failed; falling back to local model:", e)
            return embed_with_local(texts)
    else:
        return embed_with_local(texts)
