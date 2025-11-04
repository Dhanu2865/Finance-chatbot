"""
RAG pipeline for FinBot – Context-aware Gemini version.
Adds conversational continuity and generates concise, real-world applicable answers.
"""

import os
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
from .vectorstore import query_collection
from .embeddings import embed_texts

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = os.getenv("MODEL", "models/gemini-2.0-pro")

def retrieve(query: str, k: int = 4):
    emb = embed_texts([query])[0]
    res = query_collection(query_text=query, n_results=k)

    docs = res.get("documents", [[]])
    metas = res.get("metadatas", [[]])
    if isinstance(docs[0], list): docs = docs[0]
    if isinstance(metas[0], list): metas = metas[0]

    passages, sources = [], []
    for d, m in zip(docs, metas):
        passages.append(d)
        sources.append(m.get("source", "unknown"))
    return passages, sources


def build_prompt(query: str, contexts: List[str], memory: List[dict]) -> str:
    ctx_text = "\n\n---\n\n".join(contexts)

    memory_text = ""
    for msg in memory[-3:]:
        role = "User" if msg["role"] == "user" else "FinBot"
        memory_text += f"{role}: {msg['message']}\n"

    system = (
        "You are FinBot, a smart and friendly financial literacy chatbot. "
        "Use the provided document context and past conversation to respond naturally. "
        "Give real-world applicable insights concisely. "
        "Add emojis naturally to make it engaging. "
        "End each answer with a short follow-up question to continue the chat. "
        "Provide financial advice."
    )

    return f"{system}\n\nConversation:\n{memory_text}\n\nContext:\n{ctx_text}\n\nUser Question: {query}\n\nAnswer:"

def generate_with_gemini(prompt: str):
    model = genai.GenerativeModel(MODEL_NAME)
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.4,
                "max_output_tokens": 8192,
                "top_p": 0.9,
            },
        )
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        elif hasattr(response, "candidates") and response.candidates:
            parts = [p.text for p in response.candidates[0].content.parts if hasattr(p, "text")]
            return " ".join(parts).strip() if parts else ""
        return "I couldn’t produce a clear answer."
    except Exception as e:
        print("Gemini API error:", e)
        return "An error occurred while generating the response."

def generate_answer(user_query: str, memory: List[dict], k: int = 4):
    contexts, sources = retrieve(user_query, k)
    print(contexts)
    if not contexts:
        return {"answer": "No reference documents found. Please ingest data first.", "sources": []}

    prompt = build_prompt(user_query, contexts, memory)
    answer = generate_with_gemini(prompt)
    return {"answer": answer, "sources": sources}
