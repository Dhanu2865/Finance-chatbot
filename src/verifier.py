# src/verifier.py
"""
Response verifier for financial chatbot.
Ensures Gemini/OpenAI responses remain factual, non-advisory, and numerically sane.
"""

import re
import os
from dotenv import load_dotenv
load_dotenv()

PROVIDER = os.getenv("PROVIDER", "gemini")  

BANNED_ADVICE_KEYWORDS = [
    "buy", "sell", "recommend", "should i", "invest in", "investment advice",
    "stock pick", "target price", "portfolio allocation", "financial planning"
]

def contains_banned_advice(text: str) -> bool:
    """Detect potential investment or actionable financial advice."""
    t = text.lower()
    for kw in BANNED_ADVICE_KEYWORDS:
        if kw in t:
            return True
    return False


def numeric_sanity(text: str) -> bool:
    """Check for absurdly large or unrealistic numeric values."""
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    for n in nums:
        try:
            v = float(n)
            if abs(v) > 1e9: 
                return False
        except ValueError:
            continue
    return True


def verify(text: str):
    """
    Run all verification checks before showing model output.
    Returns (bool, message): (is_valid, reason)
    """
    if contains_banned_advice(text):
        return (
            False,
            "⚠️ Detected potential investment or actionable financial advice. "
            "This Gemini-powered bot provides only educational or informational content."
        )
    if not numeric_sanity(text):
        return (
            False,
            "⚠️ Numeric values in the response seem unrealistic or out of safe bounds."
        )
    return True, "ok"
