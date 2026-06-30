"""
ATHENA — core/compressor.py
Compression engine: turns a raw summary into a ThoughtRecord.

Steps: strip Q/A markers → truncate to 80 chars → extract top keywords.
No NLTK, no spacy. Pure Python re only. Runs in < 1 ms on an i5.

Run standalone:  python core/compressor.py
"""

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Words that carry no topical meaning — always ignored during keyword extraction.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "cannot", "dont", "doesnt",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "about", "that", "this", "these", "those", "it",
    "its", "what", "how", "why", "when", "where", "which", "who", "not",
    "also", "more", "very", "just", "than", "then", "and", "or", "but",
    "if", "so", "yet", "nor", "each", "every", "any", "all", "both",
    "such", "same", "other", "only", "own", "used", "using", "called",
    "mean", "means", "like", "returns", "return", "make", "makes",
}


@dataclass
class ThoughtRecord:
    """One compressed interaction. Produced by compress() and stored in the DB."""
    original_summary: str   # raw "Q: ... | A: ..." from reasoning.py
    compressed: str         # shortened, human-readable form (≤ 80 chars)
    keywords: list[str]     # top keywords for cluster detection in patterns.py
    embedding: list[float]  # 384-float vector — passed through, not recomputed
    timestamp: str          # UTC ISO-8601


def _extract_keywords(text: str) -> list[str]:
    """
    Tokenize → drop stopwords → drop short words → deduplicate → cap at MAX.
    Order is preserved (first occurrence wins) for deterministic clustering.
    """
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    seen: set[str] = set()
    keywords: list[str] = []
    for word in tokens:
        if word not in _STOPWORDS and len(word) >= config.MIN_KEYWORD_LENGTH and word not in seen:
            seen.add(word)
            keywords.append(word)
            if len(keywords) == config.MAX_KEYWORDS_PER_SUMMARY:
                break
    return keywords


def _make_compressed(summary: str) -> str:
    """Strip Q/A markers and truncate to 80 chars at a word boundary."""
    text = re.sub(r"^Q:\s*", "", summary)
    text = re.sub(r"\s*\|\s*A:\s*", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 80:
        text = text[:80].rsplit(" ", 1)[0] + "…"
    return text


def compress(summary: str, embedding: list[float]) -> ThoughtRecord:
    """Compress a summary string into a ThoughtRecord. Main public entry point."""
    return ThoughtRecord(
        original_summary=summary,
        compressed=_make_compressed(summary),
        keywords=_extract_keywords(summary),
        embedding=embedding,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    cases = [
        "Q: What is recursion? | A: Recursion is a function calling itself.",
        "Q: How does tail recursion work? | A: Tail recursion optimises the call stack.",
        "Q: What is a binary search tree? | A: A BST stores values in sorted order.",
        "Q: What is the weather today? | A: I cannot check real-time weather data.",
    ]
    print("=== compressor.py standalone test ===\n")
    for summary in cases:
        t0 = time.perf_counter()
        rec = compress(summary, [0.1] * 384)
        ms = (time.perf_counter() - t0) * 1000
        print(f"Original:   {rec.original_summary}")
        print(f"Compressed: {rec.compressed}")
        print(f"Keywords:   {rec.keywords}  ({ms:.2f} ms)\n")
        assert ms < 50
        assert len(rec.keywords) <= config.MAX_KEYWORDS_PER_SUMMARY
        assert all(len(k) >= config.MIN_KEYWORD_LENGTH for k in rec.keywords)
    print("✓ compressor.py standalone test passed.")
