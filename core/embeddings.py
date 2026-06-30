"""
ATHENA — core/embeddings.py
Wraps sentence-transformers to produce float embedding vectors.

The model is NOT loaded at import time. It loads on the first call to embed()
or embed_batch(). This keeps ATHENA's boot time fast and saves RAM until
the embedding feature is actually used.

Run standalone to verify the model downloads and produces vectors:
    python core/embeddings.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

logger = logging.getLogger(__name__)

# Module-level handle for the loaded model.
# Stays None until the first embed() or embed_batch() call.
_model = None


def _load_model() -> None:
    """
    Load the SentenceTransformer model into memory (runs exactly once).
    Called automatically by embed() and embed_batch() — do not call directly.
    """
    global _model
    if _model is not None:
        return  # already loaded, nothing to do

    # Import here so the library is not pulled in at module load time.
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model '%s' — this takes ~5 s the first time.", config.EMBEDDING_MODEL)
    _model = SentenceTransformer(config.EMBEDDING_MODEL)
    logger.info("Embedding model ready.")


def embed(text: str) -> list[float]:
    """
    Convert one text string into a float embedding vector.

    The vector has 384 dimensions (all-MiniLM-L6-v2).
    Lazy-loads the model on the first call.

    Args:
        text: any string — a question, a summary, anything.

    Returns:
        A list of 384 floats representing the semantic meaning of the text.
    """
    _load_model()
    vector = _model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of strings into embedding vectors in one efficient pass.
    Prefer this over calling embed() in a loop when you have many texts.

    Args:
        texts: list of strings to embed.

    Returns:
        A list of float vectors, one per input string, in the same order.
    """
    _load_model()
    vectors = _model.encode(texts, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("Testing embed() — single text...")
    v1 = embed("What is a recursive function?")
    print(f"  Vector dimensions : {len(v1)}")
    print(f"  First 5 values    : {[round(x, 4) for x in v1[:5]]}")

    print("\nTesting embed_batch() — two texts...")
    vecs = embed_batch(["What is a loop?", "Explain functions in Python."])
    print(f"  Batch size        : {len(vecs)}")
    print(f"  Each vector dims  : {len(vecs[0])}")

    # Quick sanity check: similar texts should have high cosine similarity.
    import numpy as np
    a = np.array(embed("What is recursion?"))
    b = np.array(embed("What is a recursive function?"))
    c = np.array(embed("What is the weather today?"))

    sim_ab = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    sim_ac = float(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c)))
    print(f"\nSimilarity check:")
    print(f"  'recursion' vs 'recursive function' : {sim_ab:.4f}  (expect > 0.85)")
    print(f"  'recursion' vs 'weather today'      : {sim_ac:.4f}  (expect < 0.40)")

    assert sim_ab > 0.7, "Similar texts scored too low — something is wrong."
    assert sim_ac < 0.6, "Unrelated texts scored too high — something is wrong."

    print("\n✓ embeddings.py standalone test passed.")
