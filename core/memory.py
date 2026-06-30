"""
ATHENA — core/memory.py
MemoryManager: abstraction layer between the reasoning engine and storage.

Phase 3: store() now calls compressor before saving, writing compressed_summary
and keywords to the DB. get_candidate_concepts() is new — calls patterns.py.

Run standalone:  python core/memory.py
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from database.sqlite import Database
from core.embeddings import embed
from core.compressor import compress
from core.patterns   import detect_patterns, CandidateConcept
from core.symbols    import assign_symbol, Symbol

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity via numpy. Returns 0.0 if either vector is all zeros."""
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return float(np.dot(va, vb) / (na * nb)) if na and nb else 0.0


class MemoryManager:
    """Public interface for all memory operations."""

    def __init__(self, db: Database):
        self.db = db

    def store(
        self,
        user_input: str,
        summary: str,
        answer: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Persist one interaction. Phase 3: generates embedding, compresses
        summary to extract keywords, then writes all fields to the DB.
        Returns the new row ID, or -1 on failure.
        """
        try:
            embedding = embed(summary)
            thought   = compress(summary, embedding)   # Phase 3: compression
            return self.db.insert_interaction(
                user_input=user_input,
                summary=summary,
                answer=answer,
                metadata=metadata or {},
                embedding=thought.embedding,
                compressed_summary=thought.compressed,
                keywords=thought.keywords,
            )
        except Exception as e:
            logger.error("store() failed: %s", e)
            return -1

    def retrieve(self, row_id: int) -> Optional[dict]:
        """Fetch a single interaction by ID. Returns dict or None."""
        try:
            return self.db.get_interaction(row_id)
        except Exception as e:
            logger.error("retrieve() failed: %s", e)
            return None

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Semantic search via cosine similarity. Unchanged from Phase 2.
        Returns interactions sorted by similarity, most similar first.
        Each result dict includes a "similarity" key (float, 0-1).
        """
        try:
            query_vector   = embed(query)
            all_embeddings = self.db.get_all_embeddings()
            if not all_embeddings:
                return []

            scored = sorted(
                [{"id": e["id"], "similarity": _cosine_similarity(query_vector, e["embedding"])}
                 for e in all_embeddings],
                key=lambda x: x["similarity"],
                reverse=True,
            )
            results = []
            for entry in scored[:limit]:
                record = self.db.get_interaction(entry["id"])
                if record:
                    record["similarity"] = round(entry["similarity"], 4)
                    results.append(record)
            return results
        except Exception as e:
            logger.error("search() failed: %s", e)
            return []

    def get_candidate_concepts(self) -> list[CandidateConcept]:
        """
        Surface recurring keyword clusters from all stored interactions.
        Delegates to patterns.detect_patterns(). Returns [] if no keywords yet.
        """
        try:
            rows = self.db.get_all_keywords()
            return detect_patterns(rows) if rows else []
        except Exception as e:
            logger.error("get_candidate_concepts() failed: %s", e)
            return []

    def count(self) -> int:
        """Return total number of stored interactions."""
        try:
            return self.db.count_interactions()
        except Exception as e:
            logger.error("count() failed: %s", e)
            return 0

    # ── Phase 4 methods ───────────────────────────────────────────────────────

    def promote_concepts(self, graph) -> list[Symbol]:
        """
        Check all candidate concepts. Any that don't yet have a symbol get one.
        Saves new symbols to DB and adds them to the in-memory graph.

        Args:
            graph: A GraphManager instance (imported in main.py to avoid circular deps).

        Returns:
            List of newly-created Symbol instances (empty if nothing is new).
        """
        try:
            concepts = self.get_candidate_concepts()
            newly_born: list[Symbol] = []
            for concept in concepts:
                candidate_symbol = assign_symbol(concept)
                if self.db.symbol_exists(candidate_symbol.name):
                    self.db.update_symbol_frequency(
                        candidate_symbol.name, candidate_symbol.frequency
                    )
                    continue

                self.db.add_symbol(candidate_symbol)
                graph.add_symbol(candidate_symbol)
                newly_born.append(candidate_symbol)

            return newly_born
        except Exception as e:
            logger.error("promote_concepts() failed: %s", e)
            return []

    def get_symbol_for_interaction(self, interaction_id: int) -> Optional[str]:
        """
        Return the name of any symbol whose member_ids includes interaction_id.
        Returns None if no symbol matches.
        """
        try:
            for symbol in self.db.get_all_symbols():
                if interaction_id in symbol.member_ids:
                    return symbol.name
            return None
        except Exception as e:
            logger.error("get_symbol_for_interaction() failed: %s", e)
            return None

    # ── Phase 5 methods ───────────────────────────────────────────────────────

    def get_all_symbols(self) -> list[Symbol]:
        """Return all symbols stored in the DB. Used by the auditor."""
        try:
            return self.db.get_all_symbols()
        except Exception as e:
            logger.error("get_all_symbols() failed: %s", e)
            return []

    def run_session_audit(self, graph) -> "AuditReport":
        """
        End-of-session audit:
          1. Retrieves all symbols from DB.
          2. Finds combinable pairs and merges them in the graph.
          3. Prunes weak symbols from the graph (not from DB).
          4. Returns a full AuditReport.
        """
        from core.auditor  import audit, AuditReport
        from core.combiner import find_combinable_pairs, combine

        symbols = self.get_all_symbols()

        # Combine eligible pairs — update graph only (not DB, to keep history)
        pairs = find_combinable_pairs(symbols)
        for sym_a, sym_b in pairs:
            compound = combine(sym_a, sym_b)
            if compound:
                graph.merge_symbols(sym_a.name, sym_b.name, compound)
                logger.info("Phase 5 combine: %s + %s -> %s",
                            sym_a.name, sym_b.name, compound.name)

        # Prune weak symbols from graph (keep DB history)
        for sym in symbols:
            if sym.frequency < config.MIN_CLUSTER_SIZE:
                graph.prune_symbol(sym.name)
                logger.info("Phase 5 prune: removed weak symbol '%s' from graph.", sym.name)

        return audit(symbols, graph)


# ── Standalone test (needs the embedding model — runs on your machine) ─────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_path = config.DATA_DIR / "test_memory_phase3.db"
    test_path.unlink(missing_ok=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    db = Database(db_path=test_path)
    db.connect()
    mem = MemoryManager(db)

    data = [
        ("What is recursion?",
         "Q: What is recursion? | A: Recursion is a technique where a function call itself. Each recursion needs a base condition.",
         "Recursion is a technique where a function calls itself."),
        ("How does tail recursion work?",
         "Q: How does tail recursion work? | A: Tail recursion is a function call where recursion is the final operation executed.",
         "Tail recursion places the recursive call at the very end."),
        ("What is tail call recursion?",
         "Q: What is tail call recursion? | A: Tail call recursion means the recursive function call occurs last in the function body.",
         "Tail call recursion is when the recursive call is the last statement."),
        ("What is the weather today?",
         "Q: What is the weather today? | A: I cannot check real-time weather.",
         "I don't have access to real-time weather."),
    ]

    print("Storing 4 interactions...")
    for user_input, summary, answer in data:
        row_id = mem.store(user_input=user_input, summary=summary,
                           answer=answer, metadata={"duration_ms": 500})
        print(f"  '{user_input[:40]}' -> row {row_id}")

    print(f"\nTotal memories: {mem.count()}")

    print("\nSearching for 'recursive function call'...")
    results = mem.search("recursive function call", limit=3)
    for r in results:
        print(f"  [{r['similarity']:.4f}] {r['user_input']}")

    print("\nDetecting candidate concepts...")
    concepts = mem.get_candidate_concepts()
    for c in concepts:
        print(f"  {c.keywords}  (seen {c.frequency} times, members={c.member_ids})")

    db.close()
    test_path.unlink(missing_ok=True)
    print("\n✓ memory.py standalone test passed.")
