"""
ATHENA — core/symbols.py
Symbol engine: converts a CandidateConcept into a short, permanent Symbol name.

A Symbol is just an uppercase string ATHENA assigns itself — e.g. "RECURSION"
or "RECURSION_FUNCTION". Once assigned it lives in SQLite forever.

Run standalone:  python core/symbols.py
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.patterns import CandidateConcept

# ── PHASE 5 HOOK ──────────────────────────────────────────────────────────────
# Contradiction detection will go here. Before assign_symbol() saves a new
# symbol, Phase 5 will compare the candidate concept's member interactions
# against each other. If two members contain conflicting statements about the
# same keywords, assign_symbol() should flag the symbol as "contradicted" in
# the DB instead of saving it as stable.
# Example hook point: after `name = _make_name(concept)`, add:
#   from core.contradictions import check_contradictions
#   if check_contradictions(concept):
#       return None   # or a Symbol with a "contradicted" flag
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Symbol:
    """A permanent named abstraction ATHENA invented from repeated patterns."""
    name: str             # "RECURSION" or "RECURSION_FUNCTION" — max 20 chars
    keywords: list[str]   # keywords that define this symbol
    member_ids: list[int] # DB interaction IDs that belong to this symbol
    frequency: int        # how many interactions triggered this symbol
    created_at: str       # UTC ISO-8601 timestamp when the symbol was born


def _make_name(concept: CandidateConcept) -> str:
    """
    Build a short uppercase name from the top 2 keywords.

    Examples:
        ['recursion', 'function'] -> "RECURSION_FUNCTION"
        ['recursion']             -> "RECURSION"
    """
    top = concept.keywords[:2]            # at most 2 words
    name = "_".join(w.upper() for w in top)
    return name[:20]                      # hard cap at 20 chars per spec


def assign_symbol(concept: CandidateConcept) -> Symbol:
    """
    Turn a CandidateConcept into a Symbol dataclass.

    Does NOT write to the DB — that is the caller's job (MemoryManager).
    Call symbol_exists() on the DB before calling this to avoid duplicates.

    Args:
        concept: A CandidateConcept from patterns.detect_patterns().

    Returns:
        A fresh Symbol ready to be saved.
    """
    name = _make_name(concept)
    return Symbol(
        name=name,
        keywords=concept.keywords,
        member_ids=concept.member_ids,
        frequency=concept.frequency,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== symbols.py standalone test ===\n")

    concept_one = CandidateConcept(
        keywords=["recursion", "function"],
        member_ids=[1, 2, 3],
        frequency=3,
        first_seen="2025-01-01T10:00:00+00:00",
    )
    s = assign_symbol(concept_one)
    assert s.name == "RECURSION_FUNCTION", f"Got: {s.name}"
    assert s.frequency == 3
    assert s.keywords == ["recursion", "function"]
    print(f"  assign_symbol(['recursion','function']) -> '{s.name}'")

    concept_single = CandidateConcept(
        keywords=["recursion"],
        member_ids=[1, 2, 3],
        frequency=3,
        first_seen="2025-01-01T10:00:00+00:00",
    )
    s2 = assign_symbol(concept_single)
    assert s2.name == "RECURSION", f"Got: {s2.name}"
    print(f"  assign_symbol(['recursion']) -> '{s2.name}'")

    long_concept = CandidateConcept(
        keywords=["verylongword", "anotherlongword"],
        member_ids=[1, 2, 3],
        frequency=3,
        first_seen="2025-01-01T10:00:00+00:00",
    )
    s3 = assign_symbol(long_concept)
    assert len(s3.name) <= 20, f"Name too long: {s3.name}"
    print(f"  Long keyword name capped to 20 chars: '{s3.name}'")

    print("\n✓ symbols.py standalone test passed.")
