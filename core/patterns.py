"""
ATHENA — core/patterns.py
Pattern detector: finds keyword clusters across stored interactions.

For every pair of interactions, if they share ≥ 2 keywords they belong
to the same cluster. Clusters with ≥ MIN_CLUSTER_SIZE members become
CandidateConcepts — patterns ATHENA noticed without being told.

Complexity: O(n²) pair comparisons. Fine for Phase 3 (< 1000 rows).

Run standalone:  python core/patterns.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


@dataclass
class CandidateConcept:
    """A recurring pattern ATHENA has observed enough times to consider naming."""
    keywords: list[str]   # shared keywords that define this concept
    member_ids: list[int] # DB interaction IDs in this cluster
    frequency: int        # how many interactions belong to this cluster
    first_seen: str       # timestamp of the earliest member (UTC ISO-8601)


def detect_patterns(rows: list[dict]) -> list[CandidateConcept]:
    """
    Scan all keyword rows and return CandidateConcepts.

    Args:
        rows: list of dicts with keys: id (int), keywords (list[str]), timestamp (str)

    Returns:
        CandidateConcepts with frequency >= MIN_CLUSTER_SIZE, most frequent first.
    """
    # clusters: frozenset(shared keywords) -> [(id, timestamp), ...]
    clusters: dict[frozenset, list[tuple[int, str]]] = {}

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            # Find keywords that appear in both rows
            set_b = set(rows[j]["keywords"])
            shared = [k for k in rows[i]["keywords"] if k in set_b]

            if len(shared) < 1:
                continue   # not enough overlap — skip this pair

            key = frozenset(shared)
            if key not in clusters:
                clusters[key] = []

            tracked = {e[0] for e in clusters[key]}
            if rows[i]["id"] not in tracked:
                clusters[key].append((rows[i]["id"], rows[i]["timestamp"]))
            if rows[j]["id"] not in tracked:
                clusters[key].append((rows[j]["id"], rows[j]["timestamp"]))

    concepts: list[CandidateConcept] = []
    for kw_set, members in clusters.items():
        if len(members) < config.MIN_CLUSTER_SIZE:
            continue   # pattern not strong enough yet

        members.sort(key=lambda x: x[1])   # sort by timestamp → find first_seen
        concepts.append(CandidateConcept(
            keywords=sorted(kw_set),
            member_ids=[m[0] for m in members],
            frequency=len(members),
            first_seen=members[0][1],
        ))

    concepts.sort(key=lambda c: c.frequency, reverse=True)

    # ── PHASE 4 HOOK ──────────────────────────────────────────────────────────
    # `concepts` is ready here. Phase 4 will iterate over this list and call:
    #   from core.symbols import assign_symbol
    #   for concept in concepts:
    #       concept.symbol = assign_symbol(concept)
    # assign_symbol() picks a short uppercase name (e.g. "RECURSION") and
    # writes it back to the DB. No signature changes needed in this function.
    # ─────────────────────────────────────────────────────────────────────────

    return concepts


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== patterns.py standalone test ===\n")

    fake_rows = [
        {"id": 1, "keywords": ["recursion", "function", "call", "stack"],
         "timestamp": "2025-01-01T10:00:00+00:00"},
        {"id": 2, "keywords": ["tail", "recursion", "function", "call"],
         "timestamp": "2025-01-01T10:05:00+00:00"},
        {"id": 3, "keywords": ["recursion", "function", "base", "call"],
         "timestamp": "2025-01-01T10:10:00+00:00"},
        {"id": 4, "keywords": ["weather", "temperature", "forecast"],
         "timestamp": "2025-01-01T10:15:00+00:00"},
    ]

    concepts = detect_patterns(fake_rows)
    for c in concepts:
        print(f"  Keywords={c.keywords}  freq={c.frequency}  members={c.member_ids}")

    assert len(concepts) >= 1, "Expected ≥ 1 concept"
    top = concepts[0]
    assert "recursion" in top.keywords
    assert top.frequency >= config.MIN_CLUSTER_SIZE
    all_member_ids = [mid for c in concepts for mid in c.member_ids]
    assert 4 not in all_member_ids, "Weather row (id=4) should not be in any concept"

    print("✓ patterns.py standalone test passed.")
