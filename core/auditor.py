"""
ATHENA — core/auditor.py
Session auditor: reviews the entire symbol graph and reports its health.

Called once per session (on exit). Produces an AuditReport that summarises
active symbols, weak symbols, combinable pairs, and potential contradictions.

Run standalone:  python core/auditor.py
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from core.symbols import Symbol

# ── FUTURE LLM PRUNING HOOK ───────────────────────────────────────────────────
# Phase 6 or later: replace keyword-overlap contradiction detection with an
# LLM pass. Hook into detect_contradictions() after the overlap check:
#
#   from core.model import LocalModel
#   llm = LocalModel()
#   for name in contradiction_candidates:
#       verdict = llm.ask(
#           f"Symbols '{name}' and its near-duplicate have these keywords: "
#           f"{...}. Are they truly contradictory or just synonyms? Reply YES/NO."
#       )
#       if verdict.strip().upper() == "YES":
#           confirmed.append(name)
#   return confirmed
#
# The current heuristic (> CONTRADICTION_THRESHOLD keyword overlap) is a fast
# approximation that works without any model calls.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AuditReport:
    """Snapshot of symbol graph health at the end of a session."""
    total_symbols: int
    active_symbols: list[str]           # frequency >= MIN_CLUSTER_SIZE
    weak_symbols: list[str]             # frequency <  MIN_CLUSTER_SIZE
    combinable_pairs: list[tuple[str, str]]
    contradiction_candidates: list[str] # symbols with > 80% keyword overlap
    graph_edges: int
    audit_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def detect_contradictions(symbols: list[Symbol]) -> list[str]:
    """
    Find symbols that are near-duplicates of another symbol.

    Two symbols "contradict" (or are redundant) if more than
    CONTRADICTION_THRESHOLD of their combined unique keywords are shared.

    Overlap ratio = |A ∩ B| / |A ∪ B|   (Jaccard similarity)

    If the ratio exceeds CONTRADICTION_THRESHOLD, both names are flagged.
    Deduplication ensures each name appears only once in the result.

    Args:
        symbols: All current symbols.

    Returns:
        Sorted list of symbol names that have a near-duplicate.
    """
    flagged: set[str] = set()
    n = len(symbols)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = symbols[i], symbols[j]
            set_a = set(a.keywords)
            set_b = set(b.keywords)
            union = set_a | set_b
            if not union:
                continue
            overlap_ratio = len(set_a & set_b) / len(union)
            if overlap_ratio > config.CONTRADICTION_THRESHOLD:
                flagged.add(a.name)
                flagged.add(b.name)

    return sorted(flagged)


def audit(symbols: list[Symbol], graph) -> AuditReport:
    """
    Produce a full health report for the current symbol graph.

    Args:
        symbols: All Symbol objects (loaded from DB).
        graph:   GraphManager instance (for edge count).

    Returns:
        An AuditReport dataclass with all health metrics filled in.
    """
    from core.combiner import find_combinable_pairs

    active = [s.name for s in symbols if s.frequency >= config.MIN_CLUSTER_SIZE]
    weak   = [s.name for s in symbols if s.frequency <  config.MIN_CLUSTER_SIZE]

    pairs     = find_combinable_pairs(symbols)
    pair_names = [(a.name, b.name) for a, b in pairs]

    contradictions = detect_contradictions(symbols)

    # Count total edges (each edge stored twice in adjacency dict; divide by 2)
    total_edges = sum(len(v) for v in graph._edges.values()) // 2

    return AuditReport(
        total_symbols=len(symbols),
        active_symbols=active,
        weak_symbols=weak,
        combinable_pairs=pair_names,
        contradiction_candidates=contradictions,
        graph_edges=total_edges,
    )


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.graph import GraphManager

    print("=== auditor.py standalone test ===\n")

    s_recursion = Symbol(
        name="RECURSION",
        keywords=["recursion", "function", "call"],
        member_ids=[1, 2, 3],
        frequency=4,
        created_at="2025-01-01T10:00:00+00:00",
    )
    # Near-duplicate of RECURSION (high keyword overlap → contradiction)
    s_refers = Symbol(
        name="REFERS",
        keywords=["recursion", "function", "call"],  # same as RECURSION -> Jaccard=1.0
        member_ids=[4, 5],
        frequency=2,
        created_at="2025-01-01T10:05:00+00:00",
    )
    s_weather = Symbol(
        name="WEATHER",
        keywords=["weather", "forecast"],
        member_ids=[6, 7, 8],
        frequency=3,
        created_at="2025-01-01T10:10:00+00:00",
    )

    g = GraphManager()
    g.add_symbol(s_recursion)
    g.add_symbol(s_refers)
    g.add_symbol(s_weather)

    report = audit([s_recursion, s_refers, s_weather], g)

    print(f"  total_symbols:           {report.total_symbols}")
    print(f"  active_symbols:          {report.active_symbols}")
    print(f"  weak_symbols:            {report.weak_symbols}")
    print(f"  combinable_pairs:        {report.combinable_pairs}")
    print(f"  contradiction_candidates:{report.contradiction_candidates}")
    print(f"  graph_edges:             {report.graph_edges}")

    assert report.total_symbols == 3
    assert "RECURSION" in report.active_symbols
    assert "WEATHER" in report.active_symbols
    assert "REFERS" in report.weak_symbols
    assert "REFERS" in report.contradiction_candidates
    assert "RECURSION" in report.contradiction_candidates

    # detect_contradictions standalone
    contras = detect_contradictions([s_recursion, s_refers, s_weather])
    assert "REFERS" in contras and "RECURSION" in contras
    assert "WEATHER" not in contras
    print(f"\n  detect_contradictions -> {contras}")

    print("\n✓ auditor.py standalone test passed.")
