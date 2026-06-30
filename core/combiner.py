"""
ATHENA — core/combiner.py
Symbol combiner: merges two related symbols into a single compound symbol.

A compound symbol is created when two symbols co-occur frequently enough
and share at least one keyword. Example: RECURSION + SORTING → RECURSION_SORTING

Run standalone:  python core/combiner.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from core.symbols import Symbol

# ── PHASE 6 HOOK ──────────────────────────────────────────────────────────────
# Phase 6 (visualization) will read combinable pairs from this module and draw
# them in the graph UI. Hook in find_combinable_pairs() — after building the
# `pairs` list, emit an event or write to a sidecar file that the visualizer
# polls:
#
#   # Phase 6: export combinable pairs for visualization
#   from core.viz_events import emit
#   emit("combinable_pairs", [(a.name, b.name) for a, b in pairs])
#
# No changes to combine() or find_combinable_pairs() signatures are required.
# ─────────────────────────────────────────────────────────────────────────────


def combine(symbol_a: Symbol, symbol_b: Symbol) -> Symbol | None:
    """
    Merge two symbols into one compound symbol.

    Rules:
      - Both symbols must share at least 1 keyword.
      - Combined frequency must meet MIN_COMBINE_FREQUENCY.
      - Returns None if conditions are not met.

    The new symbol's name is built from the first keyword of each parent,
    uppercase, joined by underscore. Example: RECURSION + SORTING → RECURSION_SORTING

    Args:
        symbol_a: First symbol to combine.
        symbol_b: Second symbol to combine.

    Returns:
        A new compound Symbol, or None if the pair does not qualify.
    """
    shared_keywords = set(symbol_a.keywords) & set(symbol_b.keywords)
    if not shared_keywords:
        return None  # No common ground — cannot combine

    combined_frequency = symbol_a.frequency + symbol_b.frequency
    if combined_frequency < config.MIN_COMBINE_FREQUENCY:
        return None  # Not frequent enough yet

    # Name: first keyword of A + first keyword of B, uppercase, underscore-joined
    part_a = symbol_a.keywords[0].upper() if symbol_a.keywords else symbol_a.name
    part_b = symbol_b.keywords[0].upper() if symbol_b.keywords else symbol_b.name
    name = f"{part_a}_{part_b}"[:20]  # Hard cap matches Symbol spec

    # Keywords: union of both, deduplicated, preserving order of A then B
    seen: set[str] = set()
    merged_keywords: list[str] = []
    for kw in symbol_a.keywords + symbol_b.keywords:
        if kw not in seen:
            seen.add(kw)
            merged_keywords.append(kw)

    # Member IDs: union of both sets
    merged_members = sorted(set(symbol_a.member_ids + symbol_b.member_ids))

    return Symbol(
        name=name,
        keywords=merged_keywords,
        member_ids=merged_members,
        frequency=combined_frequency,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def find_combinable_pairs(symbols: list[Symbol]) -> list[tuple[Symbol, Symbol]]:
    """
    Scan all symbol pairs and return those eligible for combining.

    A pair is eligible when:
      - They share at least 1 keyword.
      - Their combined frequency >= MIN_COMBINE_FREQUENCY.

    Each pair is returned only once (A, B) — not also as (B, A).

    Args:
        symbols: All current symbols (from DB or memory).

    Returns:
        List of (Symbol, Symbol) tuples that can be passed to combine().
    """
    pairs: list[tuple[Symbol, Symbol]] = []
    n = len(symbols)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = symbols[i], symbols[j]
            shared = set(a.keywords) & set(b.keywords)
            if shared and (a.frequency + b.frequency) >= config.MIN_COMBINE_FREQUENCY:
                pairs.append((a, b))
    return pairs


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== combiner.py standalone test ===\n")

    s_recursion = Symbol(
        name="RECURSION",
        keywords=["recursion", "function"],
        member_ids=[1, 2, 3],
        frequency=4,
        created_at="2025-01-01T10:00:00+00:00",
    )
    s_sorting = Symbol(
        name="SORTING",
        keywords=["function", "sort", "algorithm"],
        member_ids=[4, 5, 6],
        frequency=3,
        created_at="2025-01-01T10:10:00+00:00",
    )
    s_weather = Symbol(
        name="WEATHER",
        keywords=["weather", "forecast"],
        member_ids=[7, 8],
        frequency=2,
        created_at="2025-01-01T10:20:00+00:00",
    )

    # Test: combine two qualifying symbols
    result = combine(s_recursion, s_sorting)
    assert result is not None, "Expected a combined symbol"
    assert result.name == "RECURSION_FUNCTION", f"Got: {result.name}"
    assert result.frequency == 7
    assert "function" in result.keywords  # shared keyword appears once
    assert result.keywords.count("function") == 1  # deduplicated
    print(f"  combine(RECURSION, SORTING) -> '{result.name}' freq={result.frequency}")
    print(f"  Keywords: {result.keywords}")
    print(f"  Members:  {result.member_ids}")

    # Test: reject when no shared keywords
    no_combine = combine(s_recursion, s_weather)
    assert no_combine is None, "Should not combine unrelated symbols"
    print(f"\n  combine(RECURSION, WEATHER) -> None  ✓  (no shared keywords)")

    # Test: reject when combined frequency too low
    low_freq = Symbol(
        name="LOWFREQ",
        keywords=["function", "rare"],
        member_ids=[10],
        frequency=1,
        created_at="2025-01-01T11:00:00+00:00",
    )
    no_combine2 = combine(s_weather, low_freq)
    assert no_combine2 is None, "Should not combine when frequency too low"
    print(f"  combine(WEATHER, LOWFREQ) -> None  ✓  (frequency too low)")

    # Test: find_combinable_pairs
    pairs = find_combinable_pairs([s_recursion, s_sorting, s_weather])
    assert len(pairs) == 1
    assert pairs[0][0].name == "RECURSION"
    assert pairs[0][1].name == "SORTING"
    print(f"\n  find_combinable_pairs([RECURSION, SORTING, WEATHER]):")
    for a, b in pairs:
        print(f"    {a.name} + {b.name}")

    print("\n✓ combiner.py standalone test passed.")
