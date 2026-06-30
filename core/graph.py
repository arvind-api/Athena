"""
ATHENA — core/graph.py
Knowledge graph: a simple in-memory dict-of-dicts (no external libraries).

Nodes are symbol names (strings).
Edges connect two symbols that share at least one keyword.

The graph is saved as JSON after every interaction so it survives restarts.

Run standalone:  python core/graph.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.symbols import Symbol

logger = logging.getLogger(__name__)

# ── PHASE 5 HOOK ──────────────────────────────────────────────────────────────
# Phase 5 will combine related symbols into higher-order concepts.
# The hook lives in add_symbol(), right after edges are drawn:
#
#   # Phase 5: check if two connected nodes share enough keywords to merge
#   for neighbor in self.get_related_symbols(symbol.name):
#       from core.combiner import should_combine
#       if should_combine(self._nodes[symbol.name], self._nodes[neighbor]):
#           from core.combiner import combine_symbols
#           combined = combine_symbols(...)
#           self.add_symbol(combined)
#
# No changes to the GraphManager interface are needed for Phase 5.
# ─────────────────────────────────────────────────────────────────────────────


class GraphManager:
    """
    In-memory knowledge graph stored as a plain dict.

    Structure:
        _nodes  = { "RECURSION": {"keywords": [...], "frequency": 5, ...} }
        _edges  = { "RECURSION": {"TAIL_CALL": True, ...}, ... }
    """

    def __init__(self):
        # Each entry: symbol_name -> dict with symbol attributes
        self._nodes: dict[str, dict] = {}
        # Adjacency: symbol_name -> set of connected symbol names
        self._edges: dict[str, set] = {}

    def add_symbol(self, symbol: Symbol) -> None:
        """
        Add a symbol as a node, then connect it to any existing node
        that shares at least one keyword.
        """
        name = symbol.name

        # Store node data (serialize keywords list)
        self._nodes[name] = {
            "keywords":   symbol.keywords,
            "frequency":  symbol.frequency,
            "member_ids": symbol.member_ids,
            "created_at": symbol.created_at,
        }

        # Ensure the node has an adjacency entry
        if name not in self._edges:
            self._edges[name] = set()

        # Draw edges to existing nodes that share at least 1 keyword
        my_keywords = set(symbol.keywords)
        for other_name, other_data in self._nodes.items():
            if other_name == name:
                continue
            if my_keywords & set(other_data["keywords"]):  # intersection not empty
                self._edges[name].add(other_name)
                if other_name not in self._edges:
                    self._edges[other_name] = set()
                self._edges[other_name].add(name)

    def get_related_symbols(self, symbol_name: str) -> list[str]:
        """Return names of all symbols directly connected to symbol_name."""
        return list(self._edges.get(symbol_name, set()))

    def to_dict(self) -> dict:
        """Serialize graph to a JSON-safe dict."""
        return {
            "nodes": self._nodes,
            "edges": {k: sorted(v) for k, v in self._edges.items()},
        }

    def save(self, path: Path) -> None:
        """Write graph to a JSON file. Creates parent directories if needed."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
            logger.debug("Graph saved to %s (%d nodes)", path, len(self._nodes))
        except OSError as e:
            logger.error("Graph save failed: %s", e)

    def load(self, path: Path) -> None:
        """Load graph from a JSON file. Silently ignores missing/corrupt files."""
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._nodes = data.get("nodes", {})
            self._edges = {k: set(v) for k, v in data.get("edges", {}).items()}
            logger.info("Graph loaded: %d nodes from %s", len(self._nodes), path)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Graph load failed (%s) — starting empty", e)
            self._nodes = {}
            self._edges = {}

    def merge_symbols(self, name_a: str, name_b: str, new_symbol: Symbol) -> None:
        """
        Replace two existing nodes with one compound node.

        Transfers all edges from both old nodes to the new one, then removes
        the originals. Safe to call even if one name is missing.

        Args:
            name_a:     Name of the first symbol to remove.
            name_b:     Name of the second symbol to remove.
            new_symbol: The compound Symbol that replaces both.
        """
        # Collect all neighbours of both old nodes (excluding each other)
        old_neighbours: set[str] = set()
        for old in (name_a, name_b):
            for neighbour in self._edges.get(old, set()):
                if neighbour not in {name_a, name_b}:
                    old_neighbours.add(neighbour)

        # Remove old nodes and clean up their edge references
        for old in (name_a, name_b):
            self._nodes.pop(old, None)
            self._edges.pop(old, None)
            for adj in self._edges.values():
                adj.discard(old)

        # Add the new compound node (add_symbol draws edges by keyword match too)
        self.add_symbol(new_symbol)

        # Re-wire any neighbours that shared edges with the removed nodes
        new_name = new_symbol.name
        for neighbour in old_neighbours:
            if neighbour in self._nodes and neighbour != new_name:
                self._edges[new_name].add(neighbour)
                if neighbour not in self._edges:
                    self._edges[neighbour] = set()
                self._edges[neighbour].add(new_name)

    def prune_symbol(self, name: str) -> None:
        """
        Remove a weak symbol node and all its edges.

        Does NOT touch the database — symbol history is preserved in SQLite.
        Safe to call if the name does not exist.

        Args:
            name: Symbol name to remove from the graph.
        """
        self._nodes.pop(name, None)
        self._edges.pop(name, None)
        for adj in self._edges.values():
            adj.discard(name)
        logger.debug("Pruned symbol '%s' from graph.", name)

    def get_graph_stats(self) -> dict:
        """
        Return a summary of graph topology.

        Returns:
            {
              "nodes": int,
              "edges": int,                  # unique edges (not doubled)
              "most_connected": str | None,  # node with the most neighbours
              "isolated_nodes": list[str],   # nodes with zero edges
            }
        """
        edge_count = sum(len(v) for v in self._edges.values()) // 2

        most_connected: str | None = None
        max_degree = -1
        isolated: list[str] = []

        for name in self._nodes:
            degree = len(self._edges.get(name, set()))
            if degree == 0:
                isolated.append(name)
            if degree > max_degree:
                max_degree = degree
                most_connected = name

        return {
            "nodes": len(self._nodes),
            "edges": edge_count,
            "most_connected": most_connected,
            "isolated_nodes": isolated,
        }

    def __len__(self) -> int:
        return len(self._nodes)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile
    print("=== graph.py standalone test ===\n")

    from core.symbols import Symbol

    g = GraphManager()

    s_recursion = Symbol(
        name="RECURSION",
        keywords=["recursion", "function"],
        member_ids=[1, 2, 3],
        frequency=3,
        created_at="2025-01-01T10:00:00+00:00",
    )
    s_tail_call = Symbol(
        name="TAIL_CALL",
        keywords=["tail", "call", "function"],
        member_ids=[2, 3, 4],
        frequency=3,
        created_at="2025-01-01T10:10:00+00:00",
    )
    s_weather = Symbol(
        name="WEATHER",
        keywords=["weather", "forecast"],
        member_ids=[5, 6, 7],
        frequency=3,
        created_at="2025-01-01T10:20:00+00:00",
    )

    g.add_symbol(s_recursion)
    g.add_symbol(s_tail_call)
    g.add_symbol(s_weather)

    assert len(g) == 3

    related = g.get_related_symbols("RECURSION")
    assert "TAIL_CALL" in related, f"Expected TAIL_CALL, got {related}"
    assert "WEATHER" not in related

    related_weather = g.get_related_symbols("WEATHER")
    assert related_weather == [], f"Expected no edges from WEATHER, got {related_weather}"

    print(f"  RECURSION is connected to: {related}")
    print(f"  WEATHER is connected to: {related_weather}")

    # Save + load round-trip
    with tempfile.TemporaryDirectory() as tmp:
        graph_path = Path(tmp) / "graph.json"
        g.save(graph_path)

        g2 = GraphManager()
        g2.load(graph_path)
        assert len(g2) == 3
        assert "TAIL_CALL" in g2.get_related_symbols("RECURSION")
        print(f"  Round-trip save/load: {len(g2)} nodes restored.")

    # ── Phase 5 additions ─────────────────────────────────────────────────────
    print("\n--- Phase 5 methods ---")

    stats = g.get_graph_stats()
    assert stats["nodes"] == 3
    assert stats["edges"] == 1
    assert "WEATHER" in stats["isolated_nodes"]
    print(f"  get_graph_stats(): {stats}")

    g.prune_symbol("WEATHER")
    assert len(g) == 2
    assert "WEATHER" not in g._nodes
    print(f"  prune_symbol('WEATHER'): {len(g)} nodes remain")

    compound = Symbol(
        name="RECURSION_TAIL",
        keywords=["recursion", "function", "tail", "call"],
        member_ids=[1, 2, 3, 4],
        frequency=6,
        created_at="2025-01-01T11:00:00+00:00",
    )
    g.merge_symbols("RECURSION", "TAIL_CALL", compound)
    assert "RECURSION" not in g._nodes
    assert "TAIL_CALL" not in g._nodes
    assert "RECURSION_TAIL" in g._nodes
    print(f"  merge_symbols -> nodes: {list(g._nodes.keys())}")

    print("\n✓ graph.py standalone test passed.")
