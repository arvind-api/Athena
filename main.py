"""
ATHENA — main.py
Entry point. Two modes:

  python main.py           → normal interactive REPL (Phases 1-5 output)
  python main.py --web     → start FastAPI dashboard + open browser, then REPL

The --web flag:
  1. Starts uvicorn (api.server) in a background daemon thread
  2. Waits for the server to be ready (polls /api/state)
  3. Opens http://localhost:{API_PORT}/dashboard in the default browser
  4. Continues running the normal REPL in the foreground (fully backward-compatible)

The REPL and web server each boot their own ATHENA components.
They share the same SQLite DB and graph.json on disk.
"""

import logging
import sys
import argparse
import threading
import time
import webbrowser
import urllib.request

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import config

config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("athena.main")

from core.model      import LocalModel
from core.reasoning  import ReasoningEngine
from core.memory     import MemoryManager
from core.graph      import GraphManager
from database.sqlite import Database


# ── REPL boot ─────────────────────────────────────────────────────────────────

def boot() -> tuple[ReasoningEngine, MemoryManager, Database, GraphManager]:
    logger.info("═══ ATHENA boot sequence ═══")

    db = Database()
    if not db.connect():
        logger.warning("Database unavailable — interactions will not be saved.")

    memory = MemoryManager(db)
    print(f"ATHENA Memory Count: {memory.count()}")

    graph = GraphManager()
    graph.load(config.GRAPH_PATH)
    logger.info("Knowledge graph: %d symbol nodes loaded.", len(graph))

    for symbol in db.get_all_symbols():
        if symbol.name not in graph._nodes:
            graph.add_symbol(symbol)

    model = LocalModel()
    if not model.load():
        logger.critical("Model failed to load. Cannot continue.")
        sys.exit(1)

    logger.info("═══ ATHENA ready ═══")
    return ReasoningEngine(model), memory, db, graph


# ── REPL ──────────────────────────────────────────────────────────────────────

def run_repl(engine: ReasoningEngine, memory: MemoryManager,
             graph: GraphManager):
    print("\nATHENA is ready. Type your question (or 'exit' to quit).\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            _run_exit_audit(memory, graph)
            break

        related = memory.search(user_input, limit=1)

        result = engine.think(user_input)
        if not result["success"]:
            print(f"ATHENA: [Error] {result['error']}\n")
            continue

        print(f"\nATHENA: {result['answer']}\n")

        row_id = memory.store(
            user_input=result["input"],
            summary=result["summary"],
            answer=result["answer"],
            metadata={"duration_ms": result["duration_ms"]},
        )
        print("Database: Stored successfully" if row_id > 0
              else "Database: Write failed")
        print(f"ATHENA Memory Count: {memory.count()}")

        if related:
            top = related[0]
            if top.get("similarity", 0.0) >= config.SIMILARITY_THRESHOLD:
                print(f"Related memory: {top['user_input']} "
                      f"(similarity: {top['similarity']:.2f})")

        for concept in memory.get_candidate_concepts():
            if concept.frequency >= config.MIN_CLUSTER_SIZE:
                print(f"Emerging concept: {concept.keywords} "
                      f"(seen {concept.frequency} times)")

        new_symbols = memory.promote_concepts(graph)
        for symbol in new_symbols:
            print(f"New symbol born: {symbol.name} — "
                  "ATHENA named this concept itself")

        if row_id > 0:
            matched_symbol = memory.get_symbol_for_interaction(row_id)
            if matched_symbol:
                all_symbols = memory.db.get_all_symbols()
                first_seen_count = memory.count()
                for s in all_symbols:
                    if s.name == matched_symbol and s.member_ids:
                        first_seen_count = (memory.count()
                                            - min(s.member_ids))
                        break
                print(f"Symbol activated: {matched_symbol} "
                      f"(first seen {first_seen_count} interactions ago)")

        graph.save(config.GRAPH_PATH)
        print()


def _run_exit_audit(memory: MemoryManager, graph: GraphManager) -> None:
    print("\nRunning session audit…")
    try:
        report = memory.run_session_audit(graph)
    except Exception as e:
        logger.error("Session audit failed: %s", e)
        return

    stats    = graph.get_graph_stats()
    n_active = len(report.active_symbols)
    n_weak   = len(report.weak_symbols)

    pairs_str = (
        ", ".join(f"{a} + {b} → {a}_{b[:4]}"
                  for a, b in report.combinable_pairs)
        if report.combinable_pairs else "none"
    )
    contra_str = (
        ", ".join(report.contradiction_candidates)
        if report.contradiction_candidates else "none"
    )

    print("\n═══ ATHENA Session Audit ═══")
    print(f"Symbols: {report.total_symbols} total, "
          f"{n_active} active, {n_weak} weak")
    print(f"Graph: {stats['nodes']} nodes, {stats['edges']} edges")
    print(f"Combinable pairs: {pairs_str}")
    print(f"Contradictions detected: {contra_str}")
    print("═══════════════════════════")

    if report.combinable_pairs:
        graph.save(config.GRAPH_PATH)
        logger.info("Graph saved after symbol combinations.")


# ── Web mode helpers ──────────────────────────────────────────────────────────

def _start_uvicorn() -> None:
    """Start uvicorn in a daemon thread. Called only in --web mode."""
    import uvicorn
    # Use import string so uvicorn can find the app object correctly
    uvicorn.run(
        "server:app",
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="warning",   # reduce noise (ATHENA's own logger is already set)
    )


def _wait_for_server(timeout: int = 30) -> bool:
    """Poll /api/state until the server responds or timeout expires."""
    url = f"http://{config.API_HOST}:{config.API_PORT}/api/state"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ATHENA reasoning engine")
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start the FastAPI dashboard and open it in a browser",
    )
    args = parser.parse_args()

    if args.web:
        print("Starting ATHENA web dashboard…")

        # Boot the web server in a background daemon thread
        server_thread = threading.Thread(
            target=_start_uvicorn,
            daemon=True,   # dies automatically when main thread exits
            name="athena-uvicorn",
        )
        server_thread.start()

        # Wait for the server to come up
        dashboard_url = (
            f"http://{config.API_HOST}:{config.API_PORT}/dashboard"
        )
        print(f"Waiting for server at {dashboard_url} …")
        if _wait_for_server():
            print(f"Server ready. Opening {dashboard_url}")
            webbrowser.open(dashboard_url)
        else:
            print("WARNING: Server did not respond within 30s. "
                  "Open the browser manually.")

        print("\nDashboard is live. REPL also available below.")
        print("─" * 50)

    # Always run the REPL (web mode spawns its own state via uvicorn)
    engine, memory, db, graph = boot()
    try:
        run_repl(engine, memory, graph)
    finally:
        db.close()
        logger.info("ATHENA shut down cleanly.")


if __name__ == "__main__":
    main()
