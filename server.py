"""
ATHENA — api/server.py
FastAPI backend for the Phase 6 web dashboard.

Endpoints:
  GET  /dashboard         → serves dashboard/index.html
  GET  /api/state         → full current state (memories, graph, symbols, stats)
  GET  /api/graph         → raw graph.json contents as JSON
  POST /api/interact      → send a question; SSE events emitted during processing
  GET  /api/stream        → SSE stream (text/event-stream)

SSE event types emitted on /api/stream:
  {"type": "thinking_start",   "data": {}}
  {"type": "thinking_end",     "data": {"answer": "...", "duration_ms": N}}
  {"type": "memory_stored",    "data": {"id": N, "user_input": "...", "keywords": [...]}}
  {"type": "symbol_born",      "data": {"name": "...", "keywords": [...], "frequency": N}}
  {"type": "concept_emerging", "data": {"keywords": [...], "frequency": N}}
  {"type": "audit_complete",   "data": {...AuditReport fields...}}

Run standalone:
  cd athena/
  python api/server.py
"""

import asyncio
import json
import logging
import sys
import time
import threading
from pathlib import Path
from typing import AsyncGenerator, Optional

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

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
logger = logging.getLogger("athena.server")

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import HTMLResponse, StreamingResponse
# pyrefly: ignore [missing-import]
from pydantic import BaseModel

from core.model      import LocalModel
from core.reasoning  import ReasoningEngine
from core.memory     import MemoryManager
from core.graph      import GraphManager
from database.sqlite import Database

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ATHENA",
    description="Autonomous Thought Heuristic Engine for Neural Abstraction",
    version="6.0.0",
)

# Allow all origins — needed for browser access and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Dashboard HTML path ───────────────────────────────────────────────────────

from fastapi.staticfiles import StaticFiles

_dist_dir = Path(__file__).parent / "dist"
if _dist_dir.exists():
    app.mount("/assets", StaticFiles(directory=_dist_dir / "assets"), name="assets")
    DASHBOARD_HTML = _dist_dir / "index.html"
    logger.info("Serving compiled React frontend from athena/dist")
else:
    DASHBOARD_HTML = Path(__file__).parent / "index.html"
    logger.info("Serving legacy static dashboard from athena/index.html")

# ── ATHENA state (initialised once at startup) ────────────────────────────────

_db:     Optional[Database]        = None
_memory: Optional[MemoryManager]   = None
_graph:  Optional[GraphManager]    = None
_engine: Optional[ReasoningEngine] = None
_start_time: float                 = 0.0
_interact_lock                     = asyncio.Lock()   # one interaction at a time

# ── SSE broadcast ─────────────────────────────────────────────────────────────

# Each connected SSE client gets its own asyncio.Queue.
# broadcast() puts an event into every client's queue.
_sse_clients: list[asyncio.Queue] = []


async def broadcast(event: dict) -> None:
    """Send an SSE event to every connected browser client."""
    for q in _sse_clients:
        await q.put(event)


# ── Startup / shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
def _startup():
    global _db, _memory, _graph, _engine, _start_time
    _start_time = time.time()

    logger.info("═══ ATHENA API boot ═══")

    _db = Database()
    if not _db.connect():
        logger.warning("Database unavailable — interactions will not persist.")

    _memory = MemoryManager(_db)
    logger.info("Memory: %d stored interactions.", _memory.count())

    _graph = GraphManager()
    _graph.load(config.GRAPH_PATH)
    logger.info("Graph: %d symbol nodes loaded.", len(_graph))

    # Re-sync any symbols that exist in DB but not yet in graph
    for symbol in _db.get_all_symbols():
        if symbol.name not in _graph._nodes:
            _graph.add_symbol(symbol)

    model = LocalModel()
    if not model.load():
        logger.error("Model failed to load — /api/interact will return errors.")
    else:
        _engine = ReasoningEngine(model)
        logger.info("Model loaded. ATHENA API ready.")


@app.on_event("shutdown")
def _shutdown():
    if _db:
        _db.close()
    logger.info("ATHENA API stopped.")


# ── Request / response models ─────────────────────────────────────────────────

class InteractRequest(BaseModel):
    question: str


# ── Helper: graph → D3-friendly dict ─────────────────────────────────────────

def _graph_to_d3() -> dict:
    """Convert GraphManager internal structure to {nodes, links} for D3."""
    if not _graph:
        return {"nodes": [], "links": []}

    nodes = []
    for name, data in _graph._nodes.items():
        nodes.append({
            "id":        name,
            "keywords":  data.get("keywords", []),
            "frequency": data.get("frequency", 1),
            "createdAt": data.get("created_at", ""),
        })

    # Deduplicate edges (A→B and B→A are the same link)
    seen   = set()
    links  = []
    for src, neighbours in _graph._edges.items():
        for tgt in neighbours:
            key = tuple(sorted([src, tgt]))
            if key not in seen:
                seen.add(key)
                links.append({"source": src, "target": tgt})

    return {"nodes": nodes, "links": links}


def _recent_memories(limit: int = 20) -> list[dict]:
    """Return the most recent `limit` interactions from DB, newest first."""
    if not _db or not _db.is_connected:
        return []
    try:
        rows = _db._conn.execute(
            """SELECT id, timestamp, user_input, answer, keywords
               FROM interactions ORDER BY id DESC LIMIT ?""",
            (limit,)
        ).fetchall()
        result = []
        for r in rows:
            kw = []
            try:
                kw = json.loads(r["keywords"]) if r["keywords"] else []
            except Exception:
                pass
            result.append({
                "id":         r["id"],
                "timestamp":  r["timestamp"],
                "user_input": r["user_input"],
                "answer":     r["answer"][:200],   # truncate for wire
                "keywords":   kw,
            })
        return result
    except Exception as e:
        logger.error("_recent_memories() failed: %s", e)
        return []


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Serve the single-file dashboard."""
    if not DASHBOARD_HTML.exists():
        raise HTTPException(status_code=404,
                            detail="dashboard/index.html not found")
    return DASHBOARD_HTML.read_text(encoding="utf-8")


@app.get("/api/state")
def api_state():
    """
    Full current state snapshot — called once when the dashboard loads.
    Returns memories (newest first), graph nodes/links, symbol list, stats, and audit.
    """
    uptime = int(time.time() - _start_time)

    symbols = []
    if _db and _db.is_connected:
        for s in _db.get_all_symbols():
            symbols.append({
                "name":      s.name,
                "keywords":  s.keywords,
                "frequency": s.frequency,
            })

    stats = {
        "total_memories": _memory.count() if _memory else 0,
        "total_symbols":  len(symbols),
        "graph_edges":    sum(len(v) for v in _graph._edges.values()) // 2
                          if _graph else 0,
        "uptime_seconds": uptime,
    }

    # Run audit on load to populate the audit panel
    audit_report = None
    if _memory and _graph:
        try:
            audit_report = _memory.run_session_audit(_graph)
        except Exception as e:
            logger.error("Initial state audit failed: %s", e)

    return {
        "memories": _recent_memories(20),
        "graph":    _graph_to_d3(),
        "symbols":  symbols,
        "stats":    stats,
        "audit": {
            "total_symbols": audit_report.total_symbols,
            "active_symbols": audit_report.active_symbols,
            "weak_symbols": audit_report.weak_symbols,
            "combinable_pairs": audit_report.combinable_pairs,
            "contradiction_candidates": audit_report.contradiction_candidates,
            "graph_edges": audit_report.graph_edges,
        } if audit_report else None
    }


@app.get("/api/graph")
def api_graph():
    """Raw graph data (nodes + links) for the D3 force diagram."""
    return _graph_to_d3()


@app.get("/api/stream")
async def api_stream():
    """
    Server-Sent Events stream. The browser connects once and receives
    all real-time events emitted during ATHENA's inference cycle.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()
        _sse_clients.append(queue)
        logger.info("SSE client connected. Total: %d", len(_sse_clients))
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=25.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Keepalive comment — prevents proxy/browser timeout
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            _sse_clients.remove(queue)
            logger.info("SSE client disconnected. Total: %d", len(_sse_clients))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/api/interact")
async def api_interact(req: InteractRequest):
    """
    Main interaction endpoint. Accepts a question, runs the full ATHENA
    inference + memory + symbol pipeline, broadcasting SSE events at each step.
    Returns the final answer.
    """
    if not _engine:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Check logs.")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Serialise concurrent interactions — model is single-threaded
    async with _interact_lock:
        # 1. Notify: thinking started
        await broadcast({"type": "thinking_start", "data": {"question": question}})
        logger.info("Interact: %r", question[:80])

        # 2. Semantic search BEFORE storing (so current question never self-matches)
        related = await asyncio.to_thread(_memory.search, question, 1)

        # 3. Run model inference in thread pool (synchronous, blocks ~14s)
        result = await asyncio.to_thread(_engine.think, question)

        if not result["success"]:
            await broadcast({
                "type": "thinking_end",
                "data": {"answer": f"[Error] {result['error']}", "duration_ms": 0},
            })
            raise HTTPException(status_code=500, detail=result["error"])

        answer = result["answer"]

        # 4. Notify: thinking finished
        await broadcast({
            "type": "thinking_end",
            "data": {
                "answer":      answer,
                "duration_ms": result["duration_ms"],
                "question":    question,
            },
        })

        # 5. Store interaction in memory
        row_id = await asyncio.to_thread(
            _memory.store,
            result["input"],
            result["summary"],
            result["answer"],
            {"duration_ms": result["duration_ms"]},
        )

        # 6. Notify: memory stored
        kw = []
        if row_id > 0 and _db and _db.is_connected:
            row = _db.get_interaction(row_id)
            if row:
                try:
                    kw = json.loads(row.get("keywords") or "[]")
                except Exception:
                    pass
            await broadcast({
                "type": "memory_stored",
                "data": {
                    "id":         row_id,
                    "user_input": question,
                    "answer":     answer[:200],
                    "keywords":   kw,
                    "similarity": (related[0].get("similarity", 0)
                                   if related else 0),
                    "had_related": bool(related and
                                       related[0].get("similarity", 0) >=
                                       config.SIMILARITY_THRESHOLD),
                },
            })

        # 7. Emit any surfaced candidate concepts
        concepts = _memory.get_candidate_concepts()
        for concept in concepts:
            if concept.frequency >= config.MIN_CLUSTER_SIZE:
                await broadcast({
                    "type": "concept_emerging",
                    "data": {
                        "keywords":  concept.keywords,
                        "frequency": concept.frequency,
                    },
                })

        # 8. Promote concepts to symbols; emit symbol_born events
        new_symbols = await asyncio.to_thread(_memory.promote_concepts, _graph)
        for sym in new_symbols:
            await broadcast({
                "type": "symbol_born",
                "data": {
                    "name":      sym.name,
                    "keywords":  sym.keywords,
                    "frequency": sym.frequency,
                },
            })

        # 9. Run session audit to keep graph updated and emit status
        audit_report = None
        try:
            audit_report = await asyncio.to_thread(_memory.run_session_audit, _graph)
            # Broadcast the audit_complete event
            audit_data = {
                "total_symbols": audit_report.total_symbols,
                "active_symbols": audit_report.active_symbols,
                "weak_symbols": audit_report.weak_symbols,
                "combinable_pairs": audit_report.combinable_pairs,
                "contradiction_candidates": audit_report.contradiction_candidates,
                "graph_edges": audit_report.graph_edges,
                "audit_timestamp": audit_report.audit_timestamp,
            }
            await broadcast({"type": "audit_complete", "data": audit_data})
        except Exception as e:
            logger.error("API session audit failed: %s", e)

        # 10. Persist updated graph (which may contain merged symbols from audit)
        await asyncio.to_thread(_graph.save, config.GRAPH_PATH)

        # 11. Build and return response
        return {
            "answer":      answer,
            "duration_ms": result["duration_ms"],
            "memory_id":   row_id,
            "new_symbols": [s.name for s in new_symbols],
            "graph":       _graph_to_d3(),
            "stats": {
                "total_memories": _memory.count(),
                "total_symbols":  len(_graph._nodes),
                "graph_edges":    sum(len(v) for v in _graph._edges.values()) // 2,
                "uptime_seconds": int(time.time() - _start_time),
            },
            "audit": {
                "total_symbols": audit_report.total_symbols,
                "active_symbols": audit_report.active_symbols,
                "weak_symbols": audit_report.weak_symbols,
                "combinable_pairs": audit_report.combinable_pairs,
                "contradiction_candidates": audit_report.contradiction_candidates,
                "graph_edges": audit_report.graph_edges,
            } if audit_report else None
        }


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
    )
