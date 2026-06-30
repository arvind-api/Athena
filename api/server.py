"""
ATHENA — api/server.py
FastAPI wrapper. Exposes the reasoning engine over HTTP.
Start only after verifying main.py works.

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.memory import MemoryManager
from core.model import LocalModel
from core.reasoning import ReasoningEngine
from database.sqlite import Database

logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ATHENA",
    description="Autonomous Thought Heuristic Engine for Neural Abstraction",
    version="0.1.0",
)

# Shared state — initialised once at startup
_model   = LocalModel()
_db      = Database()
_memory  = None
_engine  = None


@app.on_event("startup")
def startup():
    global _memory, _engine

    if not _db.connect():
        logger.warning("DB unavailable — memory will not persist.")

    if not _model.load():
        logger.error("Model failed to load. /ask will not work.")
    else:
        _engine = ReasoningEngine(_model)

    _memory = MemoryManager(_db)
    logger.info("ATHENA API ready.")


@app.on_event("shutdown")
def shutdown():
    _db.close()
    logger.info("ATHENA API stopped.")


# ── Request / Response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    message: str


class AskResponse(BaseModel):
    answer:      str
    duration_ms: int
    memory_id:   int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick liveness check."""
    return {
        "status":       "ok",
        "model_loaded": _model.is_loaded,
        "db_connected": _db.is_connected,
        "memory_count": _memory.count() if _memory else 0,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Send a message to ATHENA. Returns the answer and stores the interaction.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    result = _engine.think(request.message)

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])

    memory_id = -1
    if _memory and _db.is_connected:
        memory_id = _memory.store(
            user_input=result["input"],
            summary=result["summary"],
            answer=result["answer"],
            metadata={"duration_ms": result["duration_ms"]},
        )

    return AskResponse(
        answer=result["answer"],
        duration_ms=result["duration_ms"],
        memory_id=memory_id,
    )


@app.get("/memory/{memory_id}")
def get_memory(memory_id: int):
    """Retrieve a stored interaction by ID."""
    if _memory is None:
        raise HTTPException(status_code=503, detail="Memory unavailable.")
    record = _memory.retrieve(memory_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return record


@app.get("/memory/search/{query}")
def search_memory(query: str, limit: int = 5):
    """Search stored interactions by keyword."""
    if _memory is None:
        raise HTTPException(status_code=503, detail="Memory unavailable.")
    return _memory.search(query, limit=limit)
