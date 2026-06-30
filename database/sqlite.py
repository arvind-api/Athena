"""
ATHENA — database/sqlite.py
All SQLite operations. No business logic — only raw DB access.

Phase 3 adds: compressed_summary + keywords columns, get_all_keywords().
Migration is safe: ALTER TABLE errors are silently caught (standard SQLite pattern).

Run standalone:  python database/sqlite.py
"""

import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

_CREATE_SYMBOLS_TABLE = """
CREATE TABLE IF NOT EXISTS symbols (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,
    keywords    TEXT NOT NULL,
    member_ids  TEXT NOT NULL,
    frequency   INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);
"""

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS interactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT    NOT NULL,
    user_input          TEXT    NOT NULL,
    summary             TEXT    NOT NULL,
    answer              TEXT    NOT NULL,
    metadata            TEXT    NOT NULL DEFAULT '{}',
    embedding           TEXT             DEFAULT NULL,
    compressed_summary  TEXT             DEFAULT NULL,
    keywords            TEXT             DEFAULT NULL
);
"""
_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_summary ON interactions (summary);"


class Database:
    """SQLite connection + CRUD. Single-threaded for Phase 1–3."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.DB_PATH
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> bool:
        """Open the DB, create schema, run migration. Returns True on success."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute(_CREATE_TABLE)
            self._conn.execute(_CREATE_INDEX)
            self._conn.execute(_CREATE_SYMBOLS_TABLE)
            self._conn.commit()
            self._migrate()
            return True
        except sqlite3.Error as e:
            logger.error("Database connect failed: %s", e)
            return False

    def _migrate(self) -> None:
        """Add any missing columns to existing databases — silently ignores duplicates."""
        for col_name, col_def in [
            ("embedding",          "TEXT DEFAULT NULL"),
            ("compressed_summary", "TEXT DEFAULT NULL"),
            ("keywords",           "TEXT DEFAULT NULL"),
        ]:
            try:
                self._conn.execute(f"ALTER TABLE interactions ADD COLUMN {col_name} {col_def};")
                self._conn.commit()
            except sqlite3.OperationalError:
                pass   # column already exists — normal on all runs after the first

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def is_connected(self) -> bool:
        return self._conn is not None

    # ── Write ─────────────────────────────────────────────────────────────────

    def insert_interaction(
        self,
        user_input: str,
        summary: str,
        answer: str,
        metadata: dict,
        embedding: Optional[list[float]] = None,
        compressed_summary: Optional[str] = None,
        keywords: Optional[list[str]] = None,
    ) -> int:
        """Insert one interaction row. Returns the new row's ID."""
        if not self.is_connected:
            raise RuntimeError("Database is not connected.")
        cursor = self._conn.execute(
            """INSERT INTO interactions
               (timestamp, user_input, summary, answer, metadata,
                embedding, compressed_summary, keywords)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                user_input, summary, answer,
                json.dumps(metadata),
                json.dumps(embedding)  if embedding  is not None else None,
                compressed_summary,
                json.dumps(keywords)   if keywords   is not None else None,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_interaction(self, row_id: int) -> Optional[dict]:
        """Fetch one row by ID. Returns a dict or None."""
        if not self.is_connected:
            raise RuntimeError("Database is not connected.")
        row = self._conn.execute(
            "SELECT * FROM interactions WHERE id = ?", (row_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_all_embeddings(self) -> list[dict]:
        """Return [{id, embedding}] for every row that has an embedding stored."""
        if not self.is_connected:
            raise RuntimeError("Database is not connected.")
        rows = self._conn.execute(
            "SELECT id, embedding FROM interactions WHERE embedding IS NOT NULL"
        ).fetchall()
        result = []
        for row in rows:
            try:
                result.append({"id": row["id"], "embedding": json.loads(row["embedding"])})
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Skipping row %d — bad embedding: %s", row["id"], e)
        return result

    def get_all_keywords(self) -> list[dict]:
        """Return [{id, keywords, timestamp}] for every row that has keywords stored."""
        if not self.is_connected:
            raise RuntimeError("Database is not connected.")
        rows = self._conn.execute(
            "SELECT id, keywords, timestamp FROM interactions WHERE keywords IS NOT NULL"
        ).fetchall()
        result = []
        for row in rows:
            try:
                result.append({
                    "id": row["id"],
                    "keywords": json.loads(row["keywords"]),
                    "timestamp": row["timestamp"],
                })
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Skipping row %d — bad keywords: %s", row["id"], e)
        return result

    def count_interactions(self) -> int:
        if not self.is_connected:
            raise RuntimeError("Database is not connected.")
        return self._conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]

    # ── Symbols (Phase 4) ─────────────────────────────────────────────────────

    def add_symbol(self, symbol) -> int:
        """Insert a new symbol row. Returns the new ID. Raises on duplicate name."""
        if not self.is_connected:
            raise RuntimeError("Database is not connected.")
        cursor = self._conn.execute(
            """INSERT INTO symbols (name, keywords, member_ids, frequency, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                symbol.name,
                json.dumps(symbol.keywords),
                json.dumps(symbol.member_ids),
                symbol.frequency,
                symbol.created_at,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_all_symbols(self) -> list:
        """Return all symbol rows as Symbol dataclass instances."""
        if not self.is_connected:
            raise RuntimeError("Database is not connected.")
        # Import here to avoid circular imports at module load time
        from core.symbols import Symbol
        rows = self._conn.execute(
            "SELECT name, keywords, member_ids, frequency, created_at FROM symbols"
        ).fetchall()
        result = []
        for row in rows:
            try:
                result.append(Symbol(
                    name=row["name"],
                    keywords=json.loads(row["keywords"]),
                    member_ids=json.loads(row["member_ids"]),
                    frequency=row["frequency"],
                    created_at=row["created_at"],
                ))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Skipping corrupt symbol row '%s': %s", row["name"], e)
        return result

    def symbol_exists(self, name: str) -> bool:
        """Return True if a symbol with this name is already in the DB."""
        if not self.is_connected:
            raise RuntimeError("Database is not connected.")
        row = self._conn.execute(
            "SELECT 1 FROM symbols WHERE name = ?", (name,)
        ).fetchone()
        return row is not None

    def update_symbol_frequency(self, name: str, new_frequency: int) -> None:
        """Update the frequency of an existing symbol."""
        if not self.is_connected:
            raise RuntimeError("Database is not connected.")
        self._conn.execute(
            "UPDATE symbols SET frequency = ? WHERE name = ?", (new_frequency, name)
        )
        self._conn.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d["metadata"])
        except (json.JSONDecodeError, KeyError):
            d["metadata"] = {}
        d.pop("embedding", None)   # raw embedding bytes not needed by callers
        return d


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    test_path = config.DATA_DIR / "test_sqlite_phase3.db"
    test_path.unlink(missing_ok=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    db = Database(db_path=test_path)
    assert db.connect()

    r1 = db.insert_interaction("What is recursion?",
         "Q: What is recursion? | A: Recursion is a function calling itself.",
         "Recursion is a function that calls itself.", {"duration_ms": 800},
         embedding=[0.1]*384, compressed_summary="What is recursion?...",
         keywords=["recursion", "function", "calling", "base"])
    r2 = db.insert_interaction("What is the weather?",
         "Q: What is the weather? | A: I cannot check weather.",
         "I don't have access to real-time weather.", {"duration_ms": 300})

    kw = db.get_all_keywords()
    assert len(kw) == 1 and kw[0]["id"] == r1 and "recursion" in kw[0]["keywords"]
    print(f"get_all_keywords(): {kw[0]['keywords']}")

    db.close()
    db2 = Database(db_path=test_path)
    assert db2.connect() and db2.count_interactions() == 2
    db2.close()
    test_path.unlink(missing_ok=True)
    print("✓ sqlite.py standalone test passed.")
