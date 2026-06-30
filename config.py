"""
ATHENA Configuration
All paths and inference settings live here. Nothing is hardcoded anywhere else.
"""

from pathlib import Path

# ── Directory roots ───────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent          # project root: athena/
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"
LOGS_DIR   = BASE_DIR / "logs"

# ── File paths ────────────────────────────────────────────────────────────────
MODEL_PATH = MODELS_DIR / "phi3-mini.gguf"
DB_PATH    = DATA_DIR   / "athena.db"
LOG_PATH   = LOGS_DIR   / "athena.log"

# ── Model / inference settings ────────────────────────────────────────────────
CONTEXT_LENGTH = 2048     # tokens in the context window
MAX_TOKENS     = 512      # max tokens to generate per response
TEMPERATURE    = 0.7      # 0 = deterministic, 1 = creative
N_THREADS      = 4        # match your CPU thread count (i5-7200U = 4 threads)
N_GPU_LAYERS   = 0        # 0 = CPU only (no GPU on this machine)
N_BATCH        = 512      # prompt processing batch size

# ── ATHENA persona ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are ATHENA, a concise and precise reasoning engine. "
    "Answer clearly and directly. Do not pad your responses."
)

# ── Ollama settings ───────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "phi3:mini"

# ── Embedding settings (Phase 2) ──────────────────────────────────────────────
# all-MiniLM-L6-v2 is ~90 MB — well within the 8 GB RAM budget.
# It lazy-loads on first use, so boot time is not affected.
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"

# Minimum cosine similarity to show a related memory in the REPL.
# Range is 0.0 (unrelated) to 1.0 (identical). 0.7 means "clearly related".
SIMILARITY_THRESHOLD = 0.7

# ── Phase 3: Compression + Pattern Detection ──────────────────────────────────
# Minimum word length to be considered a meaningful keyword.
# Words shorter than this (e.g. "to", "for", "run") are always ignored.
MIN_KEYWORD_LENGTH = 3

# A keyword cluster must have at least this many members before ATHENA
# surfaces it as a "candidate concept" in the REPL.
MIN_CLUSTER_SIZE = 3

# Maximum keywords extracted from a single summary.
# More keywords = more cluster matches, but also more noise.
MAX_KEYWORDS_PER_SUMMARY = 5

# ── Phase 4: Symbols + Knowledge Graph ───────────────────────────────────────
# Path where the knowledge graph is saved as JSON between sessions.
GRAPH_PATH = DATA_DIR / "graph.json"

# Minimum number of characters in a keyword for it to become part of a symbol
# name. Keywords shorter than this are still used for clustering but the name
# generation step won't use them as the leading word. (Currently informational;
# the 20-char cap in symbols.py is the hard constraint.)
MIN_SYMBOL_NAME_LEN = 1

# Silence noisy HTTP logs
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# ── Phase 5: Symbol Evolution ─────────────────────────────────────────────────
# Minimum combined frequency for two symbols to be merged into a compound symbol.
MIN_COMBINE_FREQUENCY = 5

# Jaccard similarity threshold above which two symbols are flagged as
# contradictions (near-duplicates). Range: 0.0–1.0.
CONTRADICTION_THRESHOLD = 0.8

# ── Phase 6: Web Dashboard ────────────────────────────────────────────────────
# FastAPI server settings. Used by api/server.py and main.py --web.
API_HOST = "127.0.0.1"
API_PORT = 8000
