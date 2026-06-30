"""
ATHENA — core/reasoning.py
Takes raw user input, formats it into a Phi-3 prompt, runs inference,
and returns a structured result dict.

Run standalone to verify reasoning (needs model loaded):
    python core/reasoning.py
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from core.model import LocalModel

logger = logging.getLogger(__name__)


def build_prompt(user_input: str, system_prompt: Optional[str] = None) -> str:
    """
    Format a user message into the Phi-3 chat template.
    Phi-3 uses <|system|>, <|user|>, <|assistant|> tokens.
    """
    system = system_prompt or config.SYSTEM_PROMPT
    return (
        f"<|system|>\n{system}<|end|>\n"
        f"<|user|>\n{user_input}<|end|>\n"
        f"<|assistant|>\n"
    )


def summarize_interaction(user_input: str, answer: str) -> str:
    """
    Create a short internal summary of the exchange.
    Phase 1: naive string truncation.
    Phase 2: replace with an embedding-based semantic summary.
    """
    short_input  = user_input[:80].strip()
    short_answer = answer[:120].strip()
    return f"Q: {short_input} | A: {short_answer}"


class ReasoningEngine:
    """
    Orchestrates the full think cycle:
        user_input → prompt → model → answer → summary → result dict
    """

    def __init__(self, model: LocalModel):
        self.model = model

    def think(self, user_input: str) -> dict:
        """
        Process one user turn.

        Returns a dict with:
            input       – original user text
            answer      – model's response
            summary     – short internal summary (for DB storage)
            duration_ms – inference wall-clock time in milliseconds
            success     – True/False
            error       – error message string, or None
        """
        if not user_input or not user_input.strip():
            return self._error_result(user_input, "Empty input received.")

        if not self.model.is_loaded:
            return self._error_result(user_input, "Model is not loaded.")

        prompt = build_prompt(user_input.strip())
        logger.debug("Prompt built (%d chars)", len(prompt))

        t_start = time.perf_counter()
        try:
            answer = self.model.generate(prompt)
        except Exception as e:
            logger.exception("Model inference error")
            return self._error_result(user_input, str(e))
        t_end = time.perf_counter()

        duration_ms = round((t_end - t_start) * 1000)
        summary = summarize_interaction(user_input, answer)

        logger.info("Inference done in %d ms", duration_ms)

        return {
            "input":       user_input,
            "answer":      answer,
            "summary":     summary,
            "duration_ms": duration_ms,
            "success":     True,
            "error":       None,
        }

    @staticmethod
    def _error_result(user_input: str, message: str) -> dict:
        logger.error("Reasoning failed: %s", message)
        return {
            "input":       user_input,
            "answer":      "",
            "summary":     "",
            "duration_ms": 0,
            "success":     False,
            "error":       message,
        }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model = LocalModel()
    if not model.load():
        print("ERROR: Model failed to load.")
        sys.exit(1)

    engine = ReasoningEngine(model)
    result = engine.think("What is recursion?")

    if result["success"]:
        print("Answer:", result["answer"])
        print("Summary:", result["summary"])
        print("Duration:", result["duration_ms"], "ms")
        print("\n✓ reasoning.py standalone test passed.")
    else:
        print("ERROR:", result["error"])
        sys.exit(1)
