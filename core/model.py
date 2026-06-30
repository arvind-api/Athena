"""
ATHENA — core/model.py
Calls Ollama's local API instead of llama-cpp-python.
Ollama must be running before ATHENA starts.

Run standalone:
    python core/model.py
"""

import logging
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class LocalModel:
    def __init__(self):
        self._ready = False
        self._use_gemini = False
        self._gemini_api_key = None

    def load(self) -> bool:
        import os
        self._gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if self._gemini_api_key:
            logger.info("GEMINI_API_KEY environment variable detected. Using Gemini API fallback.")
            self._use_gemini = True
            self._ready = True
            return True

        try:
            response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama is running and reachable.")
                self._ready = True
                return True
            else:
                logger.error("Ollama responded with status %s", response.status_code)
                return False
        except requests.exceptions.ConnectionError:
            logger.error(
                "Cannot reach Ollama at %s. Is Ollama running?", config.OLLAMA_BASE_URL
            )
            return False

    def generate(self, prompt: str) -> str:
        if not self._ready:
            raise RuntimeError("Model is not loaded. Call load() first.")

        if self._use_gemini:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self._gemini_api_key}"
            payload = {
                "contents": [
                    {"parts": [{"text": prompt}]}
                ],
                "generationConfig": {
                    "temperature": config.TEMPERATURE,
                    "maxOutputTokens": config.MAX_TOKENS,
                }
            }
            try:
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                res_json = response.json()
                answer = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                return answer
            except Exception as e:
                logger.exception("Gemini API inference failed: %s", e)
                raise

        payload = {
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.TEMPERATURE,
                "num_predict": config.MAX_TOKENS,
                "num_ctx": config.CONTEXT_LENGTH,
            },
        }

        try:
            response = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            logger.exception("Inference failed: %s", e)
            raise

    @property
    def is_loaded(self) -> bool:
        return self._ready


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model = LocalModel()
    if not model.load():
        print("ERROR: Ollama not reachable. Make sure Ollama is running.")
        sys.exit(1)

    response = model.generate("What is recursion? Answer in one sentence.")
    print("Response:", response)
    print("\n✓ model.py standalone test passed.")