# ATHENA — Autonomous Thought Heuristic Engine for Neural Abstraction

A local AI system with a FastAPI backend and a React dashboard frontend.

## Setup

### 1. Requirements
- Python 3.11 (not 3.12+ — some pinned dependency versions in this repo were generated on a newer Python and may need adjusting otherwise)
- ~5 GB free disk space for the model

### 2. Create a virtual environment
\\\
py -3.11 -m venv venv
venv\Scripts\activate
\\\

### 3. Install dependencies
\\\
pip install -r requirements.txt
\\\

If \llama-cpp-python\ fails to build from source (common on Windows without a C++ compiler), install the prebuilt CPU wheel instead:
\\\
pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
\\\

### 4. Download the model
The model file is not included in this repo (too large for GitHub). Download it with:
\\\
hf download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-q4.gguf --local-dir models
Rename-Item "models\Phi-3-mini-4k-instruct-q4.gguf" "phi3-mini.gguf"
\\\
This places \phi3-mini.gguf\ (~2.4 GB) at \models\phi3-mini.gguf\, which is where the code expects it.

### 5. Run
\\\
python server.py
\\\
Then open http://localhost:8000 in your browser.

## Notes
- On CPU-only machines, expect responses to take 30–60+ seconds.
- The \data/\, \env/\, \dist/\, and model files are gitignored — they're either machine-specific or regenerated on first run.
