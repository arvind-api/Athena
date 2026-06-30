# ATHENA

### *Autonomous Thought Heuristic Engine for Neural Abstraction*

> **An experimental local-first cognitive architecture exploring persistent memory, symbolic abstraction, and self-organizing knowledge structures — built to run entirely on consumer hardware.**
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite)
![Local AI](https://img.shields.io/badge/Runs-100%25%20Local-success)
![CPU Only](https://img.shields.io/badge/CPU-Only-orange)
![Research](https://img.shields.io/badge/Research-Cognitive%20Architecture-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)
---

## Overview

ATHENA is **not another ChatGPT wrapper**.

It is an experimental research project investigating how a local language model can evolve beyond simple prompt-response interactions by combining:

* Persistent long-term memory
* Symbolic knowledge representation
* Knowledge graph construction
* Autonomous memory retrieval
* Cognitive state visualization

Everything runs **locally**.

No cloud.

No API keys.

No GPU required.

No external inference services.

The long-term objective is to explore whether a small local model can gradually build reusable internal knowledge structures instead of repeatedly solving the same problems from scratch.

---

## Current Features

* Local inference using **Phi-3 Mini (GGUF)** via `llama-cpp-python`
* FastAPI backend
* Interactive research dashboard
* SQLite persistent memory
* Memory stream visualization
* Knowledge graph foundation
* Symbol dictionary framework
* Real-time system monitoring
* CPU-only execution

---

## Research Roadmap

ATHENA is being developed incrementally.

### Phase 1 — Local Cognitive Core ✅

* Local LLM
* FastAPI server
* SQLite memory
* Memory logging
* Dashboard

### Phase 2 — Persistent Memory 🚧

* Memory retrieval
* Similarity search
* Context-aware recall

### Phase 3 — Knowledge Graph 🚧

* Symbol relationships
* Graph construction
* Concept linking

### Phase 4 — Symbolic Abstraction 🚧

* Automatic symbol generation
* Concept compression
* Internal representations

### Phase 5 — Cognitive Evolution 🚧

* Symbol refinement
* Knowledge pruning
* Autonomous reasoning improvements

---

## Technology Stack

### AI

* Phi-3 Mini GGUF
* llama-cpp-python

### Backend

* Python
* FastAPI
* SQLite

### Frontend

* HTML
* JavaScript
* D3.js

### Data

* NetworkX
* RDF foundations
* Local file storage

---

## Hardware Requirements

Minimum tested configuration:

* Intel Core i5-7200U
* 8 GB RAM
* Intel HD Graphics 620
* SSD
* Windows 10

ATHENA is intentionally designed for consumer hardware rather than cloud infrastructure.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/arvind-api/Athena.git
cd Athena
```

---

## 2. Create a virtual environment

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `llama-cpp-python` fails to compile on Windows:

```bash
pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

---

## 4. Download the model

The model is not stored in this repository due to its size.

Download:

```
microsoft/Phi-3-mini-4k-instruct-gguf
```

Rename:

```
Phi-3-mini-4k-instruct-q4.gguf
```

to

```
phi3-mini.gguf
```

Place it inside:

```
models/
```

---

## 5. Run ATHENA

```bash
python server.py
```

Open:

```
http://localhost:8000
```

---

## Project Structure

```
athena/
│
├── api/
├── core/
├── database/
├── data/
├── logs/
├── models/
├── server.py
├── config.py
├── requirements.txt
└── index.html
```

---

## Design Principles

ATHENA follows a few simple rules:

* Local-first
* Explainable architecture
* Modular components
* CPU-friendly
* Persistent memory
* Future-proof design
* Research over hype

---

## Current Status

ATHENA is an active research prototype.

Several components—including symbolic abstraction, autonomous symbol evolution, and higher-level cognitive mechanisms—are under active development.

The project prioritizes architectural clarity and experimentation over feature completeness.

---

## License

MIT License

---

> *"Small models deserve long-term memory too."*
