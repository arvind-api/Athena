# ATHENA — Setup Guide (Windows 10)

## What you need before starting
- Python 3.11 (not 3.12 — llama-cpp-python wheels are more reliable on 3.11)
- Git (optional but useful)
- ~5 GB free disk space (model is ~2.4 GB)
- A terminal: use **Command Prompt** or **PowerShell**

---

## Step 1 — Get Python 3.11

1. Go to https://www.python.org/downloads/release/python-3119/
2. Download **Windows installer (64-bit)**
3. Run it. Check **"Add Python to PATH"** before clicking Install.
4. Verify: open a terminal and run:
   ```
   python --version
   ```
   You should see `Python 3.11.x`.

---

## Step 2 — Place the project folder

Put the `athena/` folder anywhere you like. For example:
```
C:\Users\YourName\projects\athena\
```

Open a terminal and navigate there:
```cmd
cd C:\Users\YourName\projects\athena
```

---

## Step 3 — Create a virtual environment

```cmd
python -m venv venv
venv\Scripts\activate
```

Your prompt should now start with `(venv)`.

---

## Step 4 — Install dependencies

```cmd
pip install -r requirements.txt
```

> **llama-cpp-python note:** The version in requirements.txt ships a pre-built
> CPU wheel for Windows — no C++ compiler needed. If pip can't find a wheel
> for your exact Python version, install it manually:
>
> ```cmd
> pip install llama-cpp-python --prefer-binary
> ```

---

## Step 5 — Download the Phi-3 Mini model

1. Go to: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
2. Find the file: **Phi-3-mini-4k-instruct-q4.gguf**
   (or any Q4_K_M variant — look for ~2.4 GB)
3. Click the download icon next to it.
4. Move the downloaded file into:
   ```
   athena\models\phi3-mini.gguf
   ```
   Rename it exactly `phi3-mini.gguf`.

   Your folder should look like:
   ```
   athena\
   └── models\
       └── phi3-mini.gguf   ← here
   ```

---

## Step 6 — Verify each component independently

Run these from the `athena\` folder with the virtualenv active.

### 6a. Database layer (no model needed)
```cmd
python database\sqlite.py
```
Expected output ends with: `✓ sqlite.py standalone test passed.`

### 6b. Memory layer (no model needed)
```cmd
python core\memory.py
```
Expected output ends with: `✓ memory.py standalone test passed.`

### 6c. Model loader (needs the GGUF file)
```cmd
python core\model.py
```
This loads the model and asks it to say hello. It will be slow the first time
(10–30 seconds). Expected output ends with: `✓ model.py standalone test passed.`

### 6d. Reasoning engine (needs the GGUF file)
```cmd
python core\reasoning.py
```
Expected: prints an answer to "What is recursion?" with timing info.

---

## Step 7 — Run ATHENA

```cmd
python main.py
```

You'll see the boot sequence in the terminal. When it says `ATHENA ready`, type:

```
You: What is recursion?
```

Expected output:
```
ATHENA: Recursion is [explanation…]

Database: Stored successfully
ATHENA Memory Count: 1
```

That's Phase 1 complete. ✓

---

## Step 8 — (Optional) Start the API server

Only do this after Step 7 works.

```cmd
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Then in a browser or curl:
```
GET  http://localhost:8000/health
POST http://localhost:8000/ask   {"message": "What is recursion?"}
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Model file not found` | Check the GGUF is at `athena\models\phi3-mini.gguf` |
| `pip install` fails on llama-cpp-python | Try `pip install llama-cpp-python --prefer-binary` |
| Very slow inference | Normal on CPU — Phi-3 Mini Q4 takes 1–5 min per response on i5-7200U |
| `ModuleNotFoundError` | Make sure venv is active: `venv\Scripts\activate` |
| DB errors in logs | Check `athena\data\` folder exists — it's created automatically |
| RAM > 6 GB | Don't run a browser or other apps alongside ATHENA |

---

## Memory and RAM expectations

| Component | RAM |
|---|---|
| Phi-3 Mini Q4_K_M model | ~2.4 GB |
| Python + llama-cpp-python | ~300 MB |
| SQLite + MemoryManager | < 50 MB |
| OS headroom | ~1.5 GB |
| **Total** | **~4.3 GB** |

You have 8 GB, so there's comfortable headroom.

---

## Where to find logs

All errors and events are written to:
```
athena\logs\athena.log
```

If something breaks, check there first.
