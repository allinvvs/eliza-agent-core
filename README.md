# ELIZA v5 — Modular Local AI Agent & JSON HTTP Core

**ELIZA v5** is a lightweight, modular local AI agent engine and REST API server built on `llama-cpp-python` (`llama.cpp`). Engineered specifically for consumer low-VRAM hardware (tested on **NVIDIA GTX 1650 4GB**), it combines a RAM-cached two-tier memory engine, a lexical-diversity adaptive inference controller, hot-reloadable plugins, OS command sandboxing, and a zero-dependency JSON over HTTP server.

---

## Core Architecture & Engineering Highlights

* **Two-Tier Hybrid Memory Engine (`core/memory.py`):**
  * **Short-Term Sliding Window:** Token-budgeted dialogue buffer with atomic JSON persistence (`.tmp` + `os.replace`) to prevent file corruption on power loss or interrupts.
  * **Long-Term Ring-Buffered RAG:** Automatic semantic retrieval powered by an in-memory cached **IDF + Russian/English morphological stemmer** with **temporal recency decay** (zero external vector DB or extra VRAM overhead).
* **Adaptive Inference Controller & Background Worker (`core/breathing_mod.py`):**
  * Thread-safe background daemon (`threading.Lock` + `threading.Event`) monitoring system telemetry and idle periods.
  * Real-time lexical diversity feedback loop (**Type-Token Ratio**) that dynamically tunes `min_p` and `repeat_penalty`—staying out of the way during natural conversation (`penalty ~1.03`) and intervening gently when repetition occurs.
* **Hot-Reloadable Plugin Architecture (`plugins/`):**
  * Dynamic module discovery via `importlib` without restarting the LLM process.
  * **Intent-Routed Web & Media Search (`plugins/web_search.py`):** Routes queries across Open-Meteo API, Wikipedia REST API, and DuckDuckGo with **SSRF protection** (including post-redirect URL validation) and isolated temporary file handling.
  * **Hardware & VRAM Telemetry (`plugins/pc_monitor.py`):** Non-blocking CPU, RAM, disk, and NVIDIA GPU/VRAM utilization monitoring via `psutil` and `nvidia-smi`.
* **Security & Sandboxing (`main.py`, `server.py`):**
  * **Indirect Prompt Injection Defense:** Wraps untrusted external plugin outputs inside `<external_data_untrusted>` boundaries with strict system policy isolation.
  * **OS Execution Sandbox:** Strict application allowlist (`ALLOWED_APPS`), `shlex` tokenization (`shell=False`), and canonical path traversal protection.
* **Zero-Dependency JSON over HTTP API (`server.py`):**
  * Multi-threaded REST server built on Python's standard `ThreadingHTTPServer` exposing `/chat`, `/status`, and `/notifications`.

---

## Repository Structure

```text
eliza-agent-core/
├── core/
│   ├── __init__.py
│   ├── breathing_mod.py   # Background daemon, telemetry & adaptive Min-P/TTR tuner
│   └── memory.py          # Atomic short-term history & RAM-cached IDF long-term RAG
├── plugins/
│   ├── pc_monitor.py      # Host CPU, RAM, Disk & NVIDIA VRAM telemetry
│   └── web_search.py      # Intent-routed search (Weather, Wiki, Web) with SSRF guard
├── tests/
│   └── test_core.py       # Unit tests for memory RAG & hardware plugins
├── main.py                # Interactive CLI with token budget control & loop detector
├── server.py              # Multi-threaded JSON over HTTP REST API server
├── install.py             # Automated GPU/CUDA detection & environment setup script
├── CLAUDE.md              # Architectural guidelines for AI-assisted development
├── .env.example           # Environment configuration template
├── requirements.txt       # Python dependencies
└── .gitignore
```

---

## Quick Start

### 1. Automated Setup
Clone the repository and run the setup script, which configures GPU acceleration or falls back to CPU automatically:
```bash
git clone https://github.com
cd eliza-agent-core
python install.py
```

### 2. Model Configuration
Place your quantized `.gguf` model (Gemma, Qwen, Llama) into `models/model.gguf` and set `MODEL_GPU_LAYERS` in `.env`.

### 3. Launch Modes
* **Interactive CLI:** `python main.py`
* **HTTP API Server:** `python server.py`
* **Unit Tests:** `python -m unittest discover tests`

### 2. Model Configuration
Place any quantized `.gguf` model (e.g., Gemma, Qwen, or Llama) into `models/model.gguf` and adjust `MODEL_GPU_LAYERS` in `.env` according to your available VRAM.

### 3. Launch Modes
* **Interactive CLI Mode:**
  ```bash
  python main.py
  ```
* **JSON over HTTP API Server:**
  ```bash
  python server.py
  ```
* **Run Unit Tests (No GPU/Model Required):**
  ```bash
  python -m unittest discover tests
  ```

---

## HTTP API Reference (`server.py`)

* `POST /chat` — Execute an inference turn with automatic plugin routing and memory retrieval:
  ```json
  { "message": "Check system hardware and VRAM load" }
  ```
* `GET /status` — Returns active plugins, memory statistics, and current adaptive sampler parameters.
* `GET /notifications` — Retrieves autonomous background observations generated during idle periods.
