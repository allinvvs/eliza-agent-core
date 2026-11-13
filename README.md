# ELIZA v5 — Modular Local AI Agent & JSON/SSE HTTP Core

[![ELIZA Core CI](https://github.com/allinvvs/eliza-agent-core/actions/workflows/tests.yml/badge.svg)](https://github.com/allinvvs/eliza-agent-core/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/allinvvs/eliza-agent-core/actions/workflows/tests.yml)
[![Hardware](https://img.shields.io/badge/VRAM-4GB%20CUDA%20Optimized-green)](https://github.com/allinvvs/eliza-agent-core/actions/workflows/tests.yml)

**ELIZA v5** is a lightweight, modular local AI agent engine and REST/SSE API server built on `llama-cpp-python` (`llama.cpp`). Engineered specifically for consumer low-VRAM hardware (tested on **NVIDIA GTX 1650 4GB**), it combines a RAM-cached two-tier memory engine, GBNF-constrained intent routing, real-time SSE token streaming, native NVML GPU telemetry, OS command sandboxing, and a lexical-diversity adaptive inference controller.

---

## Core Architecture & Engineering Highlights

* **Single-Source Agent Core & Token-Exact Budgeting (`core/engine.py`):**
  * **Unified Event-Driven Engine:** Consolidates CLI (`eliza`), native REST/SSE, and OpenAI-compatible endpoints into a single thread-safe pipeline with Flash Attention (`flash_attn=True`) and per-turn latency telemetry (`rag_ms`, `ttft_ms`, `tokens_per_sec`).
  * **Zero-False-Positive Tool Execution:** Replaces brittle keyword triggers with deterministic slash commands (`/pc`, `/web`, `/plugins`, `/metrics`) and single-pass inline tool execution, backed by an LRU-cached GBNF Structured-CoT router (`core/grammar_router.py`).
  * **Native Tokenizer Context Guard:** Measures exact token counts via `llama.tokenize()` before every turn to guarantee context window safety (`n_ctx=3072`) without truncating system policies.
* **Two-Tier Hybrid Memory Engine (`core/memory.py`):**
  * **Short-Term Sliding Window:** Token-budgeted dialogue buffer with atomic JSON persistence (`.tmp` + `os.replace`) to prevent file corruption on power loss or interrupts.
  * **Long-Term Ring-Buffered RAG:** Automatic semantic retrieval powered by an in-memory cached **IDF + Russian/English morphological stemmer** with **temporal recency decay** (zero external vector DB or extra VRAM overhead).
* **Adaptive Inference Controller & Background Worker (`core/breathing_mod.py`):**
  * Thread-safe background daemon (`threading.Lock` + `threading.Event`) monitoring system telemetry and idle periods.
  * Real-time lexical diversity feedback loop (**Type-Token Ratio**) that dynamically tunes `min_p` and `repeat_penalty`—staying out of the way during natural conversation (`penalty ~1.03`) and intervening gently when repetition occurs.
* **Hot-Reloadable Plugin Architecture (`plugins/`):**
  * Dynamic module discovery via `importlib` without restarting the LLM process.
  * **Native NVML Hardware & VRAM Telemetry (`plugins/pc_monitor.py`):** Reads GPU name, core load, chip temperature (`°C`), and exact VRAM usage directly via NVIDIA's C-API (`pynvml`) in `< 1 ms` without spawning subprocesses, with automatic fallback to `nvidia-smi` and headless CPU environments.
  * **Intent-Routed Web & Media Search (`plugins/web_search.py`):** Routes queries across Open-Meteo API, Wikipedia REST API, and DuckDuckGo with **SSRF protection** (including post-redirect URL validation) and isolated temporary file handling.
* **Security & Sandboxing (`main.py`, `server.py`):**
  * **Indirect Prompt Injection Defense:** Wraps untrusted external plugin outputs inside `<external_data_untrusted>` boundaries with strict system policy isolation.
  * **OS Execution Sandbox:** Strict application allowlist (`ALLOWED_APPS`), `shlex` tokenization (`shell=False`), and canonical path traversal protection.
* **Zero-Dependency JSON, SSE & OpenAI-Compatible HTTP API (`server.py`):**
  * Multi-threaded server built on `ThreadingHTTPServer` with socket timeout guards (`30s`) and daemon workers, exposing both native `/chat` (sync & SSE) and OpenAI-compatible `/v1/chat/completions` + `/v1/models` endpoints with on-the-fly `<thought>` filtering.
  

---

## Hardware Benchmarks (NVIDIA GTX 1650 4GB)

| Subsystem / Metric | Implementation | Measured Latency / Footprint |
| :--- | :--- | :--- |
| **GPU & VRAM Telemetry** | Direct NVML C-API (`pynvml`) | **~0.5 ms** (vs ~150 ms `nvidia-smi` spawn) |
| **Long-Term IDF RAG Lookup** | RAM-cached Russian/English stemmer | **~2.3 ms** per turn (zero external vector DB) |
| **Deterministic Slash Routing** | Prefix command dispatcher (`/pc`, `/web`) | **< 0.1 ms** (zero LLM overhead) |
| **Unit Test Suite (CI)** | `unittest` with mocked NVML/LLM layers | **~15 ms** (6 tests, zero GPU required) |
| **VRAM Budget** | Quantized GGUF (`n_ctx=3072`, `n_batch=512`, `flash_attn=True`) | Fits cleanly within **4096 MB VRAM** |

---

## Repository Structure

```text
eliza-agent-core/
├── .github/
│   └── workflows/
│       └── tests.yml        # Automated GitHub Actions CI unit test runner
├── core/
│   ├── __init__.py
│   ├── breathing_mod.py     # Background daemon, telemetry & adaptive Min-P/TTR tuner
│   ├── engine.py            # Single-source AgentEngine, token budgeting & turn metrics
│   ├── grammar_router.py    # LRU-cached GBNF Structured-CoT intent router
│   └── memory.py            # Atomic short-term history & RAM-cached IDF long-term RAG
├── plugins/
│   ├── pc_monitor.py        # Native NVML C-API GPU/VRAM/Temp & host telemetry
│   ├── sandbox.py           # Isolated OS command execution & path guard
│   └── web_search.py        # Intent-routed search (Weather, Wiki, Web) with SSRF guard
├── tests/
│   ├── __init__.py
│   └── test_core.py         # 6 headless unit tests (RAG, GBNF, AgentEngine, TTR & NVML)
├── .env.example             # Environment configuration template
├── .gitattributes           # Cross-platform LF line-ending enforcement
├── .gitignore
├── CLAUDE.md                # Architectural guidelines for AI-assisted development
├── install.py               # Hardware diagnostics & environment helper script
├── main.py                  # Lightweight interactive CLI adapter (`eliza`)
├── pyproject.toml           # PEP 621 package metadata, optional CUDA/CPU groups & CLI scripts
└── server.py                # Multi-threaded JSON, SSE & OpenAI-compatible server (`eliza-server`)
```


---

## Quick Start

### 1. PEP 621 Package Installation
Clone the repository and install it in editable mode with either CUDA or CPU extras:
```bash
git clone https://github.com/allinvvs/eliza-agent-core.git
cd eliza-agent-core

# For NVIDIA GPUs (includes pynvml C-API telemetry & llama-cpp-python):
pip install -e ".[cuda]"

# Or for CPU-only environments:
pip install -e ".[cpu]"
```

### 2. Model Configuration
Copy `.env.example` to `.env`, place any quantized `.gguf` model (e.g., Gemma, Qwen, or Llama) into `models/model.gguf`, and adjust `MODEL_GPU_LAYERS` according to your available VRAM.

### 3. Launch Modes
* **Interactive CLI Mode:**
  ```bash
  eliza
  ```
* **JSON & SSE Streaming HTTP Server:**
  ```bash
  eliza-server
  ```
* **Run Unit Tests (No GPU/Model Required):**
  ```bash
  python -m unittest discover -s tests -v
  ```

---

## HTTP API Reference (`server.py`)

* `POST /chat` — Execute an inference turn with cascaded GBNF plugin routing and memory retrieval:
  ```json
  { "message": "Check GPU chip temperature and free VRAM" }
  ```
* `POST /chat` (**SSE Token Streaming**) — Set `"stream": true` to receive real-time `text/event-stream` chunks while filtering internal `<thought>` blocks on the fly:
  ```json
  { "message": "Explain CPU pipeline stages briefly", "stream": true }
  ```
* `GET /status` — Returns active plugins, memory statistics, and current adaptive sampler parameters.
* `GET /notifications` — Retrieves autonomous background observations generated during idle periods.
* `POST /v1/chat/completions` (**OpenAI-Compatible API**) — Drop-in endpoint for Open WebUI, Continue, and standard OpenAI SDK clients (supports both sync JSON and `"stream": true` SSE chunks).
* `GET /v1/models` — Lists the local `eliza-v5-local` model for OpenAI-compatible client auto-discovery.


