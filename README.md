# ELIZA v5 — Modular Local AI Agent & JSON/SSE HTTP Core

[![ELIZA Core CI](https://github.com/allinvvs/eliza-agent-core/actions/workflows/tests.yml/badge.svg)](https://github.com/allinvvs/eliza-agent-core/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/allinvvs/eliza-agent-core/actions/workflows/tests.yml)
[![Hardware](https://img.shields.io/badge/VRAM-4GB%20CUDA%20Optimized-green)](https://github.com/allinvvs/eliza-agent-core/actions/workflows/tests.yml)

**ELIZA v5** is a lightweight, modular local AI agent engine and REST/SSE API server built on `llama-cpp-python` (`llama.cpp`). Engineered specifically for consumer low-VRAM hardware (tested on **NVIDIA GTX 1650 4GB**), it combines a RAM-cached two-tier memory engine, GBNF-constrained intent routing, real-time SSE token streaming, native NVML GPU telemetry, OS command sandboxing, and a lexical-diversity adaptive inference controller.

---

## Core Architecture & Engineering Highlights

* **Cascaded GBNF Intent Router (`core/grammar_router.py`):**
  * **Zero-Latency Fast Path:** Explicit trigger matching executes in `0 ms` without extra LLM calls.
  * **BPE-Safe GBNF Fallback:** For implicit or conversational tool requests, an LRU-cached RFC-8259 JSON grammar constrains `llama.cpp` sampling to emit 100% valid JSON with confidence gating (`high`/`medium`), eliminating malformed tool calls and hallucinated plugin names.
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
* **Zero-Dependency JSON & SSE Streaming HTTP API (`server.py`):**
  * Multi-threaded server built on Python's standard `ThreadingHTTPServer` supporting both synchronous JSON responses and real-time **Server-Sent Events (`text/event-stream`)** with on-the-fly `<thought>` tag filtering for reasoning models (e.g., Gemma).

---

## Hardware Benchmarks (NVIDIA GTX 1650 4GB)

| Subsystem / Metric | Implementation | Measured Latency / Footprint |
| :--- | :--- | :--- |
| **GPU & VRAM Telemetry** | Direct NVML C-API (`pynvml`) | **~0.5 ms** (vs ~150 ms `nvidia-smi` spawn) |
| **Explicit Tool Routing** | Compiled keyword/prefix matcher | **< 0.1 ms** (zero LLM overhead) |
| **Implicit Tool Routing** | Cached GBNF JSON sampler (`max_tokens=96`) | **100% valid JSON**, confidence-gated |
| **Unit Test Suite (CI)** | `unittest` with mocked NVML/LLM layers | **~13 ms** (5 tests, zero GPU required) |
| **VRAM Budget** | Quantized GGUF (`n_ctx=3072`, `n_gpu_layers=24`) | Fits cleanly within **4096 MB VRAM** |

---

## Repository Structure

```text
eliza-agent-core/
├── core/
│   ├── __init__.py
│   ├── breathing_mod.py   # Background daemon, telemetry & adaptive Min-P/TTR tuner
│   ├── engine.py          # Core LLM initialization & inference helpers
│   ├── grammar_router.py  # LRU-cached GBNF JSON intent router with confidence gating
│   └── memory.py          # Atomic short-term history & RAM-cached IDF long-term RAG
├── plugins/
│   ├── pc_monitor.py      # Native NVML C-API GPU/VRAM/Temp & host telemetry
│   ├── sandbox.py         # Isolated OS command execution & path guard
│   └── web_search.py      # Intent-routed search (Weather, Wiki, Web) with SSRF guard
├── tests/
│   └── test_core.py       # Headless unit tests for RAG, GBNF router, TTR & telemetry
├── pyproject.toml         # PEP 621 package metadata, optional CUDA/CPU groups & CLI scripts
├── main.py                # Interactive CLI entry point (`eliza`)
├── server.py              # Multi-threaded JSON & SSE HTTP API entry point (`eliza-server`)
├── install.py             # Hardware diagnostics & environment helper script
├── CLAUDE.md              # Architectural guidelines for AI-assisted development
├── .env.example           # Environment configuration template
└── .gitignore
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

