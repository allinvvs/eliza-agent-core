# ELIZA v5.1 — Modular Local AI Agent & Hardened JSON/SSE HTTP Core

[![ELIZA Core CI](https://github.com/allinvvs/eliza-agent-core/actions/workflows/tests.yml/badge.svg)](https://github.com/allinvvs/eliza-agent-core/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Hardware](https://img.shields.io/badge/Hardware-16GB%20RAM%20%2B%204GB%20VRAM-76b900)
![Security](https://img.shields.io/badge/Security-AST%20Sandbox%20%26%20SSRF-success)

**ELIZA v5.1** is a lightweight, modular local AI agent engine and hardened REST/SSE API server built on `llama-cpp-python` (`llama.cpp`). Engineered specifically for **16GB RAM + low-VRAM consumer hardware** (tested on **NVIDIA GTX 1650 4GB + 16GB RAM** with a custom **`Q4_K_M` / `Q5_K_M` GGUF quantization** of [`huihui-ai/Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated`](https://huggingface.co/huihui-ai/Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated)), it combines an in-memory **Okapi BM25** two-tier memory engine, token-level GBNF intent routing, real-time SSE token streaming, native NVML GPU telemetry, an AST-isolated Python & OS micro-sandbox, and a lexical-diversity adaptive inference controller.

---

## Core Architecture & Engineering Highlights

* **Single-Source Agent Core & Token-Exact Budgeting (`core/engine.py`):**
  * **Unified Event-Driven Engine:** Consolidates CLI (`eliza`), native REST/SSE, and OpenAI-compatible endpoints into a single thread-safe pipeline with Flash Attention (`flash_attn=True`), prefix KV-cache optimization, and per-turn latency telemetry (`rag_ms`, `ttft_ms`, `tokens_per_sec`).
  * **Zero-False-Positive Tool Execution:** Combines deterministic slash commands (`/pc`, `/web`, `/py`, `/sandbox`, `/plugins`, `/metrics`) with a dynamic token-sampler-enforced GBNF Structured-CoT router (`core/grammar_router.py`).
  * **Native Tokenizer Context Guard:** Measures exact token counts via `llama.tokenize()` and `llama.detokenize()` before every turn to guarantee context window safety (`n_ctx=3072`) without truncating system policies.
* **Two-Tier Hybrid Memory Engine (`core/memory.py`):**
  * **Short-Term Sliding Window:** Token-budgeted dialogue buffer with atomic JSON persistence (`.tmp` + `os.replace`) to prevent file corruption on power loss or interrupts.
  * **Long-Term Ring-Buffered Okapi BM25 RAG:** Automatic semantic retrieval powered by an incremental RAM-cached **Okapi BM25 + bilingual (RU/EN) morphological stemmer**, **Jaccard semantic deduplication**, and **temporal recency decay** (zero external vector DB or extra VRAM overhead).
* **Zero-Contention Adaptive Inference Controller (`core/breathing_mod.py`):**
  * Monotonic-clock background daemon (`threading.Lock` + non-blocking `acquire(blocking=False)`) monitoring host telemetry and idle periods without ever delaying active user turns.
  * Real-time lexical diversity feedback loop (**Type-Token Ratio**) that dynamically tunes `min_p` and `repeat_penalty`—staying unobtrusive during natural conversation (`penalty ~1.03`) and intervening gently when repetition occurs.
* **Hot-Reloadable Plugin Architecture (`plugins/`):**
  * Dynamic module discovery via `importlib` without restarting the LLM process.
  * **Native NVML Hardware & VRAM Telemetry (`plugins/pc_monitor.py`):** Lazy-initialized NVIDIA C-API (`pynvml`) reader with per-sensor fault isolation (`< 1 ms`), system-drive pagefile monitoring, and automatic fallbacks to `nvidia-smi` and `ctypes` Windows memory APIs.
  * **Zero-Trust OS & Python Micro-Sandbox (`plugins/sandbox.py`):** Executes pure computational Python snippets (`py: ...`) inside an isolated subprocess (`-I -B -S`, stripped `env={}`) with AST inspection (blocking imports, dunders, and CPU/RAM exponentiation bombs), plus canonical path-traversal-proof workspace actions (`mkdir`, `ls`, `write`, `read`).
  * **Intent-Routed Web & Media Search (`plugins/web_search.py`):** Routes queries across Open-Meteo API, Wikipedia REST API, and DuckDuckGo with DNS-resolving **SSRF protection** (blocking private/loopback IPs on every redirect hop) and binary magic-byte verification for images.
* **Hardened JSON, SSE & OpenAI-Compatible HTTP API (`server.py`):**
  * Multi-threaded server built on `ThreadingHTTPServer` with request tracing (`X-Request-ID`), **DNS-Rebinding `Host` header validation**, optional constant-time **Bearer token authentication (`ELIZA_API_KEY`)**, non-blocking VRAM queue backpressure (`429`), and graceful signal shutdown (`SIGINT` / `SIGTERM`).

---

## Hardware Benchmarks (NVIDIA GTX 1650 4GB)

| Subsystem / Metric | Implementation | Measured Latency / Footprint |
| :--- | :--- | :--- |
| **GPU & VRAM Telemetry** | Lazy-cached NVML C-API (`pynvml`) | **~0.5 ms** (vs ~150 ms `nvidia-smi` spawn) |
| **Long-Term BM25 RAG Lookup** | Incremental RAM-cached Okapi BM25 + Jaccard | **~1.8 ms** per turn (zero external vector DB) |
| **Deterministic Slash Routing** | Prefix command dispatcher (`/pc`, `/web`, `/py`) | **< 0.1 ms** (zero LLM overhead) |
| **Unit & Security Test Suite** | `unittest` (`test_core.py` + `test_server.py`) | **13 tests** (zero GPU or network required) |
| **VRAM Budget** | Quantized GGUF (`n_ctx=3072`, `n_batch=512`, `flash_attn=True`) | Fits cleanly within **4096 MB VRAM** |

---

## Repository Structure

```text
eliza-agent-core/
├── .github/
│   └── workflows/
│       └── tests.yml        # Automated GitHub Actions CI unit & security test runner
├── core/
│   ├── __init__.py
│   ├── breathing_mod.py     # Zero-contention background daemon & adaptive Min-P/TTR tuner
│   ├── engine.py            # Single-source AgentEngine, nonce prompt guards & token budgeting
│   ├── grammar_router.py    # Sampler-enforced GBNF Structured-CoT intent router
│   └── memory.py            # Atomic short-term history & incremental Okapi BM25 RAG
├── plugins/
│   ├── pc_monitor.py        # Lazy NVML C-API GPU/VRAM/Temp & system drive telemetry
│   ├── sandbox.py           # Zero-trust AST Python micro-runtime & path-guarded workspace
│   └── web_search.py        # Intent-routed search (Weather, Wiki, Web) with DNS SSRF guard
├── tests/
│   ├── __init__.py
│   ├── test_core.py         # Core, Okapi BM25 concurrency, NVML, AST sandbox & SSRF tests
│   └── test_server.py       # HTTP security, DNS-rebinding, Bearer auth, 429 & SSE tests
├── .env.example             # Environment & optional ELIZA_API_KEY configuration template
├── .gitattributes           # Cross-platform LF line-ending enforcement
├── .gitignore
├── CLAUDE.md                # Architectural guidelines for AI-assisted development
├── install.py               # Hardware diagnostics & environment helper script
├── main.py                  # Lightweight interactive CLI adapter (`eliza`)
├── pyproject.toml           # PEP 621 package metadata, optional CUDA/CPU groups & CLI scripts
└── server.py                # Hardened JSON, SSE & OpenAI-compatible server (`eliza-server`)
```

---

## Quick Start

All commands below can be run in **Windows PowerShell**, **VS Code Terminal** (`Ctrl + ~`), or **Linux/macOS Bash**.

### 1. Clone & Create an Isolated Virtual Environment

    git clone https://github.com
    cd eliza-agent-core
    python -m venv .venv

    # Windows (PowerShell):
    .venv\Scripts\Activate.ps1
    # Linux / macOS:
    source .venv/bin/activate

### 2. Install Package (PEP 621)

    # NVIDIA GPU (includes pynvml C-API telemetry & CUDA-ready llama-cpp-python):
    pip install -e ".[cuda]"

    # CPU-only machines (no NVIDIA GPU):
    pip install -e ".[cpu]"

### 3. Download a Quantized GGUF Model (16GB RAM Optimized)

> **Note:** Model weights are excluded from Git due to file size limits.

1. Open the base model card [`huihui-ai/Huihui-gemma-4-12B-coder-fable5-composer2.5-v1-abliterated`](https://huggingface.co) on Hugging Face and click **Quantizations (5 models)** in the right sidebar to grab a pre-built GGUF release.
2. Download the **`Q4_K_M.gguf`** file (~7.5 GB, fits cleanly into **16GB RAM + 4GB VRAM** alongside Windows/Linux) and save it as **`models/model.gguf`**.
3. Copy `.env.example` to `.env` and tune `ELIZA_N_GPU_LAYERS` for your GPU:
   * **CPU only:** `ELIZA_N_GPU_LAYERS=0`
   * **4 GB VRAM (e.g. GTX 1650):** `ELIZA_N_GPU_LAYERS=14` – `20` (hybrid GPU + 16GB RAM offload)
   * **8–12+ GB VRAM (e.g. RTX 3060 / 4070):** `ELIZA_N_GPU_LAYERS=-1` (100% GPU offload)

### 4. Launch & Verify

    # Interactive CLI Chat (supports /pc, /web, /py, /sandbox, /plugins, /metrics):
    eliza

    # Hardened JSON, SSE & OpenAI-Compatible API Server (http://127.0.0.1:8000):
    eliza-server

    # Run Offline Unit & Security Test Suite (13 tests, zero GPU/network needed):
    python -m unittest discover -s tests -v

---

## Connecting External UIs & IDE Extensions

Подключение OpenAI-совместимых интерфейсов (**Open WebUI**, **SillyTavern**, **Continue** в VS Code):
* **Base URL:** `http://127.0.0`
* **Model ID:** `eliza-v5-local`
* **API Key:** опционально (`ELIZA_API_KEY` в `.env`)

---

## HTTP API Reference (`server.py`)

* `POST /chat` — Синхронный режим с роутингом команд (`/pc`, `/web`, `/py`), GBNF-интентами и телеметрией.
* `POST /chat` (`"stream": true`) — Потоковый SSE-стриминг в реальном времени с фильтрацией `<thought>`.
* `POST /v1/chat/completions` — Эндпоинт, совместимый с API OpenAI (JSON и SSE).
* `GET /v1/models` — Список доступных моделей (`eliza-v5-local`).
* `GET /health` — Проверка состояния сервиса (`200 OK`).
* `GET /status` — Активные плагины, метрики памяти Okapi BM25 и параметры сэмплера.
* `GET /notifications` — Вывод фоновых наблюдений, накопленных в режиме простоя.

---

## Author & Project Philosophy

Разработано и поддерживается **ElizaDev** (`@allinvvs`) для локального ИИ и систем с ограниченной видеопамятью.

* **Open-Source:** Патчи и оптимизации выходят по мере развития проекта.
* **Поддержка:** Поддержать тесты оборудования можно через **USDT (TRC20)**: `TVkMpQQVDmBMhB4UGsvY6RAVKWjYBbeGc8`


