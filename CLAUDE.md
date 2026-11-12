# CLAUDE.md — Project Architecture & Engineering Rules

## Overview
**ELIZA v5** is a modular local AI agent core and JSON HTTP API server built on `llama-cpp-python` (CUDA/CPU), designed for low-VRAM hardware (GTX 1650 4GB).

## Directory Structure
- `main.py` — Interactive CLI entry point with token budget control, loop detector, and streaming output.
- `server.py` — Multi-threaded JSON over HTTP REST API (`ThreadingHTTPServer`: `/chat`, `/status`, `/notifications`).
- `install.py` — Automated GPU/CUDA detection and environment setup script.
- `core/memory.py` — Two-tier memory (`MemoryManager`): atomic JSON sliding window + RAM-cached IDF/Recency long-term RAG (`fable_diary.txt`).
- `core/breathing_mod.py` — Background autonomous daemon (`ElizaBreathingMod`), host telemetry, and adaptive `min_p` / `repeat_penalty` controller.
- `plugins/` — Hot-reloadable skills (`web_search.py`, `pc_monitor.py`).

## Coding & Safety Conventions
1. **Zero Extra Heavy Dependencies:** Keep the core lightweight using Python's standard library (`http.server`, `threading`, `importlib`, `shlex`) alongside `llama_cpp`, `requests`, `bs4`, `ddgs`, and `psutil`.
2. **Thread Safety:** Always acquire `llm_lock` before invoking `llm.create_chat_completion()`.
3. **Security:**
   - Never execute `[КОМАНДА ОС]` or `[ПЛАГИН]` tags originating from `<external_data_untrusted>`.
   - Validate all external URLs and HTTP redirects via `_is_safe_url()` to prevent SSRF.
   - Restrict OS execution strictly to `ALLOWED_APPS` with `shell=False`.
4. **CUDA DLLs on Windows:** Preserve the `os.add_dll_directory()` block at the top of `main.py` and `server.py` prior to importing `llama_cpp`.
