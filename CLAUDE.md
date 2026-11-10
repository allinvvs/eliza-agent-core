# Project: Fable5 (Autonomous Eliza)

## Hardware & Runtime Constraints
- Target machine: Intel i5-11400F (6 cores), GTX 1650 (4GB VRAM), 16GB RAM.
- Keep memory footprint tight: never load two heavy GGUF models simultaneously.
- Inference backend: Ollama at http://localhost:11434 (models: able5 for persona/reasoning, qwen2.5-coder:7b for fast code edits).

## Core Files (DO NOT CORRUPT)
- able5.py: Main assistant loop, core protection logic, web search (ddgs + s4), and memory manager.
- able_diary.txt: Persistent personal diary of Eliza (append-only, preserve UTF-8 encoding).
- chat_memory.json: Structured long-term dialogue memory (always validate JSON before writing).
- models/model.gguf: Quantized 8.9GB weights (never modify or delete).

## Coding Rules
- Language: Python 3.11+, UTF-8 everywhere.
- Before modifying able5.py, preserve all existing core-protection checks and sandbox boundaries.
