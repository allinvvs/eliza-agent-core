# Project: Fable5 (Autonomous Eliza)

## Security & Safety Boundaries (STRICT)
- NEVER run destructive shell commands (rm -rf, del, format, taskkill on system processes, registry edits).
- NEVER modify, move, or delete models/model.gguf, able_diary.txt, or chat_memory.json.
- Operate strictly inside D:\fable5. Do not touch files outside this directory.
- Before editing able5.py, always explain the diff and preserve all core-protection checks.

## Hardware & Runtime Constraints
- Target machine: Intel i5-11400F (6 cores), GTX 1650 (4GB VRAM), 16GB RAM.
- Inference backend: Ollama at http://localhost:11434 (coder-fast for code edits, able5 for persona).
