import sys, os, py_compile, requests
from pydantic import BaseModel

class LineEdit(BaseModel):
    start_line: int
    end_line: int
    new_code: str

class PatchPlan(BaseModel):
    edits: list[LineEdit] = __import__("pydantic").Field(min_length=1)

if len(sys.argv) < 3:
    print("Использование: python agent.py <файл> '<задача>'")
    sys.exit(1)

target, task = sys.argv[1], sys.argv[2]
with open(target, "r", encoding="utf-8") as f:
    lines = f.read().replace("\r\n", "\n").split("\n")

numbered = "\n".join(f"{i+1:03d} | {line}" for i, line in enumerate(lines))
prompt = (
    f"Выполни задачу для файла {target}: {task}\n"
    f"Укажи номера строк (start_line..end_line включительно, начиная с 1) и new_code (без номеров строк, с правильными отступами Python).\n"
    f"Если нужно вставить импорты в начало — замени строку 1..1 на импорты + старую строку 1.\n\n"
    f"Код с номерами строк:\n{numbered}"
)

print("[Agent] Генерирую патч по номерам строк...")
r = requests.post("http://localhost:11434/api/chat", json={
    "model": "coder-fast",
    "format": PatchPlan.model_json_schema(),
    "messages": [{"role": "user", "content": prompt}],
    "stream": False
}, timeout=90)

plan = PatchPlan.model_validate_json(r.json()["message"]["content"])
# Применяем правки снизу вверх, чтобы не сбивать нумерацию верхних строк
for e in sorted(plan.edits, key=lambda x: x.start_line, reverse=True):
    new_lines = e.new_code.replace("\r\n", "\n").split("\n")
    lines[e.start_line - 1 : e.end_line] = new_lines

result = "\n".join(lines)
tmp = f"{target}.tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.write(result)

py_compile.compile(tmp, doraise=True)
os.replace(tmp, target)
print(f"[OK] Патч применён по {len(plan.edits)} блокам! Синтаксис Python проверен.")
