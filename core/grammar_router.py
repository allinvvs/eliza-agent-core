from functools import lru_cache
import json
import logging

try:
    from llama_cpp import LlamaGrammar
except ImportError:
    LlamaGrammar = None

logger = logging.getLogger("eliza.grammar")

_STRUCTURED_COT_GBNF = r"""
root   ::= ws "{" ws "\"reason\"" ws ":" ws string ws "," ws "\"tool\"" ws ":" ws string ws "," ws "\"query\"" ws ":" ws string ws "}" ws
string ::= "\"" ( [^"\\\n\r] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\""
ws     ::= [ \t\n]*
""".strip()


@lru_cache(maxsize=1)
def _get_json_grammar():
    """Compiles strict ordered CoT JSON grammar ('reason' -> 'tool' -> 'query')."""
    if LlamaGrammar is None:
        return None
    return LlamaGrammar.from_string(_STRUCTURED_COT_GBNF, verbose=False)


def _invoke_router_llm(llm_instance, messages: list[dict], grammar) -> str:
    """Supports both native Llama chat templates (Gemma/Qwen/Llama) and raw callables."""
    if hasattr(llm_instance, "create_chat_completion"):
        resp = llm_instance.create_chat_completion(
            messages=messages,
            max_tokens=110,
            temperature=0.0,
            grammar=grammar,
            stop=["<end_of_turn>", "<eos>", "<|im_end|>"],
            stream=False,
        )
        choices = resp.get("choices") or []
        first = next(iter(choices), {})
        return first.get("message", {}).get("content", "").strip()

    sys_text = next(iter(messages), {}).get("content", "")
    user_text = messages[-1].get("content", "")
    prompt = f"{sys_text}\n\nЗапрос: {user_text}\n"
    resp = llm_instance(prompt, max_tokens=110, temperature=0.0, grammar=grammar)
    choices = resp.get("choices") or []
    first = next(iter(choices), {})
    return first.get("text", "").strip()

#Блок 2. Контекстно-зависимый роутер route_intent_with_grammar

def route_intent_with_grammar(
    llm_instance,
    llm_lock,
    user_input: str,
    plugins: dict,
    last_assistant_msg: str = "",
) -> tuple[str | None, str]:
    """Routes user intent via GBNF-enforced Chain-of-Thought ('reason' before 'tool')."""
    if not plugins or LlamaGrammar is None:
        return None, user_input

    grammar = _get_json_grammar()
    if grammar is None:
        return None, user_input

    valid_keys = ["none"] + sorted(plugins.keys())
    tools_desc = "\n".join(f'- "{k}": {v["name"]}' for k, v in plugins.items())

    context_hint = ""
    if last_assistant_msg.strip():
        snippet = last_assistant_msg.strip()[:160].replace("\n", " ")
        context_hint = f'Предыдущая реплика ассистента: "{snippet}"\n'

    messages = [
        {
            "role": "system",
            "content": (
                "Ты — классификатор намерений. Определи, требует ли запрос РЕАЛЬНЫХ ВНЕШНИХ ДАННЫХ через плагин.\n"
                "Правило: обсуждение кода, архитектуры, характера Элизы, её модулей, планов или обычный диалог — ВСЕГДА tool=\"none\".\n"
                "Плагин нужен ТОЛЬКО для актуальных внешних фактов или текущих метрик физического хоста.\n"
                f"Плагины:\n{tools_desc}\n"
                f"Допустимые tool: {', '.join(valid_keys)}.\n"
                'Верни JSON: {"reason": "суть намерения в 4-6 слов", "tool": "имя_или_none", "query": "запрос"}.'
            ),
        },
        {"role": "user", "content": f"{context_hint}Запрос пользователя: {user_input}"},
    ]

    raw_json = ""
    try:
        with llm_lock:
            raw_json = _invoke_router_llm(llm_instance, messages, grammar)
        parsed = json.loads(raw_json)

        reason = str(parsed.get("reason", "")).strip()
        selected_tool = str(parsed.get("tool", "none")).strip()
        extracted_query = str(parsed.get("query", "")).strip() or user_input

        if selected_tool != "none" and selected_tool in plugins:
            logger.info(f"GBNF Router -> tool='{selected_tool}' | reason='{reason}' | query='{extracted_query}'")
            return selected_tool, extracted_query
        logger.debug(f"GBNF Router -> none | reason='{reason}'")
    except Exception as e:
        logger.warning(f"GBNF routing fallback triggered: {e} | raw={raw_json!r}")

    return None, user_input
