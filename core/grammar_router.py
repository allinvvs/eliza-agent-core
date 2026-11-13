from functools import lru_cache
import json
import logging

try:
    from llama_cpp import LlamaGrammar
except ImportError:
    LlamaGrammar = None

logger = logging.getLogger("eliza.grammar")

_JSON_OBJECT_GBNF = r"""
root   ::= ws "{" ws members ws "}" ws
members ::= pair ( ws "," ws pair )*
pair   ::= string ws ":" ws value
value  ::= string | number | "true" | "false" | "null"
string ::= "\"" ( [^"\\\n\r] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\""
number ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )?
ws     ::= [ \t\n]*
""".strip()


@lru_cache(maxsize=1)
def _get_json_grammar():
    """Compiles standard RFC-8259 flat JSON object grammar once per process."""
    if LlamaGrammar is None:
        return None
    return LlamaGrammar.from_string(_JSON_OBJECT_GBNF, verbose=False)


def route_intent_with_grammar(
    llm_instance,
    llm_lock,
    user_input: str,
    plugins: dict,
) -> tuple[str | None, str]:
    """Routes user intent via universal BPE-safe GBNF JSON grammar."""
    if not plugins or LlamaGrammar is None:
        return None, user_input

    grammar = _get_json_grammar()
    if grammar is None:
        return None, user_input

    valid_keys = ["none"] + sorted(plugins.keys())
    tools_desc = "\n".join(
        f'- "{key}": {data["name"]} (примеры: {", ".join(data["triggers"][:5])})'
        for key, data in plugins.items()
    )
    prompt = (
        "<start_of_turn>user\n"
        "Ты — JSON-маршрутизатор инструментов. Выбери подходящий плагин для запроса пользователя.\n"
        f"Доступные плагины:\n{tools_desc}\n"
        '- "none": обычный разговор, если плагин не нужен.\n'
        f"Разрешённые значения поля tool: {', '.join(valid_keys)}.\n"
        f"Запрос пользователя: {user_input}\n"
        'Верни JSON с ключами "tool", "confidence" ("low"|"medium"|"high") и "query".\n'
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    raw_json = ""
    try:
        with llm_lock:
            resp = llm_instance(
                prompt,
                max_tokens=96,
                temperature=0.0,
                grammar=grammar,
                stop=["<end_of_turn>", "<eos>", "<|im_end|>"],
            )
        choices = resp.get("choices") or []
        first_choice = next(iter(choices), {})
        raw_json = first_choice.get("text", "").strip()
        parsed = json.loads(raw_json)

        selected_tool = str(parsed.get("tool", "none")).strip()
        confidence = str(parsed.get("confidence", "low")).strip().lower()
        extracted_query = str(parsed.get("query", "")).strip() or user_input

        if selected_tool != "none" and selected_tool in plugins and confidence in ("high", "medium"):
            logger.info(f"GBNF Router selected tool='{selected_tool}' (conf={confidence}), query='{extracted_query}'")
            return selected_tool, extracted_query
    except Exception as e:
        logger.warning(f"GBNF routing fallback triggered: {e} | raw={raw_json!r}")

    return None, user_input
