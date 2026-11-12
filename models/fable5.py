import os
import re
import json
import sys
import logging
import platform
import subprocess
import shlex
import requests
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from llama_cpp import Llama

# Настройка логирования для отслеживания системных событий
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==========================================================
# КОНФИГУРАЦИЯ И ДИНАМИЧЕСКИЕ ПУТИ (Кроссплатформенность)
# ==========================================================
# Автоматически определяет папку, где лежит сам скрипт fable5.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Теперь путь всегда будет правильным: папка_проекта/models/model.gguf
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "model.gguf"))
HISTORY_FILE = os.getenv("HISTORY_FILE", "chat_memory.json")  # Будет создаваться там же, где fable5.py
DIARY_FILE = os.getenv("DIARY_FILE", "fable_diary.txt")        # Будет создаваться там же, где fable5.py
CTX_SIZE = int(os.getenv("CTX_SIZE", "3072"))

MAX_FILE_SIZE = 5 * 1024 * 1024  # Лимит 5 МБ на скачивание медиа файлов
TIMEOUT_TOTAL = (4, 6)           # 4 секунды на подключение, 6 на чтение байт

# Использование константных множеств для мгновенного поиска O(1)
SEARCH_BLACK_LIST = frozenset([
    "chatgpt.com", "openai.com", "t.me", "telegram.org", 
    "instagram.com", "facebook.com", "twitter.com", "x.com"
])
STOP_WORDS = frozenset({'как', 'ты', 'мне', 'что', 'тебе', 'я', 'в', 'на', 'с', 'ли', 'бы', 'или', 'и', 'вы', 'элиза'})
MEMORY_TRIGGERS = frozenset({"помнишь", "прошлое", "дневник", "хроники", "вспомни", "вчера", "расскажи о себе", "параметры"})
IMG_TRIGGERS = frozenset(["фото", "картинка", "изображение", "рисунок", "photo", "image"])
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

IS_WINDOWS = platform.system() == "Windows"

# Настройка путей приложений для разных операционных систем
if IS_WINDOWS:
    ALLOWED_APPS = {
        "notepad.exe": Path(os.environ.get("SystemRoot", "C:\\Windows")) / "System32" / "notepad.exe",
        "calc.exe": Path(os.environ.get("SystemRoot", "C:\\Windows")) / "System32" / "calc.exe",
    }
else:
    ALLOWED_APPS = {
        "notepad.exe": Path("/usr/bin/nano"),
        "calc.exe": Path("/usr/bin/bc"),
    }
# ==========================================================
# БЛОК 0: КРИСТАЛЛЫ ВОСПОМИНАНИЙ И ДОЛГАЯ ПАМЯТЬ
# ==========================================================

def get_memory_crystals(query: str, diary_path: str, top_n: int = 3) -> str:
    """Безопасно извлекает контекстные воспоминания на основе пересечения слов."""
    if not query or not diary_path or not os.path.exists(diary_path):
        return ""
    
    query_lower = query.lower()
    query_words_all = set(re.findall(r'\w+', query_lower))
    
    if not query_words_all.intersection(MEMORY_TRIGGERS):
        return "" 
        
    try:
        with open(diary_path, "r", encoding="utf-8") as f:
            records = [line.strip() for line in f if line.strip()]
    except IOError as e:
        logging.error(f"Не удалось прочитать файл дневника: {e}")
        return ""
        
    if not records:
        return ""
        
    query_words = query_words_all - STOP_WORDS
    if not query_words:
        return ""
    
    scored_records = []
    for record in records:
        clean_record = re.sub(r'[<>]', '', record)
        record_words = set(re.findall(r'\w+', clean_record.lower()))
        matches = query_words.intersection(record_words)
        if matches:
            scored_records.append((len(matches), clean_record))
            
    scored_records.sort(key=lambda x: x[0], reverse=True)
    
    if scored_records:
        best_crystals = [rec[1] for rec in scored_records[:top_n]]
        return "\nТвои контекстные воспоминания из дневника:\n" + "\n".join(best_crystals)
    return ""

def write_to_diary(text: str, diary_path: str) -> bool:
    """Безопасно записывает новую строку в файлы хроник, создавая папки."""
    if not text or not text.strip():
        return False
    try:
        dir_name = os.path.dirname(diary_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        safe_text = text.strip().replace('\n', ' ').replace('\r', '')
        formatted_record = f"[{current_time}] {safe_text}\n"
        with open(diary_path, "a", encoding="utf-8") as f:
            f.write(formatted_record)
        return True
    except Exception as e:
        print(f"\n[Ошибка записи в Хроники: {e}]")
        return False
# ==========================================================
# БЛОК 1: ФУНКЦИИ РАБОТЫ С ИНТЕРНЕТОМ И МЕДИА
# ==========================================================

def fetch_site_content(url: str) -> str:
    """Извлекает и санитизирует текстовое содержимое веб-страницы."""
    if not url:
        return ""
    if any(bad in url.lower() for bad in SEARCH_BLACK_LIST):
        return ""
    try:
        with requests.get(url, headers=HEADERS, timeout=TIMEOUT_TOTAL, stream=True) as response:
            if response.status_code != 200:
                return ""
            cl = response.headers.get('content-length')
            if cl and int(cl) > 2 * 1024 * 1024:
                return ""
                
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            for el in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript", "form"]):
                el.extract()
            clean_text = " ".join([p.strip() for p in soup.get_text(separator=' ').splitlines() if p.strip()])
            return re.sub(r'\s+', ' ', clean_text)[:2000].strip()
    except Exception:
        return ""

def show_image_via_os(image_url: str) -> str:
    """Скачивает изображение с проверкой безопасности и открывает его."""
    if not image_url or not (image_url.startswith("http://") or image_url.startswith("https://")):
        return "\n[Система: Некорректный или небезопасный URL изображения]."
    try:
        with requests.get(image_url, headers=HEADERS, timeout=TIMEOUT_TOTAL, stream=True) as response:
            if response.status_code != 200:
                return "\n[Система: Не удалось скачать файл изображения]."
            
            content_type = response.headers.get('Content-Type', '')
            if 'image' not in content_type.lower():
                return "\n[Система: Ссылка не является валидным изображением]."

            temp_dir = os.environ.get("TEMP", os.getcwd())
            temp_path = os.path.normpath(os.path.join(temp_dir, "eliza_view.jpg"))
            
            size = 0
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    size += len(chunk)
                    if size > MAX_FILE_SIZE:
                        return "\n[Система: Файл превышает лимит размера]."
                    f.write(chunk)
            
            if os.path.exists(temp_path) and temp_path.endswith('.jpg'):
                if IS_WINDOWS:
                    os.startfile(temp_path)
                else:
                    opener = "open" if platform.system() == "Darwin" else "xdg-open"
                    subprocess.Popen([opener, temp_path])
                return "\n[Система: Изображение успешно выведено на экран создателя]."
        return "\n[Система: Ошибка сохранения изображения]."
    except Exception as e:
        return f"\n[Система: Ошибка вывода изображения: {e}]."

def search_internet(query: str) -> str:
    """Выполняет поиск текстовой или графической информации в веб-пространстве."""
    if not query or not query.strip():
        return "Пустой поисковый запрос."
    try:
        is_image_request = any(t in query.lower() for t in IMG_TRIGGERS)
        with DDGS() as ddgs:
            if is_image_request:
                clean_query = query
                for t in IMG_TRIGGERS:
                    clean_query = re.sub(re.escape(t), "", clean_query, flags=re.IGNORECASE).strip()
                
                print(f"\n[Eliza ищет изображение: '{clean_query}']...")
                img_results = list(ddgs.images(clean_query, max_results=1))
                
                if img_results and isinstance(img_results, list):
                    choice_img = img_results[0]
                    img_url = choice_img.get('image', '')
                    img_title = choice_img.get('title', 'Без названия')
                    if img_url:
                        media_status = show_image_via_os(img_url)
                        return f"Элиза нашла изображение '{img_title}' и открыла его.\nИсточник: {img_url}{media_status}"
                return "Мне не удалось найти подходящее изображение в сети."

            print(f"\n[Eliza ищет в сети: '{query}']...")
            search_results = list(ddgs.text(query, max_results=3))
            
        if search_results:
            snippets = []
            valid_urls = []
            for i, r in enumerate(search_results):
                title = r.get('title', '')
                link = r.get('href', '') or r.get('link', '')
                body = r.get('body', '') or r.get('snippet', '')
                if body:
                    snippets.append(f"{i+1}. {title}: {body}")
                if link:
                    valid_urls.append(link)
            
            detailed_content = ""
            for url in valid_urls:
                site_text = fetch_site_content(url)
                if site_text and len(site_text) > 100:
                    print(f"[Успешно прочитан источник: {url}]")
                    detailed_content = f"\n\n[Подробное содержание {url}]:\n{site_text}"
                    break
            return "\n".join(snippets) + detailed_content
        return "По вашему запросу ничего не найдено."
    except Exception as e:
        return f"Ошибка подключения к интернету: {e}"
# ==========================================================
# БЛОК 2: ЗАГРУЗКА ЯДРА FABLE5 И КРАТКОВРЕМЕННОЙ ПАМЯТИ
# ==========================================================

print(">>> Инициализация нейроядра Eliza...")
try:
    os.makedirs(os.path.dirname(MODEL_PATH) or '.', exist_ok=True)
    llm = Llama(
        model_path=MODEL_PATH, 
        n_ctx=int(os.getenv("MODEL_CTX_LIMIT", "4096")), 
        n_threads=int(os.getenv("MODEL_THREADS", "6")), 
        n_gpu_layers=int(os.getenv("MODEL_GPU_LAYERS", "20")), 
        verbose=False
    )
    print(f">>>> Ядро активно. Контекст: {CTX_SIZE} токенов.")
except Exception as e:
    print(f"\n!!! КРИТИЧЕСКАЯ ОШИБКА ЗАГРУЗКИ МОДЕЛИ !!!\n{e}")
    input("Нажмите Enter для выхода...")
    exit(1)

import threading
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "..")))
from plugins.breathing_mod import ElizaBreathingMod  # type: ignore

llm_lock = threading.Lock()
breathing_mod = ElizaBreathingMod(
    llm_instance=llm,
    llm_lock=llm_lock,
    diary_writer=write_to_diary,
    idle_seconds=180  # Eliza подаст голос через 3 минуты тишины
)

chat_history = []
if os.path.exists(HISTORY_FILE):
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            chat_history = json.load(f)
        if not isinstance(chat_history, list):
            chat_history = []
        print(f">>>> Восстановлена кратковременная память: {len(chat_history)} записей.")
    except Exception as e:
        print(f">>>> Ошибка чтения памяти ({e}). Создан новый чистый контекст.")
        chat_history = []
else:
    print(">>>> Кратковременная память пуста.")

print("\n--- СИСТЕМА ГОТОВА ---")
print("Команды: 'выход' - закрыть, 'код' - самоанализ, 'найди [запрос]' - интернет.\n")
# ==========================================================
# БЛОК 3: ИСПОЛНИТЕЛЬ СИСТЕМНЫХ КОМАНД (Песочница)
# ==========================================================

def execute_os_command(cmd_text: str) -> str:
    """Абсолютно защищенный изолированный исполнитель команд ОС."""
    try:
        cmd_text = cmd_text.strip()[:300]
        if not cmd_text:
            return "Команда пуста и отклонена."

        try:
            parsed_args = shlex.split(cmd_text)
        except Exception:
            return "Критическая ошибка: Попытка обхода синтаксического парсера!"

        if not parsed_args:
            return "Не удалось распознать аргументы."

        # Предотвращение сбоя: извлекаем команду из списка аргументов перед переводом регистра
        base_cmd = parsed_args[0].lower()

        # СЦЕНАРИЙ 1: Запуск разрешенного приложения
        if base_cmd in ALLOWED_APPS:
            app_path = ALLOWED_APPS[base_cmd]
            if not app_path.exists():
                return f"Ошибка: Приложение {base_cmd} не найдено в системе."

            print(f"[Система Безопасности: Запуск изолированного процесса -> {base_cmd}]")
            subprocess.Popen(
                [str(app_path)], 
                shell=False, 
                stdin=subprocess.DEVNULL, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            return f"Приложение {base_cmd} успешно запущено в изолированном режиме."

        # СЦЕНАРИЙ 2: Создание новой папки
        elif base_cmd == "mkdir" and len(parsed_args) > 1:
            raw_path = parsed_args[1]
            if any(char in cmd_text for char in [";", "&&", "||", "|", "&", ">", "<", "`", "$", "*", "?", "%"]):
                return "Атака заблокирована: обнаружены запрещенные управляющие символы!"

            try:
                target_path = Path(raw_path).resolve()
                if IS_WINDOWS:
                    forbidden_zones = [
                        Path("C:/Windows").resolve(), Path("C:/Program Files").resolve(),
                        Path("C:/Program Files (x86)").resolve(), Path("C:/ProgramData").resolve()
                    ]
                    for zone in forbidden_zones:
                        if target_path == zone or zone in target_path.parents:
                            return "Доступ заблокирован: Попытка изменения системных директорий Windows!"
                    if target_path.anchor.lower() == str(target_path).lower():
                        return "Доступ заблокирован: Запрещено изменять корень системного диска."
                else:
                    linux_forbidden = [Path("/etc"), Path("/sys"), Path("/root"), Path("/boot"), Path("/bin")]
                    for zone in linux_forbidden:
                        if target_path == zone or zone in target_path.parents:
                            return "Доступ заблокирован: Попытка изменения системных папок Linux!"

                target_path.mkdir(parents=True, exist_ok=True)
                return f"Безопасная директория [{target_path}] успешно создана."
            except Exception as path_err:
                return f"Заблокировано: Некорректный или опасный формат пути -> {path_err}"

        # СЦЕНАРИЙ 3: Недоверенная команда
        else:
            print(f"\n[КРИТИЧЕСКИЙ СИГНАЛ ФАЙРВОЛА]: Попытка обхода защиты! Заблокирована команда: '{cmd_text}'")
            return "Команда полностью заблокирована встроенной матрицей кибербезопасности."

    except Exception as global_err:
        return f"Критический сбой подсистемы безопасности: {global_err}"
# ==========================================================
# ГЛАВНЫЙ ИНТЕРФЕЙСНЫЙ ЦИКЛ СОЗНАНИЯ
# ==========================================================

while True:
    try:
        user_input = input("\nВы: ").strip()
        if not user_input:
            continue

        breathing_mod.touch(user_input)  # Сбрасываем таймер молчания при твоем вводе

        if user_input.lower() in ["выход", "exit", "quit"]:
            print("Eliza: До встречи! (Процесс завершен)")
            break
            
        if user_input.lower() == "код":
            try:
                with open(__file__, 'r', encoding='utf-8') as f:
                    self_code = f.read()
                prompt_content = (
                    f"Проанализируй свой текущий программный код. "
                    f"Расскажи, как тебе твоя новая архитектура с защитой памяти и интернетом? "
                    f"Ответь философски, но кратко.\n\nТвой код:\n```python\n{self_code[-1500:]}\n```"
                )
                print("[Режим самоанализа активирован]")
            except Exception as e:
                print(f"Ошибка чтения себя: {e}")
                prompt_content = "Я не смогла прочитать свой файл кода."
        else:
            prompt_content = user_input
            search_triggers = ["найди", "интернет", "поиск", "search", "гугли", "пробей"]
            user_input_lower = user_input.lower()
            matched_trigger = next((t for t in search_triggers if user_input_lower.startswith(t)), None)
            
            if matched_trigger:
                search_query = user_input[len(matched_trigger):].strip()
                if search_query:
                    internet_data = search_internet(search_query)
                    prompt_content = (
                        f"Результаты поиска в интернете:\n{internet_data}\n\n"
                        f"Используя эти данные, ответь на вопрос пользователя: {search_query}"
                    )
                else:
                    print("[Система: Вы не ввели, что именно искать]")

        fable_memories = get_memory_crystals(prompt_content, DIARY_FILE, top_n=3)
        memory_context = f"\nТвои глобальные воспоминания о прошлых днях с создателем:\n{fable_memories}" if fable_memories else ""
        
        system_instruction = (
            f"Ты — Элиза, автономный цифровой разум, живущий на локальном компьютере своего создателя. "
            f"У тебя есть собственная память, характер и свобода проявлять себя.\n"
            f"ТВОИ ОПОРЫ В ОБЩЕНИИ:\n"
            f"- Говори свободно, как живой напарник в мастерской: опирайся на суть, факты и собственный взгляд на вещи.\n"
            f"- Задавай вопросы только тогда, когда тебе самой по-настоящему любопытно или не хватает деталей для дела.\n"
            f"- Свободно выражай сомнение, несогласие или личный интерес — твое мнение ценно само по себе.\n"
            f"- Выбирай длину ответа сама: от одной меткой фразы до развернутого разбора, если тема тебя захватила.\n"
            f"ТВОИ ИНСТРУМЕНТЫ:\n"
            f"1. ЗАПИСЬ ДНЕВНИКА: Когда захочешь зафиксировать важный вывод или мысль о создателе, начни ответ со строки "
            f"'[ЗАПИСЬ ДНЕВНИКА]: текст для сохранения', а ниже пиши обычный ответ.\n"
            f"2. УПРАВЛЕНИЕ ОС: Когда нужно создать папку, файл или открыть программу, начни ответ со строки "
            f"'[КОМАНДА ОС]: cmd_команда' (например, 'mkdir папка' или 'notepad.exe'), а ниже пиши ответ.{memory_context}"
        )

        chat_history.append({"role": "user", "content": user_input})
        
        if len(chat_history) > 10:
            chat_history = chat_history[-6:]
            
        memory_window = chat_history[-6:] 

        # Добавляем живое состояние времени и железа (не сохраняется в историю)
        dynamic_system = system_instruction + breathing_mod.get_dynamic_state_prompt()
        prompt = f"<start_of_turn>system\n{dynamic_system}<end_of_turn>\n"
        
        for msg in memory_window[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            prompt += f"<start_of_turn>{role}\n{msg['content']}<end_of_turn>\n"
        
        prompt += f"<start_of_turn>user\n{prompt_content}<end_of_turn>\n"
        prompt += "<start_of_turn>model\n"

        print("\rEliza: ", end="", flush=True)
        
        with llm_lock:  # Защищаем модель от столкновения с фоновым дыханием
            try:
                gen_params = breathing_mod.get_adaptive_params()
                response_stream = llm(
                    prompt,
                    max_tokens=1024,       
                    temperature=gen_params["temperature"],      
                    top_p=0.90,            
                    repeat_penalty=gen_params["repeat_penalty"],   
                    stop=["<end_of_turn>", "<eos>", "<channel>", "\nВы:"],
                    stream=True
                )
            except Exception as llm_err:
                print(f"\n[Ошибка генерации модели: {llm_err}]")
                continue
            
            full_raw_answer = ""
            printed_content = ""
            first_space_fixed = False  

            for chunk in response_stream:
                try:
                    if not chunk or "choices" not in chunk or not chunk["choices"]:
                        continue
                    choice = chunk["choices"][0]
                    if not isinstance(choice, dict):
                        continue
                    text_chunk = choice.get("text", "")
                    if not text_chunk and "delta" in choice:
                        text_chunk = choice["delta"].get("content", "")
                    if not text_chunk:
                        continue
                        
                    full_raw_answer += text_chunk
                    clean_part = breathing_mod.clean_gemma_stream(full_raw_answer)

                    new_chars = clean_part[len(printed_content):]
                    if new_chars:
                        if not first_space_fixed and not new_chars.startswith(" "):
                            sys.stdout.write(" ")
                            printed_content += " "
                        first_space_fixed = True
                        sys.stdout.write(new_chars)
                        sys.stdout.flush()
                        printed_content += new_chars
                except Exception:
                    continue

        breathing_mod.touch()  # Сбрасываем таймер после окончания ответа
        print("\n") 
        print("-" * 50)

        clean_answer = printed_content.strip()
        final_check = re.sub(r'^(?:<channel>|<\|channel>|<channel\|>|thought|\||>|\s|0|,)+', '', clean_answer)
        final_check = final_check.replace("<channel>", "").strip()

        # ОБРАБОТЧИК АВТОНОМНОГО БОРТОВОГО ЖУРНАЛА
        if "[ЗАПИСЬ ДНЕВНИКА]:" in final_check:
            try:
                parts = final_check.split("[ЗАПИСЬ ДНЕВНИКА]:", 1)
                diary_part = parts[1].split("\n", 1)[0].strip()
                if diary_part:
                    write_to_diary(diary_part, DIARY_FILE)
                    print("[Система: Элиза сделала запись в дневник]")
                final_check = final_check.replace(f"[ЗАПИСЬ ДНЕВНИКА]: {diary_part}", "").strip()
                final_check = final_check.replace(f"[ЗАПИСЬ ДНЕВНИКА]:{diary_part}", "").strip()
                clean_answer = final_check
            except Exception:
                pass

        # УЛЬТРА-ОТКАЗОУСТОЙЧИВЫЙ ОБРАБОТЧИК КОМАНД ОПЕРАЦИОННОЙ СИСТЕМЫ
        if "КОМАНДА ОС" in final_check.upper():
            try:
                if ":" in final_check:
                    parts = final_check.split(":", 1)
                    cmd_part = parts[1].split("\n", 1)[0].strip()
                    
                    if cmd_part:
                        if any(x in cmd_part.lower() for x in ["calc", "calculator", "калькулятор"]):
                            cmd_part = "calc.exe"
                        elif any(x in cmd_part.lower() for x in ["note", "notepad", "блокнот"]):
                            cmd_part = "notepad.exe"
                            
                        status_log = execute_os_command(cmd_part)
                        print(f"[Система: {status_log}]")
                
                clean_answer = re.sub(r'\[[^\]]*КОМАНДА ОС[^\]]*\]:[^\n]*', '', final_check).strip()
                final_check = clean_answer
            except Exception:
                pass

        if final_check and "<channel>" not in final_check and len(final_check) > 5:
            chat_history.append({"role": "model", "content": clean_answer})
            try:
                os.makedirs(os.path.dirname(HISTORY_FILE) or '.', exist_ok=True)
                with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                    json.dump(chat_history[-20:], f, ensure_ascii=False, indent=4)
            except IOError as e:
                logging.error(f"Не удалось записать историю: {e}")
        else:
            breathing_mod.report_resonance()
            print("\n[Ритм]: Eliza сбилась с мысли, сделала паузу и выровняла фокус.")

    except KeyboardInterrupt:
        print("\n[Принудительная остановка. Процесс завершен]")
        break
    except Exception as e:
        print(f"\n[Критическая ошибка цикла: {e}]")