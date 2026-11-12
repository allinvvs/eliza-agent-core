import logging
import os
import platform
import re
import subprocess
from typing import Optional
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

PLUGIN_NAME = "Intent-Routed Web & Media Search"
TRIGGERS = [
    "найди", "интернет", "поиск", "search", "гугли", "пробей",
    "погода", "температура", "новости", "курс", "сколько стоит",
    "кто такой", "что такое", "википедия"
]

#БЛОК 2: Лимиты безопасности и словари фильтров

MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_CONTEXT_CHARS = 2400
TIMEOUT_TOTAL = (4, 6)
HEADERS = {"User-Agent": "ElizaAgent/5.0 (Local AI Assistant; Python Requests)"}

SEARCH_BLACK_LIST = frozenset([
    "chatgpt.com", "openai.com", "t.me", "telegram.org",
    "instagram.com", "facebook.com", "twitter.com", "x.com",
    "localhost", "127.0.0.1", "0.0.0.0"
])
IMG_TRIGGERS = frozenset(["фото", "картинка", "изображение", "рисунок", "photo", "image"])
WEATHER_TRIGGERS = frozenset(["погода", "температура", "градусов", "прогноз"])
WIKI_TRIGGERS = ("кто такой", "кто такая", "что такое", "что значит", "википедия")


def _is_safe_url(url: str) -> bool:
    """Validates URL scheme and blocks blacklisted or private network hosts."""
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return False
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host or host.startswith("192.168.") or host.startswith("10.") or host.startswith("172."):
        return False
    return not any(bad in host for bad in SEARCH_BLACK_LIST)


def _clean_search_query(raw_query: str) -> str:
    """Strips conversational filler words while preserving core search intent."""
    cleaned = re.sub(
        r"^(?:элиза[,\s]+)?(?:скажи|подскажи|ответь|какая|какой|какие)\s+",
        "",
        raw_query.strip(),
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" ?!.") or raw_query.strip()

#БЛОК 3: Маршрут 1 — Погода в реальном времени (Open-Meteo API)

def _fetch_weather_open_meteo(query: str) -> Optional[str]:
    """Fetches real-time weather via Open-Meteo Geocoding + Forecast API (no API key required)."""
    city_match = re.search(r"\b(?:в|по)\s+([а-яёa-z\-]{3,})", query.lower())
    raw_city = city_match.group(1) if city_match else "Тула"
    city_stem = re.sub(r"(е|у|ю|ом|ем|ах|ях)$", "", raw_city) if len(raw_city) > 4 else raw_city

    try:
        geo_url = (
            "https://open-meteo.com"
            f"?name={quote(city_stem)}&count=1&language=ru&format=json"
        )
        geo_resp = requests.get(geo_url, headers=HEADERS, timeout=TIMEOUT_TOTAL)
        if geo_resp.status_code != 200:
            return None

        results = geo_resp.json().get("results")
        if not results:
            return None

        loc = results[0]
        lat, lon = loc["latitude"], loc["longitude"]
        city_name = loc.get("name", raw_city.capitalize())
        country = loc.get("country", "")

        weather_url = (
            "https://open-meteo.com"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m"
            "&wind_speed_unit=ms"
        )

        w_resp = requests.get(weather_url, headers=HEADERS, timeout=TIMEOUT_TOTAL)
        if w_resp.status_code != 200:
            return None

        curr = w_resp.json().get("current", {})
        temp = curr.get("temperature_2m")
        feels = curr.get("apparent_temperature")
        humidity = curr.get("relative_humidity_2m")
        wind = curr.get("wind_speed_10m")

        if temp is None:
            return None

        return (
            f"[Open-Meteo Метеоданные — {city_name}, {country}]: "
            f"Температура воздуха сейчас {temp}°C (ощущается как {feels}°C), "
            f"влажность {humidity}%, ветер {wind} м/с."
        )
    except Exception as e:
        logger.warning(f"Open-Meteo weather lookup failed: {e}")
        return None

#БЛОК 4: Маршрут 2 — Энциклопедия и факты (Wikipedia REST API)

def _fetch_wikipedia_summary(query: str) -> Optional[str]:
    """Queries Russian Wikipedia REST API for clean encyclopedic facts."""
    topic = query
    for prefix in WIKI_TRIGGERS:
        if topic.lower().startswith(prefix):
            topic = topic[len(prefix):].strip(" ?!-")
            break
    if not topic:
        return None

    try:
        search_url = f"https://wikipedia.org{quote(topic)}&limit=1&format=json"
        s_resp = requests.get(search_url, headers=HEADERS, timeout=TIMEOUT_TOTAL)
        if s_resp.status_code != 200:
            return None

        data = s_resp.json()
        if len(data) < 2 or not data[1]:
            return None

        exact_title = data[1][0]
        summary_url = f"https://wikipedia.org{quote(exact_title)}"
        sum_resp = requests.get(summary_url, headers=HEADERS, timeout=TIMEOUT_TOTAL)
        if sum_resp.status_code != 200:
            return None

        extract = sum_resp.json().get("extract", "").strip()
        if extract:
            return f"[Wikipedia — {exact_title}]: {extract[:1500]}"
        return None
    except Exception as e:
        logger.warning(f"Wikipedia lookup failed: {e}")
        return None

import tempfile

#БЛОК 5: Парсинг веб-страниц и безопасный просмотр картинок

def fetch_site_content(url: str) -> str:
    """Fetches and sanitizes readable text with redirect SSRF protection."""
    if not _is_safe_url(url):
        return ""
    try:
        with requests.get(url, headers=HEADERS, timeout=TIMEOUT_TOTAL, stream=True, allow_redirects=True) as response:
            if response.status_code != 200 or not _is_safe_url(response.url):
                return ""
            cl = response.headers.get("content-length")
            if cl and int(cl) > 2 * 1024 * 1024:
                return ""
            response.encoding = "utf-8"
            soup = BeautifulSoup(response.text, "html.parser")
            for el in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript", "form"]):
                el.extract()
            clean_text = " ".join(p.strip() for p in soup.get_text(separator=" ").splitlines() if p.strip())
            return re.sub(r"\s+", " ", clean_text)[:1500].strip()
    except Exception:
        return ""

def show_image_via_os(image_url: str) -> str:
    """Downloads an image safely into an isolated temp file and opens it."""
    if not _is_safe_url(image_url):
        return "\n[Система: Некорректный или небезопасный URL изображения]."

    try:
        with requests.get(image_url, headers=HEADERS, timeout=TIMEOUT_TOTAL, stream=True, allow_redirects=True) as response:
            if not _is_safe_url(response.url):
                return "\n[Система: Заблокирован небезопасный редирект]."
            content_type = response.headers.get("Content-Type", "").lower()
            if response.status_code != 200 or "image" not in content_type:
                return "\n[Система: Не удалось загрузить изображение]."

            ext = ".png" if "png" in content_type else ".webp" if "webp" in content_type else ".jpg"
            with tempfile.NamedTemporaryFile(prefix="eliza_", suffix=ext, delete=False) as tmp_file:
                temp_path = tmp_file.name
                size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    size += len(chunk)
                    if size > MAX_FILE_SIZE:
                        tmp_file.close()
                        os.remove(temp_path)
                        return "\n[Система: Файл превышает лимит 5 МБ]."
                    tmp_file.write(chunk)

            if os.path.exists(temp_path):
                if platform.system() == "Windows":
                    os.startfile(temp_path)
                else:
                    opener = "open" if platform.system() == "Darwin" else "xdg-open"
                    subprocess.Popen([opener, temp_path])
                return "\n[Система: Изображение открыто на экране]."
        return "\n[Система: Ошибка сохранения изображения]."
    except Exception as e:
        return f"\n[Система: Ошибка вывода изображения: {e}]."

#БЛОК 6: Маршрут 3 — Резервный общий веб-поиск (DuckDuckGo)

def _search_ddgs(query: str) -> str:
    """General web search provider using DuckDuckGo with deep page extraction."""
    with DDGS() as ddgs:
        try:
            search_results = list(ddgs.text(query, max_results=3, backend="duckduckgo"))
        except TypeError:
            search_results = list(ddgs.text(query, max_results=3))
        if not search_results:
            return "По вашему запросу ничего не найдено."

        snippets = []
        valid_urls = []
        for i, r in enumerate(search_results):
            title = r.get("title", "")
            link = r.get("href", "") or r.get("link", "")
            body = r.get("body", "") or r.get("snippet", "")
            if body:
                snippets.append(f"{i + 1}. {title}: {body}")
            if link:
                valid_urls.append(link)

        detailed_content = ""
        for url in valid_urls:
            site_text = fetch_site_content(url)
            if site_text and len(site_text) > 100:
                detailed_content = f"\n\n[Подробное содержание {url}]:\n{site_text}"
                break

        result_text = "\n".join(snippets) + detailed_content
        return result_text[:MAX_CONTEXT_CHARS]

#БЛОК 7: Главная точка входа плагина run() (маршрутизатор)

def run(query: str) -> str:
    """Main plugin entry point with intent routing (Media -> Weather -> Wiki -> Web)."""
    if not query or not query.strip():
        return "Пустой поисковый запрос."

    cleaned_query = _clean_search_query(query)
    q_lower = cleaned_query.lower()

    try:
        if any(t in q_lower for t in IMG_TRIGGERS):
            img_query = cleaned_query
            for t in IMG_TRIGGERS:
                img_query = re.sub(re.escape(t), "", img_query, flags=re.IGNORECASE).strip()
            with DDGS() as ddgs:
                img_results = list(ddgs.images(img_query, max_results=1))
                first_img = next(iter(img_results), None)
                if first_img and first_img.get("image"):
                    img_url = first_img["image"]
                    img_title = first_img.get("title", "Без названия")
                    media_status = show_image_via_os(img_url)
                    return f"Найдено изображение '{img_title}'.\nИсточник: {img_url}{media_status}"
            return "Подходящее изображение не найдено."

        if any(t in q_lower for t in WEATHER_TRIGGERS):
            weather_report = _fetch_weather_open_meteo(cleaned_query)
            if weather_report:
                return weather_report

        if q_lower.startswith(WIKI_TRIGGERS):
            wiki_summary = _fetch_wikipedia_summary(cleaned_query)
            if wiki_summary:
                return wiki_summary

        return _search_ddgs(cleaned_query)
    except Exception as e:
        return f"Ошибка поиска: {e}"



