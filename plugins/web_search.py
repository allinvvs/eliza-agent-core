from __future__ import annotations

import ipaddress
import json
import logging
import os
import platform
import re
import socket
import subprocess
import tempfile
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import importlib
from typing import Any


def _optional_import(module_path: str, attr: str | None = None) -> Any:
    """Safely imports optional third-party backends without crashing headless CI."""
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, attr) if attr else mod
    except Exception:
        return None


requests = _optional_import("requests")
BeautifulSoup = _optional_import("bs4", "BeautifulSoup")
DDGS = _optional_import("ddgs", "DDGS") or _optional_import("duckduckgo_search", "DDGS")

logger = logging.getLogger(__name__)

PLUGIN_NAME = "Intent-Routed Web & Media Search"
TRIGGERS = [
    "search", "find", "google", "lookup", "web",
    "weather", "temperature", "forecast", "news", "price",
    "who is", "what is", "wiki", "wikipedia",
    "найди", "интернет", "поиск", "гугли", "пробей",
    "погода", "температура", "новости", "курс", "сколько стоит",
    "кто такой", "что такое", "википедия",
]

MAX_IMAGE_BYTES = 5 * 1024 * 1024   # 5 MB limit for images
MAX_HTML_BYTES = 512 * 1024         # 512 KB stream cap to protect 16 GB RAM
MAX_CONTEXT_CHARS = 2200            # Strict budget for n_ctx=3072
MAX_REDIRECTS = 3
TIMEOUT_TOTAL = (3.5, 6.0)

HEADERS = {
    "User-Agent": "ELIZA-AgentCore/5.0 (Local-AI-Assistant; +https://github.com)",
    "Accept": "application/json, text/html;q=0.9, image/*;q=0.8",
}

DOMAIN_BLACKLIST = frozenset({
    "chatgpt.com", "openai.com", "t.me", "telegram.org",
    "instagram.com", "facebook.com", "twitter.com", "x.com",
    "localhost", "ip6-localhost", "ip6-loopback",
})

IMG_TRIGGERS = frozenset({"photo", "image", "picture", "draw", "фото", "картинка", "изображение", "рисунок"})
WEATHER_TRIGGERS = frozenset({"weather", "temperature", "forecast", "погода", "температура", "градусов", "прогноз"})
WIKI_TRIGGERS = (
    "who is", "what is", "wiki ", "wikipedia ",
    "кто такой", "кто такая", "что такое", "что значит", "википедия",
)


def _is_safe_url(url: str) -> bool:
    """Validates URL scheme, blacklists domains, and resolves DNS to block private/loopback IPs (SSRF)."""
    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url.strip())
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False

        host = parsed.hostname.lower().rstrip(".")
        if host in DOMAIN_BLACKLIST or any(host.endswith(f".{bad}") for bad in DOMAIN_BLACKLIST):
            return False

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addr_info = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        if not addr_info:
            return False

        for _, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip_obj = ipaddress.ip_address(ip_str)
            if (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_multicast
                or ip_obj.is_reserved
                or ip_obj.is_unspecified
            ):
                logger.warning("SSRF Guard blocked unsafe target %s -> %s", host, ip_str)
                return False

        return True
    except (ValueError, socket.gaierror, OSError):
        return False

#Блок 2. Безопасный потоковый HTTP-клиент (с проверкой каждого редиректа до подключения)

def _safe_stream_get(
    url: str,
    params: dict[str, Any] | None = None,
    max_bytes: int = MAX_HTML_BYTES,
) -> tuple[bytes, str, str]:
    """Performs HTTP GET validating every redirect hop before connecting and capping download size."""
    if requests is None:
        raise RuntimeError("HTTP client ('requests') is not installed.")

    current_url = url
    current_params = params

    with requests.Session() as session:
        session.headers.update(HEADERS)

        for _ in range(MAX_REDIRECTS + 1):
            if not _is_safe_url(current_url):
                raise PermissionError(f"Blocked unsafe URL: {current_url}")

            with session.get(
                current_url,
                params=current_params,
                timeout=TIMEOUT_TOTAL,
                stream=True,
                allow_redirects=False,
            ) as resp:
                if resp.is_redirect or resp.is_permanent_redirect:
                    location = resp.headers.get("Location")
                    if not location:
                        raise ValueError("Redirect missing Location header")
                    current_url = urljoin(current_url, location)
                    current_params = None
                    continue

                resp.raise_for_status()

                chunks: list[bytes] = []
                downloaded = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise ValueError(f"Response exceeded maximum limit of {max_bytes} bytes")
                    chunks.append(chunk)

                content_type = resp.headers.get("Content-Type", "").lower()
                return b"".join(chunks), content_type, current_url

    raise RuntimeError("Too many HTTP redirects")


def _clean_search_query(raw_query: str) -> str:
    """Strips conversational filler words while preserving core search intent."""
    cleaned = re.sub(
        r"^(?:eliza[,\s]+|элиза[,\s]+)?(?:tell me|show me|find|search|скажи|подскажи|ответь|найди|какая|какой|какие)\s+",
        "",
        raw_query.strip(),
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" ?!.") or raw_query.strip()

#Блок 3. Рабочие маршруты: Open-Meteo API, Wikipedia REST API и безопасный парсер страниц

def _fetch_weather_open_meteo(query: str) -> str | None:
    """Fetches real-time weather via official Open-Meteo Geocoding + Forecast APIs."""
    city_match = re.search(r"\b(?:in|for|в|по)\s+([a-zа-яё\-]{3,})", query.lower())
    raw_city = city_match.group(1) if city_match else "Tula"
    # Базовое отсечение русских падежных окончаний для геокодера
    city_stem = re.sub(r"(е|у|ю|ом|ем|ах|ях)$", "", raw_city) if len(raw_city) > 4 else raw_city

    try:
        geo_bytes, _, _ = _safe_stream_get(
            "https://open-meteo.com",
            params={"name": city_stem, "count": 1, "language": "en", "format": "json"},
            max_bytes=64 * 1024,
        )
        geo_data = json.loads(geo_bytes.decode("utf-8", errors="replace"))
        results = geo_data.get("results")
        if not results:
            return None

        loc = results[0]
        lat, lon = loc["latitude"], loc["longitude"]
        city_name = loc.get("name", raw_city.capitalize())
        country = loc.get("country", "")

        w_bytes, _, _ = _safe_stream_get(
            "https://open-meteo.com",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m",
                "wind_speed_unit": "ms",
                "timezone": "auto",
            },
            max_bytes=64 * 1024,
        )
        w_data = json.loads(w_bytes.decode("utf-8", errors="replace"))
        curr = w_data.get("current", {})
        temp = curr.get("temperature_2m")
        if temp is None:
            return None

        return (
            f"[Open-Meteo Weather — {city_name}, {country}]: "
            f"Temperature {temp}°C (feels like {curr.get('apparent_temperature')}°C), "
            f"humidity {curr.get('relative_humidity_2m')}%, wind {curr.get('wind_speed_10m')} m/s."
        )
    except Exception as exc:
        logger.warning("Open-Meteo lookup failed: %s", exc)
        return None


def _fetch_wikipedia_summary(query: str) -> str | None:
    """Queries official Wikipedia OpenSearch + REST Summary APIs (auto-detects RU/EN)."""
    topic = query
    for prefix in WIKI_TRIGGERS:
        if topic.lower().startswith(prefix):
            topic = topic[len(prefix):].strip(" ?!-")
            break
    if not topic:
        return None

    lang = "ru" if re.search(r"[а-яё]", topic.lower()) else "en"
    base_api = f"https://{lang}.wikipedia.org"

    try:
        search_bytes, _, _ = _safe_stream_get(
            f"{base_api}/w/api.php",
            params={"action": "opensearch", "search": topic, "limit": 1, "format": "json"},
            max_bytes=64 * 1024,
        )
        data = json.loads(search_bytes.decode("utf-8", errors="replace"))
        if len(data) < 2 or not data[1]:
            return None

        exact_title = data[1][0]
        encoded_title = quote(exact_title.replace(" ", "_"))
        sum_bytes, _, _ = _safe_stream_get(
            f"{base_api}/api/rest_v1/page/summary/{encoded_title}",
            max_bytes=128 * 1024,
        )
        sum_data = json.loads(sum_bytes.decode("utf-8", errors="replace"))
        extract = sum_data.get("extract", "").strip()
        if extract and sum_data.get("type") != "disambiguation":
            return f"[Wikipedia — {exact_title}]: {extract[:1500]}"
        return None
    except Exception as exc:
        logger.warning("Wikipedia lookup failed: %s", exc)
        return None


def fetch_site_content(url: str) -> str:
    """Fetches and sanitizes readable text with strict byte-limit and SSRF protection."""
    try:
        raw_bytes, content_type, _ = _safe_stream_get(url, max_bytes=MAX_HTML_BYTES)
        if "text/html" not in content_type and "text/plain" not in content_type:
            return ""

        html_text = raw_bytes.decode("utf-8", errors="replace")
        if BeautifulSoup is None:
            return re.sub(r"<[^>]+>", " ", html_text)[:1400].strip()

        soup = BeautifulSoup(html_text, "html.parser")
        for el in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript", "form", "aside"]):
            el.extract()

        clean_text = " ".join(line.strip() for line in soup.get_text(separator=" ").splitlines() if line.strip())
        return re.sub(r"\s+", " ", clean_text)[:1400].strip()
    except Exception:
        return ""

#Блок 4. Безопасный просмотр изображений (с проверкой Magic Bytes), DuckDuckGo и роутер run()

def _detect_image_extension(data: bytes) -> str | None:
    """Verifies actual image file signature (magic bytes) to prevent spoofed Content-Type attacks."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    return None


def show_image_via_os(image_url: str) -> str:
    """Downloads an image safely, verifies magic bytes, and opens it via default OS viewer."""
    try:
        img_bytes, content_type, _ = _safe_stream_get(image_url, max_bytes=MAX_IMAGE_BYTES)
        if "image" not in content_type:
            return "\n[System: Target URL did not return an image content type]."

        ext = _detect_image_extension(img_bytes)
        if not ext:
            return "\n[System: Blocked image with invalid or unsupported binary header]."

        with tempfile.NamedTemporaryFile(prefix="eliza_", suffix=ext, delete=False) as tmp_file:
            tmp_file.write(img_bytes)
            temp_path = tmp_file.name

        if platform.system() == "Windows":
            os.startfile(temp_path)  # type: ignore[attr-defined]
        else:
            opener = "open" if platform.system() == "Darwin" else "xdg-open"
            subprocess.Popen([opener, temp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return "\n[System: Image verified and opened on screen]."
    except PermissionError as sec_err:
        return f"\n[System: {sec_err}]."
    except ValueError as size_err:
        return f"\n[System: {size_err}]."
    except Exception as exc:
        logger.warning("Image display failed: %s", exc)
        return f"\n[System: Failed to display image ({type(exc).__name__})]."


def _search_ddgs(query: str) -> str:
    """General web search provider using DuckDuckGo with safe single-page content extraction."""
    if DDGS is None:
        return "[Web Search]: Search backend ('ddgs' / 'duckduckgo-search') is not installed."

    with DDGS() as ddgs:
        try:
            search_results = list(ddgs.text(query, max_results=3, backend="duckduckgo"))
        except TypeError:
            search_results = list(ddgs.text(query, max_results=3))

        if not search_results:
            return f"No web results found for '{query}'."

        snippets: list[str] = []
        valid_urls: list[str] = []
        for idx, item in enumerate(search_results, start=1):
            title = item.get("title", "").strip()
            link = (item.get("href") or item.get("link") or "").strip()
            body = (item.get("body") or item.get("snippet") or "").strip()
            if body:
                snippets.append(f"{idx}. {title}: {body}")
            if link and _is_safe_url(link):
                valid_urls.append(link)

        detailed_content = ""
        for url in valid_urls[:2]:  # Проверяем максимум 2 ссылки, чтобы не задерживать ответ
            site_text = fetch_site_content(url)
            if len(site_text) > 120:
                detailed_content = f"\n\n[Extracted content from {url}]:\n{site_text}"
                break

        combined = "\n".join(snippets) + detailed_content
        return combined[:MAX_CONTEXT_CHARS]


def run(query: str) -> str:
    """Main plugin entry point with intent routing (Media -> Weather -> Wiki -> Web)."""
    if not query or not query.strip():
        return "Empty search query."

    cleaned_query = _clean_search_query(query)
    q_lower = cleaned_query.lower()

    try:
        if any(t in q_lower for t in IMG_TRIGGERS):
            img_query = cleaned_query
            for t in IMG_TRIGGERS:
                img_query = re.sub(rf"\b{re.escape(t)}\b", "", img_query, flags=re.IGNORECASE).strip()

            with DDGS() as ddgs:
                first_img = next(iter(ddgs.images(img_query or cleaned_query, max_results=1)), None)
                if first_img and first_img.get("image"):
                    img_url = first_img["image"]
                    img_title = first_img.get("title", "Untitled")
                    media_status = show_image_via_os(img_url)
                    return f"Found image '{img_title}'.\nSource: {img_url}{media_status}"
            return "No suitable image found."

        if any(t in q_lower for t in WEATHER_TRIGGERS):
            weather_report = _fetch_weather_open_meteo(cleaned_query)
            if weather_report:
                return weather_report

        if q_lower.startswith(WIKI_TRIGGERS):
            wiki_summary = _fetch_wikipedia_summary(cleaned_query)
            if wiki_summary:
                return wiki_summary

        return _search_ddgs(cleaned_query)
    except Exception as exc:
        logger.error("Web search plugin error for '%s': %s", cleaned_query, exc, exc_info=True)
        return f"Search error: {type(exc).__name__}"
