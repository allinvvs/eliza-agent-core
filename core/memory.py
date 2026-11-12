import json
import math
import os
import re
import threading
from datetime import datetime
from typing import Dict, List, Set


class MemoryManager:
    """
    Production-grade Two-Tier Memory Engine:
    - Short-term sliding window with atomic JSON persistence.
    - Long-term ring-buffered diary with RAM-cached IDF/Recency RAG.
    """

    MAX_DIARY_RECORDS = 500
    STOP_WORDS = frozenset({
        "как", "ты", "мне", "что", "тебе", "я", "в", "на", "с", "ли", "бы", "или", "и", "вы",
        "элиза", "это", "то", "не", "но", "а", "да", "нет", "по", "за", "из", "от", "для", "про"
    })

    def __init__(self, history_file: str, diary_file: str, max_window: int = 16):
        self.history_file = history_file
        self.diary_file = diary_file
        self.max_window = max_window
        self._lock = threading.Lock()
        self.short_term: List[Dict[str, str]] = self._load_history()

        self._cache_mtime: float = 0.0
        self._cached_records: List[str] = []
        self._cached_doc_tokens: List[Set[str]] = []
        self._cached_df: Dict[str, int] = {}

#БЛОК 2: Загрузка и атомарное сохранение кратковременной истории

    def _load_history(self) -> List[Dict[str, str]]:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data[-self.max_window:]
            except Exception:
                pass
        return []

    def save_history(self) -> None:
        """Atomically writes short-term memory to disk to prevent file corruption."""
        with self._lock:
            try:
                dir_name = os.path.dirname(self.history_file)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                tmp_file = f"{self.history_file}.tmp"
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(self.short_term, f, ensure_ascii=False, indent=2)
                os.replace(tmp_file, self.history_file)
            except IOError:
                pass

    def add_message(self, role: str, content: str) -> List[Dict[str, str]]:
        """Appends message to sliding window and returns evicted messages on overflow."""
        with self._lock:
            self.short_term.append({
                "role": role,
                "content": content,
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            evicted = []
            if len(self.short_term) > self.max_window:
                overflow = len(self.short_term) - self.max_window
                evicted = self.short_term[:overflow]
                self.short_term = self.short_term[overflow:]
        self.save_history()
        return evicted

#БЛОК 3: Кольцевой буфер долговременного дневника с дедупликацией

    def write_to_diary(self, text: str) -> bool:
        """Appends a deduplicated, timestamped memory crystal with ring-buffer rotation."""
        if not text or not text.strip():
            return False

        safe_text = re.sub(r"\s+", " ", text.strip())
        if len(self._tokenize(safe_text)) < 2:
            return False

        with self._lock:
            try:
                dir_name = os.path.dirname(self.diary_file)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)

                existing_lines: List[str] = []
                if os.path.exists(self.diary_file):
                    with open(self.diary_file, "r", encoding="utf-8") as f:
                        existing_lines = [line.strip() for line in f if line.strip()]

                if any(safe_text.lower() in line.lower() for line in existing_lines[-30:]):
                    return False

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                existing_lines.append(f"[{timestamp}] {safe_text}")

                if len(existing_lines) > self.MAX_DIARY_RECORDS:
                    existing_lines = existing_lines[-self.MAX_DIARY_RECORDS:]

                tmp_diary = f"{self.diary_file}.tmp"
                with open(tmp_diary, "w", encoding="utf-8") as f:
                    f.write("\n".join(existing_lines) + "\n")
                os.replace(tmp_diary, self.diary_file)

                self._cache_mtime = 0.0
                return True
            except IOError:
                return False

#БЛОК 4: Морфологический токенизатор и кэширование индекса в RAM

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        """Extracts normalized word stems for lightweight Russian/English matching."""
        raw_words = re.findall(r"[а-яёa-z0-9]{3,}", text.lower())
        stems = []
        for w in raw_words:
            if w in cls.STOP_WORDS:
                continue
            if len(w) > 5 and re.search(r"[а-яё]", w):
                w = re.sub(
                    r"(иями|ями|ами|ого|ему|ому|ыми|ими|ее|ие|ые|ое|ей|ий|ый|ой|ем|им|ым|ом|ах|ях|у|ю|а|я|е|и|ы|о)$",
                    "",
                    w,
                )
            stems.append(w)
        return stems

    def _refresh_index_if_needed(self) -> None:
        """Rebuilds IDF index in RAM only if the diary file was modified on disk."""
        if not os.path.exists(self.diary_file):
            self._cached_records = []
            self._cached_doc_tokens = []
            self._cached_df = {}
            return

        mtime = os.path.getmtime(self.diary_file)
        if mtime == self._cache_mtime and self._cached_records:
            return

        try:
            with open(self.diary_file, "r", encoding="utf-8") as f:
                records = [line.strip() for line in f if line.strip()]
        except IOError:
            return

        doc_tokens: List[Set[str]] = []
        df: Dict[str, int] = {}
        for rec in records:
            clean_rec = re.sub(r"^\[[^\]]+\]\s*", "", rec)
            tokens = set(self._tokenize(clean_rec))
            doc_tokens.append(tokens)
            for t in tokens:
                df[t] = df.get(t, 0) + 1

        self._cached_records = records
        self._cached_doc_tokens = doc_tokens
        self._cached_df = df
        self._cache_mtime = mtime

#БЛОК 5: Ассоциативный поиск по IDF и свежести (retrieve_crystals)

    def retrieve_crystals(self, query: str, top_n: int = 3) -> str:
        """Retrieves top-N memories from RAM-cached IDF index with recency decay."""
        if not query:
            return ""

        query_stems = set(self._tokenize(query))
        if not query_stems:
            return ""

        with self._lock:
            self._refresh_index_if_needed()
            records = list(self._cached_records)
            doc_tokens = list(self._cached_doc_tokens)
            df = dict(self._cached_df)

        if not records:
            return ""

        total_docs = len(records)
        scored = []
        for idx, (raw_rec, tokens) in enumerate(zip(records, doc_tokens)):
            overlap = query_stems.intersection(tokens)
            if not overlap:
                continue

            idf_score = sum(math.log((total_docs + 1) / (df.get(term, 0) + 0.5)) for term in overlap)
            recency_boost = 1.0 + 0.25 * (idx / max(1, total_docs - 1))
            final_score = idf_score * recency_boost

            if final_score > 0.65:
                clean_out = re.sub(r"[<>]", "", raw_rec)
                scored.append((final_score, clean_out))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return ""

        best = [text for _, text in scored[:top_n]]
        return "\n[Релевантные воспоминания из долгой памяти]:\n" + "\n".join(best)

#БЛОК 6: Телеметрия памяти (get_stats)

    def get_stats(self) -> Dict[str, int]:
        """Returns telemetry metrics for short-term and long-term memory stores."""
        with self._lock:
            self._refresh_index_if_needed()
            diary_count = len(self._cached_records)
            short_count = len(self.short_term)

        return {
            "short_term_messages": short_count,
            "max_window": self.max_window,
            "long_term_crystals": diary_count,
            "max_diary_records": self.MAX_DIARY_RECORDS,
        }

