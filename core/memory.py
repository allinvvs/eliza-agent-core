from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from datetime import datetime
from typing import Any

logger = logging.getLogger("eliza.memory")


class MemoryManager:
    """
    Production-grade Two-Tier Memory Engine:
    - Short-term sliding window with atomic JSON persistence.
    - Long-term ring-buffered crystal store with incremental Okapi BM25 + Recency RAG
      and Jaccard semantic deduplication.
    """

    MAX_DIARY_RECORDS = 500
    DEDUP_JACCARD_THRESHOLD = 0.78

    STOP_WORDS = frozenset({
        # Russian stop words
        "как", "ты", "мне", "что", "тебе", "я", "в", "на", "с", "ли", "бы", "или", "и", "вы",
        "элиза", "это", "то", "не", "но", "а", "да", "нет", "по", "за", "из", "от", "для", "про",
        "при", "уже", "еще", "все", "так", "вот", "тут", "там", "где", "когда", "чем", "кто",
        # English stop words
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one",
        "our", "out", "has", "have", "been", "were", "they", "this", "that", "with", "from", "what",
        "how", "why", "who", "when", "where", "which", "eliza", "into", "about", "just", "like",
    })

    def __init__(self, history_file: str, diary_file: str, max_window: int = 16) -> None:
        self.history_file = history_file
        self.diary_file = diary_file
        self.max_window = max_window
        self._lock = threading.Lock()
        self.short_term: list[dict[str, str]] = self._load_history()

        # Incremental RAM BM25 index state
        self._cache_mtime: float = -1.0
        self._cached_records: list[str] = []
        self._cached_doc_tf: list[dict[str, int]] = []
        self._cached_doc_lens: list[int] = []
        self._cached_df: dict[str, int] = {}
        self._avg_doc_len: float = 1.0

    @staticmethod
    def _sanitize_crystal_text(text: str) -> str:
        """Strips XML tags and tool-invocation directives to prevent Memory Poisoning attacks."""
        cleaned = re.sub(r"<[^>]+>", "", text)
        cleaned = re.sub(
            r"\[(?:PLUGIN|ПЛАГИН|OS_CMD|КОМАНДА ОС|DIARY|ЗАПИСЬ ДНЕВНИКА)\]\s*:",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", cleaned).strip()

    def _load_history(self) -> list[dict[str, str]]:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data[-self.max_window :]
            except Exception as exc:
                logger.warning("Failed to load chat history from %s: %s", self.history_file, exc)
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
            except OSError as exc:
                logger.error("Atomic history save failed: %s", exc)

    def add_message(self, role: str, content: str) -> list[dict[str, str]]:
        """Appends message to sliding window and returns evicted messages on overflow."""
        with self._lock:
            self.short_term.append({
                "role": role,
                "content": content,
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            evicted: list[dict[str, str]] = []
            if len(self.short_term) > self.max_window:
                overflow = len(self.short_term) - self.max_window
                evicted = self.short_term[:overflow]
                self.short_term = self.short_term[overflow:]
        self.save_history()
        return evicted

#Блок 2 из 2: Запись с Jaccard-дедупликацией, двуязычный токенизатор, Okapi BM25 и телеметрия

    def write_to_diary(self, text: str) -> bool:
        """Appends a deduplicated, sanitized, timestamped memory crystal with ring-buffer rotation."""
        if not text or not text.strip():
            return False

        safe_text = self._sanitize_crystal_text(text)
        new_tokens = set(self._tokenize(safe_text))
        if len(new_tokens) < 2:
            return False

        with self._lock:
            try:
                dir_name = os.path.dirname(self.diary_file)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)

                existing_lines: list[str] = []
                if os.path.exists(self.diary_file):
                    with open(self.diary_file, "r", encoding="utf-8") as f:
                        existing_lines = [line.strip() for line in f if line.strip()]

                # Semantic Jaccard similarity deduplication across recent 100 crystals
                for line in existing_lines[-100:]:
                    body = re.sub(r"^\[[^\]]+\]\s*", "", line)
                    if safe_text.lower() == body.lower():
                        return False
                    existing_tokens = set(self._tokenize(body))
                    if existing_tokens:
                        union_len = len(new_tokens | existing_tokens)
                        if union_len > 0 and (len(new_tokens & existing_tokens) / union_len) >= self.DEDUP_JACCARD_THRESHOLD:
                            return False

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                existing_lines.append(f"[{timestamp}] {safe_text}")

                if len(existing_lines) > self.MAX_DIARY_RECORDS:
                    existing_lines = existing_lines[-self.MAX_DIARY_RECORDS :]

                tmp_diary = f"{self.diary_file}.tmp"
                with open(tmp_diary, "w", encoding="utf-8") as f:
                    f.write("\n".join(existing_lines) + "\n")
                os.replace(tmp_diary, self.diary_file)

                # Rebuild RAM index immediately in-memory without extra disk read
                self._rebuild_index_from_lines(existing_lines, os.path.getmtime(self.diary_file))
                return True
            except OSError as exc:
                logger.error("Failed to write memory crystal: %s", exc)
                return False

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        """Extracts normalized word stems for lightweight Russian/English matching."""
        raw_words = re.findall(r"[а-яёa-z0-9]{2,}", text.lower())
        stems: list[str] = []
        for w in raw_words:
            if w in cls.STOP_WORDS or (len(w) < 3 and not w.isdigit()):
                continue
            if len(w) > 5 and re.search(r"[а-яё]", w):
                w = re.sub(
                    r"(иями|ями|ами|ого|ему|ому|ыми|ими|ее|ие|ые|ое|ей|ий|ый|ой|ем|им|ым|ом|ах|ях|у|ю|а|я|е|и|ы|о)$",
                    "",
                    w,
                )
            elif len(w) > 4 and w.isascii() and w.isalpha():
                w = re.sub(r"(ing|ed|es|ly|s)$", "", w)
            stems.append(w)
        return stems

    def _rebuild_index_from_lines(self, records: list[str], mtime: float) -> None:
        """Builds Okapi BM25 term-frequency and document-frequency tables in RAM."""
        doc_tf: list[dict[str, int]] = []
        doc_lens: list[int] = []
        df: dict[str, int] = {}

        for rec in records:
            clean_rec = re.sub(r"^\[[^\]]+\]\s*", "", rec)
            tokens = self._tokenize(clean_rec)
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for unique_t in tf:
                df[unique_t] = df.get(unique_t, 0) + 1
            doc_tf.append(tf)
            doc_lens.append(len(tokens))

        self._cached_records = records
        self._cached_doc_tf = doc_tf
        self._cached_doc_lens = doc_lens
        self._cached_df = df
        self._avg_doc_len = (sum(doc_lens) / len(doc_lens)) if doc_lens else 1.0
        self._cache_mtime = mtime

    def _refresh_index_if_needed(self) -> None:
        """Rebuilds BM25 index in RAM only if the diary file was modified externally."""
        if not os.path.exists(self.diary_file):
            self._rebuild_index_from_lines([], -1.0)
            return

        try:
            mtime = os.path.getmtime(self.diary_file)
        except OSError:
            return

        if mtime == self._cache_mtime and self._cached_records:
            return

        try:
            with open(self.diary_file, "r", encoding="utf-8") as f:
                records = [line.strip() for line in f if line.strip()]
            self._rebuild_index_from_lines(records, mtime)
        except OSError as exc:
            logger.warning("Failed to refresh diary index: %s", exc)

    def retrieve_crystals(self, query: str, top_n: int = 3) -> str:
        """Retrieves top-N memories using Okapi BM25 ranking + recency boost."""
        if not query:
            return ""

        query_terms = set(self._tokenize(query))
        if not query_terms:
            return ""

        with self._lock:
            self._refresh_index_if_needed()
            records = list(self._cached_records)
            doc_tf = list(self._cached_doc_tf)
            doc_lens = list(self._cached_doc_lens)
            df = dict(self._cached_df)
            avgdl = max(1.0, self._avg_doc_len)

        total_docs = len(records)
        if total_docs == 0:
            return ""

        # Standard Okapi BM25 hyperparameters
        k1 = 1.5
        b = 0.75
        scored: list[tuple[float, str]] = []

        for idx, (raw_rec, tf_map, d_len) in enumerate(zip(records, doc_tf, doc_lens)):
            bm25_score = 0.0
            for term in query_terms:
                freq = tf_map.get(term, 0)
                if freq == 0:
                    continue
                doc_freq = df.get(term, 0)
                idf = math.log(1.0 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                denom = freq + k1 * (1.0 - b + b * (d_len / avgdl))
                bm25_score += idf * ((freq * (k1 + 1.0)) / denom)

            if bm25_score <= 0.0:
                continue

            recency_boost = 1.0 + 0.25 * (idx / max(1, total_docs - 1))
            final_score = bm25_score * recency_boost

            if final_score > 0.45:
                safe_out = self._sanitize_crystal_text(raw_rec)
                scored.append((final_score, safe_out))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return ""

        best = [text for _, text in scored[:top_n]]
        return "\n[Relevant Long-Term Memories]:\n" + "\n".join(best)

    def get_stats(self) -> dict[str, Any]:
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
            "scoring_engine": "okapi-bm25",
        }