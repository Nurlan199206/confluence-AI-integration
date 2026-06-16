"""
title: Confluence Knowledge Base Search
author: nurlan
version: 2.0.0
requirements: requests, qdrant-client, sentence-transformers
"""

import re
import requests
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

STOPWORDS = [
    "кто такой",
    "кто такая",
    "кто такие",
    "что такое",
    "что это",
    "расскажи про",
    "расскажи о",
    "информация о",
    "инфо о",
    "найди",
    "покажи",
    "дай",
    "кто",
    "что",
    "где",
    "как",
]


class Tools:
    class Valves(BaseModel):
        CONFLUENCE_BASE_URL: str = Field(default="https://confluence.test.kz")
        CONFLUENCE_TOKEN: str = Field(
            default="...token"
        )
        QDRANT_URL: str = Field(default="http://qdrant:6333")
        QDRANT_COLLECTION: str = Field(default="confluence_kb")

    def __init__(self):
        self.valves = self.Valves()
        self.citation = False
        self._qdrant = None
        self._embedder = None

    # ---------- вспомогательные методы (не видны модели как отдельные тулы) ----------

    def _clean_query(self, query: str) -> str:
        q = query.lower()
        for sw in STOPWORDS:
            q = q.replace(sw, " ")
        q = re.sub(r"[^\w\s]", " ", q, flags=re.UNICODE)
        q = re.sub(r"\s+", " ", q).strip()
        return q or query.strip()

    def _run_cql(self, url: str, headers: dict, cql: str, limit: int = 5):
        params = {"cql": cql, "limit": limit, "expand": "body.view"}
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("results", [])

    def _confluence_content_search(self, query: str):
        url = f"{self.valves.CONFLUENCE_BASE_URL}/rest/api/content/search"
        headers = {"Authorization": f"Bearer {self.valves.CONFLUENCE_TOKEN}"}

        # 1. Точная фраза
        results = self._run_cql(url, headers, f'text ~ "{query}"')
        if results:
            return results

        # 2. Если не нашли — ищем по отдельным словам через OR
        tokens = [t for t in query.split() if len(t) > 2]
        if len(tokens) > 1:
            cql_or = " OR ".join(f'text ~ "{t}"' for t in tokens)
            results = self._run_cql(url, headers, f"({cql_or})")
        return results

    def _confluence_people_search(self, query: str):
        url = f"{self.valves.CONFLUENCE_BASE_URL}/rest/api/search"
        headers = {"Authorization": f"Bearer {self.valves.CONFLUENCE_TOKEN}"}
        return self._run_cql(url, headers, f'type=user and text ~ "{query}"')

    def _vector_search(self, query: str):
        if self._qdrant is None:
            self._qdrant = QdrantClient(url=self.valves.QDRANT_URL)
        if self._embedder is None:
            self._embedder = SentenceTransformer("intfloat/multilingual-e5-large")
        vector = self._embedder.encode(query).tolist()
        return self._qdrant.search(
            collection_name=self.valves.QDRANT_COLLECTION,
            query_vector=vector,
            limit=5,
        )

    def _format_content(self, results) -> str:
        chunks = []
        for r in results:
            title = r.get("title", "")
            html = r.get("body", {}).get("view", {}).get("value", "")
            page_url = f"{self.valves.CONFLUENCE_BASE_URL}/pages/viewpage.action?pageId={r.get('id')}"
            chunks.append(f"[{title}]({page_url})\n{html[:1500]}")
        return "\n\n---\n\n".join(chunks)

    def _format_people(self, results) -> str:
        chunks = []
        for r in results:
            user = r.get("user", {})
            chunks.append(
                f"Имя: {user.get('displayName', '')}\n"
                f"Email: {user.get('email', 'скрыт')}\n"
                f"Профиль: {self.valves.CONFLUENCE_BASE_URL}/display/~{user.get('username', '')}"
            )
        return "\n\n---\n\n".join(chunks)

    def _format_vector(self, hits) -> str:
        return "\n\n---\n\n".join(
            f"[{h.payload.get('title', '')}]\n{h.payload.get('text', '')}" for h in hits
        )

    # ---------- единственный тул, видимый модели ----------

    def search_knowledge_base(self, query: str) -> str:
        """
        Главный инструмент поиска по базе знаний компании. ВСЕГДА вызывай его
        для любого вопроса, который может касаться компании: документация,
        микросервисы, инфраструктура, процессы, сотрудники, их роли, контакты,
        команды, организационная структура. Никогда не отвечай на такие вопросы
        из собственных знаний — у тебя нет информации о компании без этого тула.
        В query передавай только ключевые слова или имя/фамилию человека, без
        вопросительных слов типа "кто такой" и без знаков вопроса.
        Отвечай пользователю только на русском языке, даже если найденный текст
        на другом языке.

        :param query: ключевые слова, имя сотрудника или название сервиса для поиска
        """
        query = self._clean_query(query)

        content_error = None
        try:
            content_results = self._confluence_content_search(query)
        except Exception as e:
            content_results = []
            content_error = str(e)

        if content_results:
            return self._format_content(content_results)

        try:
            people_results = self._confluence_people_search(query)
        except Exception:
            people_results = []

        if people_results:
            return self._format_people(people_results)

        try:
            vector_results = self._vector_search(query)
        except Exception:
            vector_results = []

        if vector_results:
            return self._format_vector(vector_results)

        if content_error:
            return (
                f"CONFLUENCE_ERROR: {content_error}\n"
                "Инструкция модели: сообщи пользователю на русском, что сервис "
                "поиска временно недоступен, и предложи повторить запрос позже."
            )

        return (
            "NOT_FOUND\n"
            "Инструкция модели: информация не найдена ни в Confluence, ни в "
            "векторной базе. Сообщи об этом пользователю на русском и попроси "
            "уточнить имя/название или ключевые слова."
        )
