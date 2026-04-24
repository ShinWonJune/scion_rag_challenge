from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from src.search.base_client import BaseSearchClient
from src.search.cache import RequestCache
from src.search.throttle import AimdThrottle
from src.utils.dedup import remove_duplicates


class ScienceONAdapter(BaseSearchClient):
    """Adapter for immutable ScienceONAPIClient."""

    def __init__(
        self,
        scienceon_client: Any,
        max_pages: int = 5,
        *,
        cache: RequestCache | None = None,
        throttle: AimdThrottle | None = None,
        max_retries: int = 5,
        retry_base_sleep_sec: float = 2.0,
        retry_max_sleep_sec: float = 60.0,
    ):
        self.client = scienceon_client
        self.max_pages = max_pages
        self.cache = cache
        self.throttle = throttle or AimdThrottle()
        self.max_retries = max_retries
        self.retry_base_sleep_sec = retry_base_sleep_sec
        self.retry_max_sleep_sec = retry_max_sleep_sec
        self.api_error_count = 0
        self._last_status_code: int | None = None
        self._last_retry_after: float | None = None
        self._wrap_session_get()

    def search(self, search_terms: List[str], max_results: int) -> List[Dict]:
        if not search_terms:
            return []

        all_docs: List[Dict] = []

        for page in range(1, self.max_pages + 1):
            if len(all_docs) >= max_results:
                break
            remaining = max_results - len(all_docs)
            page_docs = self._search_page(search_terms, page, remaining)
            if not page_docs:
                break
            all_docs.extend(page_docs)

        return remove_duplicates(all_docs, key="doc_id", fallback_keys=("title",))[:max_results]

    def get_request_stats(self) -> dict[str, Any]:
        cache_stats = self.cache.stats if self.cache else None
        throttle_stats = self.throttle.get_counters() if self.throttle else {}
        return {
            "api_error_count": self.api_error_count,
            "cache": {
                "enabled": bool(self.cache and self.cache.enabled),
                "hit_count": cache_stats.hit_count if cache_stats else 0,
                "miss_count": cache_stats.miss_count if cache_stats else 0,
                "write_count": cache_stats.write_count if cache_stats else 0,
            },
            **throttle_stats,
        }

    def _wrap_session_get(self) -> None:
        session = getattr(self.client, "session", None)
        if session is None or getattr(session, "_shrag_wrapped", False):
            return

        original_get = session.get

        def wrapped_get(*args: Any, **kwargs: Any):
            response = original_get(*args, **kwargs)
            self._last_status_code = getattr(response, "status_code", None)
            retry_after = response.headers.get("Retry-After")
            self._last_retry_after = self._parse_retry_after(retry_after)
            return response

        session.get = wrapped_get  # type: ignore[method-assign]
        session._shrag_wrapped = True  # type: ignore[attr-defined]

    def _search_page(
        self, search_terms: List[str], page: int, remaining: int
    ) -> List[Dict]:
        page_docs: List[Dict] = []
        for term in search_terms:
            rows = self._search_articles_with_resilience(
                query=term,
                cur_page=page,
                row_count=10,
                fields=["CN", "title", "abstract", "author", "year", "link"],
            )
            filtered = [self._to_common_schema(r) for r in (rows or []) if self._is_quality_document(r)]
            if not filtered:
                continue

            page_docs.extend(filtered[: min(10, remaining - len(page_docs))])
            if len(page_docs) >= remaining:
                break

        return page_docs

    def _search_articles_with_resilience(
        self,
        *,
        query: str,
        cur_page: int,
        row_count: int,
        fields: list[str],
    ) -> list[dict[str, Any]]:
        key = RequestCache.make_key(
            source="scienceon",
            term=query,
            cur_page=cur_page,
            row_count=row_count,
            fields=fields,
        )
        cached = self.cache.get(key) if self.cache else None
        if cached is not None:
            if isinstance(cached, list):
                return cached
            return list(cached.get("rows", []))

        attempt = 0
        while True:
            attempt += 1
            self._last_status_code = None
            self._last_retry_after = None
            self.throttle.acquire()
            try:
                rows = self.client.search_articles(
                    query=query,
                    cur_page=cur_page,
                    row_count=row_count,
                    fields=fields,
                )
            except Exception as exc:
                self.api_error_count += 1
                logging.error("ScienceON search failed (term=%s, page=%s): %s", query, cur_page, exc)
                rows = []
            finally:
                self.throttle.release()

            if self._last_status_code == 429:
                self.api_error_count += 1
                self.throttle.report_429(self._last_retry_after)
                if attempt <= self.max_retries:
                    time.sleep(self._backoff_sleep(attempt, self._last_retry_after))
                    continue
                logging.warning(
                    "ScienceON 429 persisted after %s retries (term=%s, page=%s)",
                    self.max_retries,
                    query,
                    cur_page,
                )
                return []

            self.throttle.report_success()
            if self.cache is not None:
                self.cache.put(key, rows or [])
            return rows or []

    def _backoff_sleep(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(self.retry_max_sleep_sec, max(0.0, retry_after))
        sleep_sec = self.retry_base_sleep_sec * (2 ** max(0, attempt - 1))
        return min(self.retry_max_sleep_sec, sleep_sec)

    def _parse_retry_after(self, value: str | None) -> float | None:
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _is_quality_document(self, row: Dict[str, Any]) -> bool:
        title = str(row.get("title", "") or "").strip()
        if not title or len(title) < 5:
            return False

        abstract = str(row.get("abstract", "") or "").strip()
        if (not abstract or abstract == "없음") and len(title) < 20:
            return False

        cn = str(row.get("CN", "") or "").strip()
        return bool(cn)

    def _to_common_schema(self, row: Dict[str, Any]) -> Dict[str, Any]:
        doc_id = str(row.get("CN", "") or "").strip()
        return {
            "doc_id": doc_id,
            "CN": doc_id,
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "authors": row.get("author", ""),
            "year": row.get("year", ""),
            "url": row.get("link", "")
            or (
                f"http://click.ndsl.kr/servlet/OpenAPIDetailView?keyValue={doc_id}&target=NART&cn={doc_id}"
                if doc_id
                else ""
            ),
            "source": "ScienceON",
        }
