from __future__ import annotations

from typing import Any, Dict, List

from src.search.base_client import BaseSearchClient
from src.search.resilient_caller import ResilientCaller
from src.utils.dedup import remove_duplicates


SCIENCEON_FIELDS = ["CN", "title", "abstract", "author", "year", "link"]
SCIENCEON_ROW_COUNT = 10


class ScienceONAdapter(BaseSearchClient):
    """Adapter for the immutable ScienceONAPIClient.

    Converts raw ScienceON rows to the common schema and delegates HTTP
    request lifecycle (cache / throttle / 429 retry) to a ResilientCaller.
    """

    source_name = "ScienceON"

    def __init__(
        self,
        scienceon_client: Any,
        max_pages: int = 5,
        *,
        caller: ResilientCaller | None = None,
    ):
        self.client = scienceon_client
        self.max_pages = max_pages
        self.caller = caller or ResilientCaller()
        self._last_status_code: int | None = None
        self._last_retry_after: float | None = None
        self._wrap_session_get()

    def search(self, search_terms: List[str], max_results: int) -> List[Dict[str, Any]]:
        if not search_terms:
            return []

        all_docs: List[Dict[str, Any]] = []
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
        return self.caller.stats()

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
    ) -> List[Dict[str, Any]]:
        page_docs: List[Dict[str, Any]] = []
        for term in search_terms:
            rows = self._fetch_term_page(term, page)
            filtered = [self._to_common_schema(r) for r in rows if self._is_quality_document(r)]
            if not filtered:
                continue
            page_docs.extend(filtered[: min(SCIENCEON_ROW_COUNT, remaining - len(page_docs))])
            if len(page_docs) >= remaining:
                break
        return page_docs

    def _fetch_term_page(self, term: str, page: int) -> list[dict[str, Any]]:
        def _do_fetch() -> list[dict[str, Any]]:
            self._last_status_code = None
            self._last_retry_after = None
            return self.client.search_articles(
                query=term,
                cur_page=page,
                row_count=SCIENCEON_ROW_COUNT,
                fields=SCIENCEON_FIELDS,
            ) or []

        return self.caller.call(
            source="scienceon",
            term=term,
            cur_page=page,
            row_count=SCIENCEON_ROW_COUNT,
            fields=SCIENCEON_FIELDS,
            fetch=_do_fetch,
            last_status=lambda: (self._last_status_code, self._last_retry_after),
        )

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _is_quality_document(row: Dict[str, Any]) -> bool:
        title = str(row.get("title", "") or "").strip()
        if not title or len(title) < 5:
            return False
        abstract = str(row.get("abstract", "") or "").strip()
        if (not abstract or abstract == "없음") and len(title) < 20:
            return False
        return bool(str(row.get("CN", "") or "").strip())

    @staticmethod
    def _to_common_schema(row: Dict[str, Any]) -> Dict[str, Any]:
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
