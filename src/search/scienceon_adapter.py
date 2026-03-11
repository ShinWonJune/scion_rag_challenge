from __future__ import annotations

from typing import Any, Dict, List

from src.search.base_client import BaseSearchClient
from src.utils.dedup import remove_duplicates


class ScienceONAdapter(BaseSearchClient):
    """Adapter for immutable ScienceONAPIClient."""

    def __init__(self, scienceon_client: Any, max_pages: int = 5):
        self.client = scienceon_client
        self.max_pages = max_pages

    def search(self, search_terms: List[str], max_results: int) -> List[Dict]:
        if not search_terms:
            return []

        all_docs: List[Dict] = []
        used_terms: set[str] = set()

        for page in range(1, self.max_pages + 1):
            if len(all_docs) >= max_results:
                break
            remaining = max_results - len(all_docs)
            page_docs = self._search_page(search_terms, page, used_terms, remaining)
            if not page_docs:
                break
            all_docs.extend(page_docs)

        return remove_duplicates(all_docs, key="title")[:max_results]

    def _search_page(
        self, search_terms: List[str], page: int, used_terms: set[str], remaining: int
    ) -> List[Dict]:
        page_docs: List[Dict] = []
        for term in search_terms:
            if term in used_terms:
                continue

            rows = self.client.search_articles(
                query=term,
                cur_page=page,
                row_count=10,
                fields=["CN", "title", "abstract", "author", "year", "link"],
            )
            filtered = [self._to_common_schema(r) for r in (rows or []) if self._is_quality_document(r)]
            if not filtered:
                continue

            page_docs.extend(filtered[: min(10, remaining - len(page_docs))])
            used_terms.add(term)
            if len(page_docs) >= remaining:
                break

        return page_docs

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
