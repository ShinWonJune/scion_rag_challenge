from __future__ import annotations

from typing import Any, Dict, List

from src.search.base_client import BaseSearchClient


class ScienceONAdapter(BaseSearchClient):
    """Adapter for immutable ScienceONAPIClient."""

    def __init__(self, scienceon_client: Any):
        self.client = scienceon_client

    def search(self, search_terms: List[str], max_results: int) -> List[Dict]:
        docs: List[Dict] = []
        if not search_terms:
            return docs

        per_term = max(1, min(10, max_results // max(1, len(search_terms))))
        for term in search_terms:
            rows = self.client.search_articles(
                query=term,
                cur_page=1,
                row_count=per_term,
                fields=["CN", "title", "abstract", "author", "year", "link"],
            )
            for row in rows or []:
                doc_id = row.get("CN", "")
                docs.append(
                    {
                        "doc_id": doc_id,
                        "title": row.get("title", ""),
                        "abstract": row.get("abstract", ""),
                        "authors": row.get("author", ""),
                        "year": row.get("year", ""),
                        "url": row.get("link", "") or (
                            f"http://click.ndsl.kr/servlet/OpenAPIDetailView?keyValue={doc_id}&target=NART&cn={doc_id}"
                            if doc_id
                            else ""
                        ),
                        "source": "ScienceON",
                    }
                )
                if len(docs) >= max_results:
                    return docs
        return docs

