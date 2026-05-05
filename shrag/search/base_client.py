from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseSearchClient(ABC):
    """Common interface for academic search platforms.

    Output document schema:
        doc_id, title, abstract, authors, year, url, source

    `source` value: "ScienceON" | "PubMed" | "Wikipedia" | ...
    """

    source_name: str = ""

    @abstractmethod
    def search(self, search_terms: List[str], max_results: int) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_request_stats(self) -> Dict[str, Any]:
        return {}
