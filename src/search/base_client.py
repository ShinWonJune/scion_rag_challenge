from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseSearchClient(ABC):
    @abstractmethod
    def search(self, search_terms: List[str], max_results: int) -> List[Dict]:
        """Return schema:
        {
          "doc_id", "title", "abstract", "authors", "year", "url", "source"
        }
        source: "ScienceON" | "PubMed" | "Wikipedia"
        """
        raise NotImplementedError

