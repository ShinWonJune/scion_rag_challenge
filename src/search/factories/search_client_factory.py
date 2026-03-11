from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.search.base_client import BaseSearchClient


class _DryRunSearchClient(BaseSearchClient):
    def __init__(self, source: str):
        self.source = source

    def search(self, search_terms: list[str], max_results: int) -> list[dict]:
        return []


def create_search_client(source: str, config: Dict[str, Any]) -> BaseSearchClient:
    source_norm = source.strip().lower()
    if config.get("dry_run"):
        return _DryRunSearchClient(source_norm)

    if source_norm == "scienceon":
        from src.search.scienceon_adapter import ScienceONAdapter
        from src.search.clients.scienceon_api_example import ScienceONAPIClient

        credentials_path = Path(
            config.get(
                "scienceon_credentials_path",
                "configs/credentials/scienceon_api_credentials.json",
            )
        )
        return ScienceONAdapter(
            ScienceONAPIClient(credentials_path),
            max_pages=int(config.get("scienceon_max_pages", 5)),
        )

    if source_norm == "pubmed":
        from src.search.clients.pubmed_api_client import PubMedAPIClient

        credentials_path = Path(
            config.get(
                "pubmed_credentials_path",
                "configs/credentials/pubmed_api_credentials.json",
            )
        )
        return PubMedAPIClient(credentials_path)

    if source_norm == "wikipedia":
        from src.search.clients.wikipedia_api_client import WikipediaAPIClient

        return WikipediaAPIClient(lang=config.get("lang", "ko"))

    raise ValueError(f"Unsupported search source: {source}")
