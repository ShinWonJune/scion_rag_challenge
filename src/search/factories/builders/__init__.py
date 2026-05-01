"""Search client builders.

Importing this package registers every built-in source with the registry.
To add a new source, drop a file `<name>_builder.py` in this directory that
calls `@register_search_client("<name>")` and append it to the import block
below.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.search.base_client import BaseSearchClient


class DryRunSearchClient(BaseSearchClient):
    source_name = "dry-run"

    def __init__(self, source: str):
        self.source = source
        self.source_name = source

    def search(self, search_terms: List[str], max_results: int) -> List[Dict[str, Any]]:
        return []


from . import scienceon_builder  # noqa: E402,F401
from . import pubmed_builder  # noqa: E402,F401
from . import wikipedia_builder  # noqa: E402,F401
