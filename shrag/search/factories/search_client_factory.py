"""Public entry point for building search clients.

Backwards-compatible re-export of `create_search_client`. The actual logic
lives in `src.search.factories.registry`. Importing this module also imports
`src.search.factories.builders` so all built-in sources self-register.
"""

from __future__ import annotations

from shrag.search.factories import builders  # noqa: F401  # ensures source registration
from shrag.search.factories.registry import (
    create_search_client,
    list_search_clients,
    register_search_client,
)

__all__ = ["create_search_client", "list_search_clients", "register_search_client"]
