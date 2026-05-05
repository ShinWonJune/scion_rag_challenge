from __future__ import annotations

from typing import Any, Callable

from shrag.search.base_client import BaseSearchClient


SearchClientBuilder = Callable[[dict[str, Any]], BaseSearchClient]
_REGISTRY: dict[str, SearchClientBuilder] = {}


def register_search_client(name: str) -> Callable[[SearchClientBuilder], SearchClientBuilder]:
    """Decorator: register a builder for a named source.

    Usage:
        @register_search_client("scienceon")
        def build(config): return ScienceONAdapter(...)
    """

    def deco(builder: SearchClientBuilder) -> SearchClientBuilder:
        _REGISTRY[name.strip().lower()] = builder
        return builder

    return deco


def create_search_client(source: str, config: dict[str, Any]) -> BaseSearchClient:
    """Build a search client for `source` using the registry.

    `dry_run` config short-circuits to an empty client without touching IO.
    """

    key = source.strip().lower()
    if config.get("dry_run"):
        from shrag.search.factories.builders import DryRunSearchClient

        return DryRunSearchClient(key)

    builder = _REGISTRY.get(key)
    if builder is None:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ValueError(f"Unsupported search source: {source!r}. Available: {available}")
    return builder(config)


def list_search_clients() -> list[str]:
    return sorted(_REGISTRY)
