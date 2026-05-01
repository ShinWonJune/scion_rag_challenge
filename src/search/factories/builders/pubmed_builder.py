from __future__ import annotations

from pathlib import Path
from typing import Any

from src.search.base_client import BaseSearchClient
from src.search.cache import RequestCache
from src.search.factories.registry import register_search_client
from src.search.resilient_caller import ResilientCaller
from src.search.throttle import AimdThrottle


@register_search_client("pubmed")
def build_pubmed(config: dict[str, Any]) -> BaseSearchClient:
    from src.search.clients.pubmed_api_client import PubMedAPIClient

    credentials_path = Path(
        config.get(
            "pubmed_credentials_path",
            "configs/credentials/pubmed_api_credentials.json",
        )
    )
    cache_root = Path(
        config.get("cache_root")
        or config.get("pubmed_cache_root")
        or Path("outputs") / "_shared_cache"
    )
    cache = RequestCache(root=cache_root, enabled=not bool(config.get("disable_cache")))
    throttle = AimdThrottle(
        max_concurrency=int(config.get("pubmed_max_concurrency", 2)),
        min_interval_sec=float(config.get("pubmed_min_interval_sec", 0.34)),
        fixed_concurrency=bool(config.get("pubmed_fixed_concurrency", True)),
    )
    caller = ResilientCaller(
        cache=cache,
        throttle=throttle,
        max_retries=int(config.get("pubmed_max_retries", 3)),
        retry_base_sleep_sec=float(config.get("pubmed_retry_base_sleep_sec", 1.0)),
        retry_max_sleep_sec=float(config.get("pubmed_retry_max_sleep_sec", 30.0)),
    )
    return PubMedAPIClient(
        credentials_path,
        caller=caller,
        per_term_max_results=int(config.get("pubmed_per_term_max_results", 10)),
        max_terms=int(config.get("pubmed_max_terms", 10)),
    )
