from __future__ import annotations

from pathlib import Path
from typing import Any

from src.search.base_client import BaseSearchClient
from src.search.cache import RequestCache
from src.search.factories.registry import register_search_client
from src.search.resilient_caller import ResilientCaller
from src.search.throttle import AimdThrottle


@register_search_client("wikipedia")
def build_wikipedia(config: dict[str, Any]) -> BaseSearchClient:
    from src.search.clients.wikipedia_api_client import WikipediaAPIClient

    cache_root = Path(
        config.get("cache_root")
        or config.get("wikipedia_cache_root")
        or Path("outputs") / "_shared_cache"
    )
    cache = RequestCache(root=cache_root, enabled=not bool(config.get("disable_cache")))
    throttle = AimdThrottle(
        max_concurrency=int(config.get("wikipedia_max_concurrency", 4)),
        min_interval_sec=float(config.get("wikipedia_min_interval_sec", 0.1)),
        fixed_concurrency=bool(config.get("wikipedia_fixed_concurrency", True)),
    )
    caller = ResilientCaller(
        cache=cache,
        throttle=throttle,
        max_retries=int(config.get("wikipedia_max_retries", 3)),
        retry_base_sleep_sec=float(config.get("wikipedia_retry_base_sleep_sec", 1.0)),
        retry_max_sleep_sec=float(config.get("wikipedia_retry_max_sleep_sec", 30.0)),
    )
    return WikipediaAPIClient(
        lang=config.get("lang", "ko"),
        caller=caller,
    )
