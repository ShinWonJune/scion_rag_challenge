from __future__ import annotations

from pathlib import Path
from typing import Any

from src.search.base_client import BaseSearchClient
from src.search.cache import RequestCache
from src.search.factories.registry import register_search_client
from src.search.resilient_caller import ResilientCaller
from src.search.throttle import AimdThrottle


@register_search_client("scienceon")
def build_scienceon(config: dict[str, Any]) -> BaseSearchClient:
    from src.search.clients.scienceon_api_example import ScienceONAPIClient
    from src.search.scienceon_adapter import ScienceONAdapter

    credentials_path = Path(
        config.get(
            "scienceon_credentials_path",
            "configs/credentials/scienceon_api_credentials.json",
        )
    )
    cache_root = Path(
        config.get("cache_root")
        or config.get("scienceon_cache_root")
        or Path("outputs") / "_shared_cache"
    )
    cache = RequestCache(root=cache_root, enabled=not bool(config.get("disable_cache")))
    throttle = AimdThrottle(
        max_concurrency=int(config.get("scienceon_max_concurrency", 2)),
        min_interval_sec=float(config.get("scienceon_min_interval_sec", 0.5)),
        increase_step=float(config.get("scienceon_throttle_aimd_increase", 0.25)),
        decrease_after_success=int(config.get("scienceon_throttle_aimd_decrease_after", 20)),
        interval_cap_sec=float(config.get("scienceon_interval_cap_sec", 10.0)),
        fixed_concurrency=bool(config.get("scienceon_fixed_concurrency", False)),
    )
    caller = ResilientCaller(
        cache=cache,
        throttle=throttle,
        max_retries=int(config.get("scienceon_max_retries", 5)),
        retry_base_sleep_sec=float(config.get("scienceon_retry_base_sleep_sec", 2.0)),
        retry_max_sleep_sec=float(config.get("scienceon_retry_max_sleep_sec", 60.0)),
    )
    return ScienceONAdapter(
        ScienceONAPIClient(credentials_path),
        max_pages=int(config.get("scienceon_max_pages", 5)),
        caller=caller,
    )
