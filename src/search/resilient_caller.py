from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.search.cache import RequestCache
from src.search.throttle import AimdThrottle


StatusProbe = Callable[[], "tuple[int | None, float | None]"]


@dataclass
class ResilientCaller:
    """Cache + AIMD throttle + 429 retry helper for HTTP search clients.

    Each search client owns its own raw HTTP fetch closure and passes it to
    `call(...)`. The caller then handles cache hit/miss, throttle acquire/
    release, 429 detection (via optional `last_status` probe) and exponential
    backoff. Cache key is derived from (source, term, page, row_count, fields).
    """

    cache: RequestCache | None = None
    throttle: AimdThrottle | None = None
    max_retries: int = 5
    retry_base_sleep_sec: float = 2.0
    retry_max_sleep_sec: float = 60.0
    api_error_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.throttle is None:
            self.throttle = AimdThrottle()

    def call(
        self,
        *,
        source: str,
        term: str,
        cur_page: int = 1,
        row_count: int = 0,
        fields: list[str] | None = None,
        fetch: Callable[[], list[dict[str, Any]]],
        last_status: StatusProbe | None = None,
    ) -> list[dict[str, Any]]:
        key = RequestCache.make_key(
            source=source,
            term=term,
            cur_page=cur_page,
            row_count=row_count,
            fields=list(fields or []),
        )
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                if isinstance(cached, list):
                    return cached
                return list(cached.get("rows", []))

        attempt = 0
        while True:
            attempt += 1
            assert self.throttle is not None
            self.throttle.acquire()
            try:
                rows = fetch()
            except Exception as exc:
                self.api_error_count += 1
                logging.error(
                    "%s search failed (term=%s, page=%s): %s", source, term, cur_page, exc
                )
                rows = []
            finally:
                self.throttle.release()

            status, retry_after = (last_status() if last_status else (None, None))
            if status == 429:
                self.api_error_count += 1
                self.throttle.report_429(retry_after)
                if attempt <= self.max_retries:
                    time.sleep(self._backoff(attempt, retry_after))
                    continue
                logging.warning(
                    "%s 429 persisted after %s retries (term=%s, page=%s)",
                    source,
                    self.max_retries,
                    term,
                    cur_page,
                )
                return []

            self.throttle.report_success()
            if self.cache is not None:
                self.cache.put(key, rows or [])
            return rows or []

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(self.retry_max_sleep_sec, max(0.0, retry_after))
        return min(
            self.retry_max_sleep_sec,
            self.retry_base_sleep_sec * (2 ** max(0, attempt - 1)),
        )

    def stats(self) -> dict[str, Any]:
        cache_stats = self.cache.stats if self.cache else None
        throttle_stats = self.throttle.get_counters() if self.throttle else {}
        return {
            "api_error_count": self.api_error_count,
            "cache": {
                "enabled": bool(self.cache and self.cache.enabled),
                "hit_count": cache_stats.hit_count if cache_stats else 0,
                "miss_count": cache_stats.miss_count if cache_stats else 0,
                "write_count": cache_stats.write_count if cache_stats else 0,
            },
            **throttle_stats,
        }
