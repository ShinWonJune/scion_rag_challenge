from pathlib import Path

from src.search.cache import RequestCache
from src.search.resilient_caller import ResilientCaller


def test_cache_hit_skips_fetch(tmp_path: Path) -> None:
    cache = RequestCache(tmp_path)
    caller = ResilientCaller(cache=cache)
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return [{"doc_id": "1"}]

    args = dict(source="x", term="t", cur_page=1, row_count=5, fields=["a"])
    first = caller.call(**args, fetch=fetch)
    second = caller.call(**args, fetch=fetch)
    assert first == second == [{"doc_id": "1"}]
    assert calls["n"] == 1


def test_429_retry_succeeds_after_recovery() -> None:
    states = iter([(429, 0.0), (429, 0.0), (None, None)])
    caller = ResilientCaller(max_retries=5, retry_base_sleep_sec=0.0, retry_max_sleep_sec=0.0)

    out = caller.call(
        source="x",
        term="t",
        cur_page=1,
        row_count=1,
        fields=[],
        fetch=lambda: [{"ok": True}],
        last_status=lambda: next(states),
    )
    assert out == [{"ok": True}]
    assert caller.api_error_count == 2


def test_429_persists_returns_empty() -> None:
    caller = ResilientCaller(max_retries=2, retry_base_sleep_sec=0.0, retry_max_sleep_sec=0.0)
    out = caller.call(
        source="x",
        term="t",
        cur_page=1,
        row_count=1,
        fields=[],
        fetch=lambda: [{"x": 1}],
        last_status=lambda: (429, 0.0),
    )
    assert out == []


def test_stats_shape() -> None:
    caller = ResilientCaller()
    stats = caller.stats()
    assert "api_error_count" in stats
    assert "cache" in stats and stats["cache"]["enabled"] is False
    assert "request_count" in stats
