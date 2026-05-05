from pathlib import Path

from shrag.search.cache import RequestCache


def test_cache_round_trip(tmp_path: Path) -> None:
    cache = RequestCache(tmp_path)
    key = cache.make_key("scienceon", "term", 1, 10, ["title"])
    assert cache.get(key) is None
    cache.put(key, [{"title": "doc"}])
    assert cache.get(key) == [{"title": "doc"}]
