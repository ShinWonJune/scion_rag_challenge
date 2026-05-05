import pytest

from shrag.search.factories.search_client_factory import (
    create_search_client,
    list_search_clients,
    register_search_client,
)


def test_builtin_sources_registered() -> None:
    sources = set(list_search_clients())
    assert {"scienceon", "pubmed", "wikipedia"} <= sources


def test_dry_run_returns_empty_client() -> None:
    for src in ["scienceon", "pubmed", "wikipedia"]:
        client = create_search_client(src, {"dry_run": True})
        assert client.search(["term"], 5) == []
        assert client.get_request_stats() == {}


def test_unknown_source_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported search source"):
        create_search_client("definitely-not-a-source", {})


def test_custom_source_can_be_registered() -> None:
    from shrag.search.base_client import BaseSearchClient

    class _FakeClient(BaseSearchClient):
        source_name = "fake"

        def search(self, search_terms, max_results):
            return [{"doc_id": "f1", "source": "Fake"}]

    @register_search_client("fake-test-source")
    def _build(config):
        return _FakeClient()

    try:
        client = create_search_client("fake-test-source", {})
        assert client.search([], 1)[0]["source"] == "Fake"
    finally:
        from shrag.search.factories.registry import _REGISTRY

        _REGISTRY.pop("fake-test-source", None)
