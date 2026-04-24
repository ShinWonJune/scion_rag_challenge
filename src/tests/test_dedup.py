from src.utils.dedup import remove_duplicates


def test_prefers_doc_id_over_title() -> None:
    docs = [
        {"doc_id": "CN1", "title": "same title"},
        {"doc_id": "CN2", "title": "same title"},
    ]
    assert len(remove_duplicates(docs)) == 2


def test_fallback_to_title_when_doc_id_missing() -> None:
    docs = [{"title": "A"}, {"title": "A"}]
    assert len(remove_duplicates(docs)) == 1
