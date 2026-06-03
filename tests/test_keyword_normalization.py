from shrag.search.extractors.llm_keyword_extractor import LLMKeywordExtractor


class DummyKeywordExtractor(LLMKeywordExtractor):
    def _call_llm(self, prompt: str) -> str:
        return ""


def test_legacy_keyword_parser_filters_colons_and_quotes():
    parser = DummyKeywordExtractor()
    out = parser._parse_keyword_list(
        'Keywords: "machine learning", vector:space, "AI"',
        prefix="Keywords:",
    )
    assert out == ["machine", "learning", "vector", "space", "AI"]


def test_legacy_keyword_parser_filters_escaped_quotes():
    parser = DummyKeywordExtractor()
    out = parser._parse_keyword_list(
        r'Keywords: \"논리와 구조\", normal',
        prefix="Keywords:",
    )
    assert out == ["논리와", "구조", "normal"]
