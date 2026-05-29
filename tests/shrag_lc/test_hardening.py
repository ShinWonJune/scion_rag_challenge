from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from shrag.search.resilient_caller import ResilientCaller

from shrag_lc.config import PipelineConfig
from shrag_lc.corpus import build_corpus
from shrag_lc.generate import AnswerGenerator
from shrag_lc.pipeline import SHRAGPipeline
from shrag_lc.reporting import build_manifest
from shrag_lc.schemas import PipelineStatus, StageName, StageStatus, stage_report
from shrag_lc.search.acquire import Acquirer
from shrag_lc.search.clients import ScienceONClient
from shrag_lc.search.keywords import KeywordExtractor
from shrag_lc.vectorstore import build_faiss, load_faiss_artifact, save_faiss_artifact


def test_keyword_extractor_structured_json_and_retry():
    llm = FakeListChatModel(
        responses=[
            "not json",
            '{"keywords": ["검색", "임베딩", "검색"]}',
            '{"keywords": ["free electronic textbook", "vector-space", "AI", "ai"]}',
        ]
    )
    extractor = KeywordExtractor(llm, language="all")

    result = extractor.extract("RAG embedding search")

    assert result["korean"] == ["검색", "임베딩"]
    assert result["english"] == ["free", "electronic", "textbook", "vector", "space", "AI"]


def test_acquirer_records_partial_search_failure():
    class Extractor:
        def extract(self, query):
            return {"korean": ["실패", "성공"], "english": []}

    class Client:
        source_name = "ScienceON"

        def search(self, search_terms, max_results):
            if search_terms == ["실패|성공"]:
                raise TimeoutError("boom")
            return [
                {
                    "doc_id": "D1",
                    "title": "Title",
                    "abstract": "Abstract",
                    "source": "ScienceON",
                }
            ]

    cfg = PipelineConfig(target_documents=2, number_of_operators=0)
    docs, meta = Acquirer(Extractor(), Client(), cfg).acquire("query")

    assert len(docs) == 1
    search_stage = meta["stages"][StageName.SEARCH]
    assert search_stage["status"] == StageStatus.PARTIAL_SUCCESS
    assert search_stage["counts"]["failed_terms"] == 1
    assert search_stage["errors"][0]["error_type"] == "TimeoutError"


def test_scienceon_client_retries_429_with_resilient_caller():
    class FakeScienceONApi:
        def __init__(self, owner):
            self.owner = owner
            self.calls = 0

        def search_articles(self, query, cur_page, row_count, fields):
            self.calls += 1
            if self.calls == 1:
                self.owner._last_status_code = 429
                self.owner._last_retry_after = 0.0
                return []
            self.owner._last_status_code = None
            self.owner._last_retry_after = None
            return [{"CN": "D1", "title": "Recovered title", "abstract": "Recovered abstract"}]

    client = ScienceONClient.__new__(ScienceONClient)
    client.client = FakeScienceONApi(client)
    client.caller = ResilientCaller(max_retries=2, retry_base_sleep_sec=0.0, retry_max_sleep_sec=0.0)
    client._last_status_code = None
    client._last_retry_after = None

    rows = client._fetch_term_page("term", 1)

    assert rows[0]["CN"] == "D1"
    assert client.client.calls == 2
    assert client.get_request_stats()["rate_limit_count"] == 1


def test_acquirer_records_rate_limit_stats_without_exception():
    class Extractor:
        def extract(self, query):
            return {"korean": ["제한"], "english": []}

    class Client:
        source_name = "ScienceON"

        def __init__(self):
            self.stats = {"rate_limit_count": 0, "request_count": 0, "api_error_count": 0}

        def get_request_stats(self):
            return self.stats

        def search(self, search_terms, max_results):
            self.stats = {"rate_limit_count": 1, "request_count": 1, "api_error_count": 1}
            return []

    docs, meta = Acquirer(Extractor(), Client(), PipelineConfig()).acquire("query")

    assert docs == []
    search_stage = meta["stages"][StageName.SEARCH]
    assert search_stage["status"] == StageStatus.FAILED
    assert search_stage["warnings"] == ["Search client reported 1 rate-limit response(s)."]
    assert search_stage["metadata"]["request_stats_delta"]["rate_limit_count"] == 1


def test_pipeline_graph_returns_structured_status():
    class FakeAcquirer:
        def acquire(self, query):
            return [
                Document(
                    page_content="alpha alpha alpha beta",
                    metadata={"doc_id": "D1", "title": "alpha", "abstract": "beta"},
                )
            ], {
                "source": "test",
                "keywords": {"korean": [], "english": ["alpha"]},
                "search_terms": ["alpha"],
                "document_count": 1,
                "stages": {
                    StageName.SEARCH: stage_report(
                        StageName.SEARCH,
                        StageStatus.SUCCESS,
                        counts={"document_count": 1},
                    ).to_dict()
                },
            }

    pipeline = SHRAGPipeline.__new__(SHRAGPipeline)
    pipeline.cfg = PipelineConfig(use_reranker=False, dense_top_k=1, rerank_top_n=1)
    pipeline.acquirer = FakeAcquirer()
    pipeline.embeddings = DeterministicFakeEmbedding(size=32)
    pipeline.reranker = None
    pipeline.generator = AnswerGenerator(FakeListChatModel(responses=["answer [doc_id=D1]"]))
    pipeline.graph = RunnableLambda(pipeline._run_one_graph)

    result = pipeline.run_one("alpha?", question_id="Q1")

    assert result["id"] == "Q1"
    assert result["status"] == PipelineStatus.SUCCESS
    assert result["stages"][StageName.RETRIEVAL]["metadata"]["index_mode"] == "per_query"
    assert result["used_context"][0]["doc_id"] == "D1"


def test_corpus_dedups_documents_by_source_and_doc_id():
    docs = [
        Document(page_content="a", metadata={"source": "S", "doc_id": "1", "title": "A"}),
        Document(page_content="a2", metadata={"source": "S", "doc_id": "1", "title": "A"}),
        Document(page_content="b", metadata={"source": "S", "doc_id": "2", "title": "B"}),
    ]

    corpus = build_corpus([docs])

    assert corpus.document_count == 2
    assert corpus.duplicate_count == 1
    assert corpus.source_counts == {"S": 2}


def test_faiss_artifact_rejects_mismatched_embedding_config(tmp_path: Path):
    cfg = PipelineConfig(embedding_model="model-a")
    emb = DeterministicFakeEmbedding(size=32)
    docs = [Document(page_content="alpha", metadata={"doc_id": "D1"})]
    store = build_faiss(docs, emb)

    save_faiss_artifact(store, tmp_path / "idx", cfg, document_count=1)

    mismatched = PipelineConfig(embedding_model="model-b")
    try:
        load_faiss_artifact(tmp_path / "idx", emb, mismatched)
    except ValueError as exc:
        assert "mismatched embedding_model" in str(exc)
    else:
        raise AssertionError("Expected mismatched index metadata to fail")


def test_manifest_contains_stage_summary_and_failures():
    results = [
        {
            "id": "1",
            "stages": {
                "search": {
                    "status": "failed",
                    "errors": [{"stage": "search", "error_type": "TimeoutError", "message": "boom"}],
                }
            },
            "errors": [],
        }
    ]

    manifest = build_manifest(
        config=PipelineConfig(),
        questions=[{"id": "1", "question": "q"}],
        results=results,
        pipeline_mode="corpus_first",
        generated_at="2026-05-29T00:00:00",
    )

    assert manifest["stage_summary"]["search"]["failed"] == 1
    assert manifest["failures"][0]["error_type"] == "TimeoutError"
