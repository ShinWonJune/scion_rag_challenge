"""Offline smoke tests for the retrieval/rerank/generate wiring.

Uses fake embeddings, a stub cross-encoder, and FakeListChatModel so no model
downloads or network calls are needed.
"""

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from shrag_lc.embeddings import build_embedding_text, doc_to_document
from shrag_lc.generate import AnswerGenerator, build_context_json
from shrag_lc.rerank import CrossEncoderRerankCompressor
from shrag_lc.search.keywords import parse_keyword_list
from shrag_lc.vectorstore import build_faiss


def _docs(n=5):
    return [
        Document(
            page_content=f"title{i} title{i} title{i} abstract about topic {i}",
            metadata={"doc_id": f"D{i}", "title": f"title{i}", "abstract": f"abstract {i}"},
        )
        for i in range(n)
    ]


def test_embedding_text_3ta():
    text = build_embedding_text({"title": "T", "abstract": "A"}, "3*title+abstract")
    assert text == "T T T A"


def test_doc_to_document_metadata():
    doc = doc_to_document(
        {"CN": "CN1", "title": "Ti", "abstract": "Ab", "source": "ScienceON"}, "3*title+abstract"
    )
    assert doc.metadata["doc_id"] == "CN1"
    assert doc.page_content == "Ti Ti Ti Ab"


def test_faiss_build_and_retrieve():
    emb = DeterministicFakeEmbedding(size=64)
    store = build_faiss(_docs(5), emb)
    retriever = store.as_retriever(search_kwargs={"k": 3})
    hits = retriever.invoke("title2 topic 2")
    assert len(hits) == 3
    assert all(isinstance(h, Document) for h in hits)


def test_reranker_compressor_orders_and_truncates():
    class StubModel:
        # score by trailing index so D4 > D3 > ... (reverse of input order)
        def score(self, pairs):
            return [float(i) for i in range(len(pairs))]

    compressor = CrossEncoderRerankCompressor(model=StubModel(), top_n=2)
    out = list(compressor.compress_documents(_docs(5), "q"))
    assert len(out) == 2
    assert out[0].metadata["rank"] == 1 and out[1].metadata["rank"] == 2
    # highest score (last input) ranked first
    assert out[0].metadata["doc_id"] == "D4"
    assert out[0].metadata["rerank_score"] >= out[1].metadata["rerank_score"]


def test_build_context_json_structure():
    import json

    docs = [Document(page_content="c", metadata={"doc_id": "D1", "title": "t", "rank": 1})]
    ctx = json.loads(build_context_json("my question", docs))
    assert ctx["queries"][0]["query_meta"]["type"] == "original"
    assert ctx["queries"][0]["hits"][0]["doc_id"] == "D1"


def test_answer_generator_with_fake_llm():
    llm = FakeListChatModel(responses=["##Title##\n\nGrounded answer [doc_id=D1]"])
    gen = AnswerGenerator(llm)
    docs = [Document(page_content="c", metadata={"doc_id": "D1", "title": "t", "rank": 1})]
    result = gen.generate("question?", docs)
    assert "Grounded answer" in result["answer"]
    assert result["used_context"][0]["doc_id"] == "D1"


def test_parse_keyword_list_splits_and_dedups():
    out = parse_keyword_list("machine learning, vector-space, AI, ai", prefix="Keywords:")
    assert "machine" in out and "learning" in out and "vector" in out and "space" in out
    # case-insensitive dedup of AI/ai
    assert sum(1 for w in out if w.lower() == "ai") == 1
