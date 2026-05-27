"""End-to-end SHRAG pipeline (LangChain).

Per query: acquire corpus (search) -> embed + FAISS -> dense top-k ->
cross-encoder rerank top-n -> grounded answer. Heavy models (embeddings,
reranker, chat models) are built once and reused across queries; the FAISS
store is rebuilt per query because acquisition is per-query (as in the
original pipeline).
"""

from __future__ import annotations

import logging
import re

from langchain_core.documents import Document

from .config import INSUFFICIENT_EN, INSUFFICIENT_KO, PipelineConfig, Settings
from .embeddings import build_embeddings
from .generate import AnswerGenerator
from .llm import build_extractor_model, build_generation_model
from .rerank import build_reranker
from .search.acquire import Acquirer
from .search.clients import create_search_client
from .search.keywords import KeywordExtractor
from .vectorstore import build_faiss

logger = logging.getLogger(__name__)
_HANGUL_RE = re.compile(r"[가-힣]")


def _is_korean(text: str) -> bool:
    return bool(_HANGUL_RE.search(text or ""))


class SHRAGPipeline:
    def __init__(self, cfg: PipelineConfig, settings: Settings | None = None):
        self.cfg = cfg
        self.settings = settings or Settings()

        extractor_llm = build_extractor_model(cfg, self.settings)
        keyword_extractor = KeywordExtractor(extractor_llm, language=cfg.keyword_lang)
        client = create_search_client(cfg.source, self.settings, keyword_lang=cfg.keyword_lang)
        self.acquirer = Acquirer(keyword_extractor, client, cfg)

        self.embeddings = build_embeddings(cfg)
        self.reranker = build_reranker(cfg) if cfg.use_reranker else None
        self.generator = AnswerGenerator(build_generation_model(cfg, self.settings))

    def _retrieve(self, query: str, docs: list[Document]) -> list[Document]:
        store = build_faiss(docs, self.embeddings)
        retriever = store.as_retriever(search_kwargs={"k": self.cfg.dense_top_k})
        retrieved = retriever.invoke(query)
        if self.reranker is not None:
            return list(self.reranker.compress_documents(retrieved, query))
        # No reranker: assign ranks to the dense order.
        out: list[Document] = []
        for rank, doc in enumerate(retrieved[: self.cfg.rerank_top_n], start=1):
            out.append(Document(page_content=doc.page_content, metadata={**doc.metadata, "rank": rank}))
        return out

    def run_one(self, query: str, question_id: str | None = None) -> dict:
        documents, meta = self.acquirer.acquire(query)
        if not documents:
            insufficient = INSUFFICIENT_KO if _is_korean(query) else INSUFFICIENT_EN
            result = {"question": query, "answer": insufficient, "used_context": []}
        else:
            reranked = self._retrieve(query, documents)
            result = self.generator.generate(query, reranked)
        result["id"] = question_id
        result["source"] = meta["source"]
        result["keywords"] = meta["keywords"]
        result["search_terms"] = meta["search_terms"]
        result["acquired_count"] = meta["document_count"]
        return result

    def run(self, questions: list[dict]) -> list[dict]:
        """Run over [{'id', 'question'}, ...]; returns answer dicts."""
        results = []
        for i, item in enumerate(questions, start=1):
            qid = str(item.get("id", i))
            query = item["question"]
            logger.info("[%d/%d] %s", i, len(questions), query[:60])
            try:
                results.append(self.run_one(query, question_id=qid))
            except Exception as e:  # noqa: BLE001
                logger.error("Pipeline failed for qid=%s: %s", qid, e)
                results.append(
                    {
                        "id": qid,
                        "question": query,
                        "answer": "",
                        "used_context": [],
                        "source": self.cfg.source,
                        "keywords": {"korean": [], "english": []},
                        "search_terms": [],
                        "acquired_count": 0,
                        "error": str(e),
                    }
                )
        return results
