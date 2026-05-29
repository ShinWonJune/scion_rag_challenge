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
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from .config import INSUFFICIENT_EN, INSUFFICIENT_KO, PipelineConfig, Settings
from .corpus import build_corpus, save_corpus_jsonl
from .embeddings import build_embeddings
from .generate import AnswerGenerator
from .llm import build_extractor_model, build_generation_model
from .rerank import build_reranker
from .schemas import (
    PipelineError,
    PipelineStatus,
    StageName,
    StageStatus,
    errors_to_dict,
    stage_report,
)
from .search.acquire import Acquirer
from .search.clients import create_search_client
from .search.keywords import KeywordExtractor
from .vectorstore import build_faiss, load_faiss_artifact, save_faiss_artifact

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
        client = create_search_client(cfg.source, self.settings, keyword_lang=cfg.keyword_lang, cfg=cfg)
        self.acquirer = Acquirer(keyword_extractor, client, cfg)

        self.embeddings = build_embeddings(cfg)
        self.reranker = build_reranker(cfg) if cfg.use_reranker else None
        self.generator = AnswerGenerator(build_generation_model(cfg, self.settings))
        self.graph = RunnableLambda(self._run_one_graph)
        self.last_run_context: dict = {}

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

    def _rerank_or_rank(self, query: str, retrieved: list[Document]) -> list[Document]:
        if self.reranker is not None:
            return list(self.reranker.compress_documents(retrieved, query))
        out: list[Document] = []
        for rank, doc in enumerate(retrieved[: self.cfg.rerank_top_n], start=1):
            out.append(Document(page_content=doc.page_content, metadata={**doc.metadata, "rank": rank}))
        return out

    def _insufficient_answer(self, query: str) -> str:
        return INSUFFICIENT_KO if _is_korean(query) else INSUFFICIENT_EN

    def _run_one_graph(self, item: dict) -> dict:
        qid = str(item.get("id", ""))
        query = item["question"]
        stages: dict[str, dict] = {}
        errors: list[PipelineError] = []
        warnings: list[str] = []

        try:
            documents, meta = self.acquirer.acquire(query)
            stages.update({str(k): v for k, v in (meta.get("stages") or {}).items()})
        except Exception as e:  # noqa: BLE001
            error = PipelineError.from_exception(
                StageName.SEARCH, e, question_id=qid, question=query, source=self.cfg.source
            )
            errors.append(error)
            stages[StageName.SEARCH] = stage_report(
                StageName.SEARCH, StageStatus.FAILED, errors=[error]
            ).to_dict()
            return {
                "id": qid,
                "question": query,
                "answer": "",
                "used_context": [],
                "source": self.cfg.source,
                "keywords": {"korean": [], "english": []},
                "search_terms": [],
                "acquired_count": 0,
                "status": PipelineStatus.FAILED,
                "stages": stages,
                "errors": errors_to_dict(errors),
                "warnings": warnings,
            }

        if not documents:
            result = {
                "id": qid,
                "question": query,
                "answer": self._insufficient_answer(query),
                "used_context": [],
                "source": meta["source"],
                "keywords": meta["keywords"],
                "search_terms": meta["search_terms"],
                "acquired_count": meta["document_count"],
                "status": PipelineStatus.NO_DOCUMENTS,
                "stages": stages,
                "errors": errors_to_dict(errors),
                "warnings": warnings,
            }
            return result

        try:
            store = build_faiss(documents, self.embeddings)
            retriever = store.as_retriever(search_kwargs={"k": self.cfg.dense_top_k})
            retrieved = retriever.invoke(query)
            stages[StageName.RETRIEVAL] = stage_report(
                StageName.RETRIEVAL,
                StageStatus.SUCCESS,
                counts={"retrieved_count": len(retrieved), "dense_top_k": self.cfg.dense_top_k},
                metadata={"index_mode": "per_query"},
            ).to_dict()
        except Exception as e:  # noqa: BLE001
            error = PipelineError.from_exception(StageName.RETRIEVAL, e, question_id=qid, question=query)
            errors.append(error)
            stages[StageName.RETRIEVAL] = stage_report(
                StageName.RETRIEVAL, StageStatus.FAILED, errors=[error]
            ).to_dict()
            return self._failed_result(qid, query, meta, stages, errors, warnings)

        try:
            reranked = self._rerank_or_rank(query, retrieved)
            stages[StageName.RERANK] = stage_report(
                StageName.RERANK,
                StageStatus.SUCCESS,
                counts={"reranked_count": len(reranked), "rerank_top_n": self.cfg.rerank_top_n},
            ).to_dict()
        except Exception as e:  # noqa: BLE001
            error = PipelineError.from_exception(StageName.RERANK, e, question_id=qid, question=query)
            errors.append(error)
            stages[StageName.RERANK] = stage_report(
                StageName.RERANK, StageStatus.FAILED, errors=[error]
            ).to_dict()
            return self._failed_result(qid, query, meta, stages, errors, warnings)

        try:
            result = self.generator.generate(query, reranked)
            stages[StageName.GENERATION] = stage_report(
                StageName.GENERATION, StageStatus.SUCCESS
            ).to_dict()
        except Exception as e:  # noqa: BLE001
            error = PipelineError.from_exception(StageName.GENERATION, e, question_id=qid, question=query)
            errors.append(error)
            stages[StageName.GENERATION] = stage_report(
                StageName.GENERATION, StageStatus.FAILED, errors=[error]
            ).to_dict()
            return self._failed_result(qid, query, meta, stages, errors, warnings)

        has_partial = any(report.get("status") == StageStatus.PARTIAL_SUCCESS for report in stages.values())
        result.update(
            {
                "id": qid,
                "source": meta["source"],
                "keywords": meta["keywords"],
                "search_terms": meta["search_terms"],
                "acquired_count": meta["document_count"],
                "status": PipelineStatus.PARTIAL_SUCCESS if has_partial else PipelineStatus.SUCCESS,
                "stages": stages,
                "errors": errors_to_dict(errors),
                "warnings": warnings,
            }
        )
        return result

    def _failed_result(
        self,
        qid: str,
        query: str,
        meta: dict,
        stages: dict[str, dict],
        errors: list[PipelineError],
        warnings: list[str],
    ) -> dict:
        return {
            "id": qid,
            "question": query,
            "answer": "",
            "used_context": [],
            "source": meta.get("source", self.cfg.source),
            "keywords": meta.get("keywords", {"korean": [], "english": []}),
            "search_terms": meta.get("search_terms", []),
            "acquired_count": meta.get("document_count", 0),
            "status": PipelineStatus.FAILED,
            "stages": stages,
            "errors": errors_to_dict(errors),
            "warnings": warnings,
        }

    def run_one(self, query: str, question_id: str | None = None) -> dict:
        return self.graph.invoke({"id": question_id or "", "question": query})

    def run(self, questions: list[dict]) -> list[dict]:
        """Run over [{'id', 'question'}, ...]; returns answer dicts."""
        results = []
        for i, item in enumerate(questions, start=1):
            qid = str(item.get("id", i))
            query = item["question"]
            logger.info("[%d/%d] %s", i, len(questions), query[:60])
            try:
                results.append(self.graph.invoke({"id": qid, "question": query}))
            except Exception as e:  # noqa: BLE001
                logger.error("Pipeline failed for qid=%s: %s", qid, e)
                error = PipelineError.from_exception(StageName.GENERATION, e, question_id=qid, question=query)
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
                        "status": PipelineStatus.FAILED,
                        "stages": {
                            StageName.GENERATION: stage_report(
                                StageName.GENERATION, StageStatus.FAILED, errors=[error]
                            ).to_dict()
                        },
                        "errors": [error.to_dict()],
                        "warnings": [],
                        "error": str(e),
                    }
                )
        return results

    def run_corpus_first(
        self,
        questions: list[dict],
        *,
        artifact_dir: str | Path | None = None,
        index_dir: str | Path | None = None,
        reuse_index: bool = False,
    ) -> list[dict]:
        """Acquire all documents first, build one shared FAISS index, then answer."""
        acquired: list[tuple[dict, list[Document], dict]] = []
        document_groups: list[list[Document]] = []
        for i, item in enumerate(questions, start=1):
            qid = str(item.get("id", i))
            query = item["question"]
            try:
                docs, meta = self.acquirer.acquire(query)
            except Exception as e:  # noqa: BLE001
                error = PipelineError.from_exception(
                    StageName.SEARCH, e, question_id=qid, question=query, source=self.cfg.source
                )
                meta = {
                    "source": self.cfg.source,
                    "keywords": {"korean": [], "english": []},
                    "search_terms": [],
                    "document_count": 0,
                    "stages": {
                        StageName.SEARCH: stage_report(
                            StageName.SEARCH, StageStatus.FAILED, errors=[error]
                        ).to_dict()
                    },
                }
                docs = []
            acquired.append(({"id": qid, "question": query}, docs, meta))
            document_groups.append(docs)

        artifact_path = Path(artifact_dir) if artifact_dir else None
        index_path = Path(index_dir) if index_dir else (artifact_path / "faiss_index" if artifact_path else None)

        index_metadata: dict = {}
        if reuse_index and index_path is not None and index_path.exists():
            store, index_metadata = load_faiss_artifact(index_path, self.embeddings, self.cfg)
            corpus_manifest = {"document_count": index_metadata.get("document_count", 0), "reused": True}
        else:
            corpus = build_corpus(document_groups)
            corpus_manifest = corpus.to_manifest()
            if artifact_path is not None:
                save_corpus_jsonl(corpus, artifact_path / "corpus.jsonl")
            if not corpus.documents:
                self.last_run_context = {
                    "pipeline_mode": "corpus_first",
                    "corpus": corpus_manifest,
                    "index": index_metadata,
                }
                return [
                    self._corpus_no_documents_result(item, meta)
                    for item, _docs, meta in acquired
                ]
            store = build_faiss(corpus.documents, self.embeddings)
            if index_path is not None:
                index_metadata = save_faiss_artifact(
                    store, index_path, self.cfg, corpus.document_count
                )
            else:
                index_metadata = {"document_count": corpus.document_count}

        retriever = store.as_retriever(search_kwargs={"k": self.cfg.dense_top_k})
        results: list[dict] = []
        for item, _docs, meta in acquired:
            results.append(self._answer_with_retriever(item, meta, retriever))

        self.last_run_context = {
            "pipeline_mode": "corpus_first",
            "corpus": corpus_manifest,
            "index": index_metadata,
        }
        return results

    def _corpus_no_documents_result(self, item: dict, meta: dict) -> dict:
        qid = str(item.get("id", ""))
        query = item["question"]
        stages = {str(k): v for k, v in (meta.get("stages") or {}).items()}
        stages[StageName.CORPUS_BUILD] = stage_report(
            StageName.CORPUS_BUILD, StageStatus.NO_DOCUMENTS, counts={"document_count": 0}
        ).to_dict()
        return {
            "id": qid,
            "question": query,
            "answer": self._insufficient_answer(query),
            "used_context": [],
            "source": meta.get("source", self.cfg.source),
            "keywords": meta.get("keywords", {"korean": [], "english": []}),
            "search_terms": meta.get("search_terms", []),
            "acquired_count": meta.get("document_count", 0),
            "status": PipelineStatus.NO_DOCUMENTS,
            "stages": stages,
            "errors": [],
            "warnings": [],
        }

    def _answer_with_retriever(self, item: dict, meta: dict, retriever) -> dict:
        qid = str(item.get("id", ""))
        query = item["question"]
        stages = {str(k): v for k, v in (meta.get("stages") or {}).items()}
        errors: list[PipelineError] = []
        warnings: list[str] = []
        try:
            retrieved = retriever.invoke(query)
            stages[StageName.RETRIEVAL] = stage_report(
                StageName.RETRIEVAL,
                StageStatus.SUCCESS,
                counts={"retrieved_count": len(retrieved), "dense_top_k": self.cfg.dense_top_k},
                metadata={"index_mode": "corpus_first"},
            ).to_dict()
            reranked = self._rerank_or_rank(query, retrieved)
            stages[StageName.RERANK] = stage_report(
                StageName.RERANK,
                StageStatus.SUCCESS,
                counts={"reranked_count": len(reranked), "rerank_top_n": self.cfg.rerank_top_n},
            ).to_dict()
            result = self.generator.generate(query, reranked)
            stages[StageName.GENERATION] = stage_report(
                StageName.GENERATION, StageStatus.SUCCESS
            ).to_dict()
        except Exception as e:  # noqa: BLE001
            error = PipelineError.from_exception(StageName.GENERATION, e, question_id=qid, question=query)
            errors.append(error)
            stages[StageName.GENERATION] = stage_report(
                StageName.GENERATION, StageStatus.FAILED, errors=[error]
            ).to_dict()
            return self._failed_result(qid, query, meta, stages, errors, warnings)

        has_partial = any(report.get("status") == StageStatus.PARTIAL_SUCCESS for report in stages.values())
        result.update(
            {
                "id": qid,
                "source": meta.get("source", self.cfg.source),
                "keywords": meta.get("keywords", {"korean": [], "english": []}),
                "search_terms": meta.get("search_terms", []),
                "acquired_count": meta.get("document_count", 0),
                "status": PipelineStatus.PARTIAL_SUCCESS if has_partial else PipelineStatus.SUCCESS,
                "stages": stages,
                "errors": errors_to_dict(errors),
                "warnings": warnings,
            }
        )
        return result
