"""
SearchMetaSystem orchestration class.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.search.scienceon_adapter import ScienceONAdapter
from src.search_pipeline.settings import Settings


class SearchMetaSystem:
    """Top-level integration for search pipeline."""

    def __init__(
        self,
        gemini_api_key: str | None = None,
        use_vllm: bool = False,
        vllm_base_url: str = "http://localhost:8000/v1",
        vllm_model: str = "openai/gpt-oss-120B",
        use_chatgpt: bool = False,
        chatgpt_model: str = "gpt-4o-mini",
        openai_api_key: str | None = None,
        keyword_lang: str = "all",
        pubmed_api_key: str | None = None,
        pubmed_email: str | None = None,
        pubmed_credentials_path: str = "configs/credentials/pubmed_api_credentials.json",
        scienceon_credentials_path: str = "configs/credentials/scienceon_api_credentials.json",
        skip_keyword_extraction: bool = False,
    ):
        self.settings = Settings()
        self.use_vllm = use_vllm
        self.use_chatgpt = use_chatgpt
        self.skip_keyword_extraction = skip_keyword_extraction
        self.keyword_lang = keyword_lang

        if use_vllm:
            self.settings.set("use_vllm", True)
            self.settings.set("vllm_base_url", vllm_base_url)
            self.settings.set("vllm_model", vllm_model)
        elif use_chatgpt:
            self.settings.set("use_chatgpt", True)
            self.settings.set("chatgpt_model", chatgpt_model)
            self.settings.set("openai_api_key", openai_api_key)
            self.settings.set("keyword_lang", keyword_lang)
        else:
            self.settings.set("gemini_api_key", gemini_api_key)
            self.settings.set("keyword_lang", keyword_lang)

        self._setup_logging()

        from src.search.clients.pubmed_api_client import PubMedIntegration
        from src.search.clients.scienceon_api_example import ScienceONAPIClient
        from src.search.clients.wikipedia_api_client import WikipediaIntegration

        self.scienceon_client = ScienceONAPIClient(Path(scienceon_credentials_path))
        self.scienceon_adapter = ScienceONAdapter(self.scienceon_client)
        self.pubmed_integration = PubMedIntegration(pubmed_credentials_path)
        self.wikipedia_integration = WikipediaIntegration(keyword_lang=self.keyword_lang)

        self._initialize_components()
        logging.info("SearchMetaSystem initialized.")

    def _setup_logging(self) -> None:
        log_level = self.settings.get("log_level", "INFO")
        log_format = self.settings.get("log_format")
        logging.basicConfig(level=getattr(logging, log_level.upper()), format=log_format)

    def _initialize_components(self) -> None:
        from src.search_pipeline.core.document_searcher import DocumentSearcher
        from src.search_pipeline.processors.batch_query_processor import BatchQueryProcessor
        from src.search_pipeline.processors.single_query_processor import SingleQueryProcessor
        from src.search_pipeline.utils.file_manager import FileManager
        from src.search_pipeline.utils.result_converter import ResultConverter

        self.file_manager = FileManager(output_dir=self.settings.get("output_directory"))
        self.result_converter = ResultConverter(self.file_manager)

        if self.use_vllm:
            from src.search_pipeline.core.vllm_keyword_extractor import VLLMKeywordExtractor

            self.keyword_extractor = VLLMKeywordExtractor(
                self.settings.get("vllm_base_url"), self.settings.get("vllm_model")
            )
        elif self.use_chatgpt:
            from src.search_pipeline.core.chatgpt_keyword_extractor import ChatGPTKeywordExtractor

            openai_key = self.settings.get("openai_api_key")
            if openai_key:
                self.keyword_extractor = ChatGPTKeywordExtractor(
                    api_key=openai_key,
                    model=self.settings.get("chatgpt_model"),
                    language=self.settings.get("keyword_lang"),
                )
            else:
                self.keyword_extractor = ChatGPTKeywordExtractor(
                    credentials_file="./configs/chatgpt_api_credentials.json",
                    model=self.settings.get("chatgpt_model"),
                    language=self.settings.get("keyword_lang"),
                )
        else:
            from src.search_pipeline.core.keyword_extractor import KeywordExtractor

            self.keyword_extractor = KeywordExtractor(
                api_key=self.settings.get("gemini_api_key"),
                language=self.settings.get("keyword_lang", "all"),
            )

        self.document_searcher = DocumentSearcher(self.scienceon_client)
        search_config = self.settings.get_search_config()
        self.document_searcher.set_target_documents(search_config["target_documents"])
        self.document_searcher.set_max_pages(search_config["max_pages"])

        self.single_processor = SingleQueryProcessor(
            self.keyword_extractor, self.document_searcher
        )
        self.batch_processor = BatchQueryProcessor(
            self.single_processor, self.file_manager, self.result_converter, self
        )

    def _build_search_terms(self, query: str) -> tuple[Dict[str, Any], List[str]]:
        if self.skip_keyword_extraction:
            return {"english": [query], "korean": []}, [query]
        keywords = self.keyword_extractor.extract_keywords(query)
        search_terms = self.keyword_extractor.generate_search_terms(keywords)
        return keywords, search_terms

    def process_single_query(
        self, query: str, target_documents: int | None = None
    ) -> Dict[str, Any]:
        if target_documents is None:
            target_documents = self.settings.get("target_documents_per_query")
        return self.single_processor.process_query(query, target_documents)

    def process_single_query_with_pubmed(
        self, query: str, use_pubmed: bool = True
    ) -> Dict[str, Any]:
        keywords, search_terms = self._build_search_terms(query)
        return self.pubmed_integration.search_with_pubmed(
            query=query,
            search_terms=search_terms,
            keywords=keywords,
            use_pubmed=use_pubmed,
        )

    def process_single_query_with_pubmed_simple(self, query: str) -> Dict[str, Any]:
        return self.pubmed_integration.search_with_pubmed_simple(query)

    def process_single_query_with_wikipedia(self, query: str) -> Dict[str, Any]:
        keywords, search_terms = self._build_search_terms(query)
        return self.wikipedia_integration.search_with_wikipedia(
            query=query, search_terms=search_terms, keywords=keywords
        )

    def process_batch_from_csv(
        self, csv_path: str, max_queries: Optional[int] = None, target_documents: int | None = None
    ) -> Dict[str, Any]:
        if target_documents is None:
            target_documents = self.settings.get("target_documents_per_query")
        return self.batch_processor.process_queries_from_csv(
            csv_path, target_documents, max_queries
        )

    def process_batch_queries(
        self, queries: List[str], target_documents: int | None = None
    ) -> Dict[str, Any]:
        if target_documents is None:
            target_documents = self.settings.get("target_documents_per_query")
        return self.batch_processor.process_queries(queries, target_documents)

    def convert_latest_results(self) -> Dict[str, str]:
        csv_file = self.result_converter.convert_latest_json_to_csv()
        jsonl_file = self.result_converter.convert_latest_json_to_jsonl()
        return {"csv": csv_file, "jsonl": jsonl_file}

    def update_settings(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            self.settings.set(key, value)
        self._update_components()

    def _update_components(self) -> None:
        from src.search_pipeline.utils.file_manager import FileManager
        from src.search_pipeline.utils.result_converter import ResultConverter

        search_config = self.settings.get_search_config()
        self.document_searcher.set_target_documents(search_config["target_documents"])
        self.document_searcher.set_max_pages(search_config["max_pages"])
        self.file_manager = FileManager(output_dir=self.settings.get("output_directory"))
        self.result_converter = ResultConverter(self.file_manager)

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "settings": self.settings.get_all(),
            "components": {
                "keyword_extractor": "initialized",
                "document_searcher": "initialized",
                "single_processor": "initialized",
                "batch_processor": "initialized",
                "file_manager": "initialized",
                "result_converter": "initialized",
            },
        }

    def print_system_info(self) -> None:
        print("SearchMetaSystem")
        print("=" * 40)
        self.settings.print_settings()

    def cleanup(self) -> None:
        logging.info("SearchMetaSystem cleanup complete.")
