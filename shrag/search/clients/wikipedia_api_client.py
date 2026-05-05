"""
Wikipedia API client and integration helper.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import requests

from shrag.search.base_client import BaseSearchClient
from shrag.search.resilient_caller import ResilientCaller
from shrag.utils.dedup import remove_duplicates


def _wiki_api_url(lang: str) -> str:
    lang_code = lang if lang in {"ko", "en"} else "en"
    return f"https://{lang_code}.wikipedia.org/w/api.php"


class WikipediaAPIClient(BaseSearchClient):
    source_name = "Wikipedia"

    def __init__(
        self,
        lang: str = "ko",
        *,
        caller: ResilientCaller | None = None,
    ):
        self.lang = lang
        self.base_url = _wiki_api_url(lang)
        self.rate_limit_delay = 0.1
        self.headers = {
            "User-Agent": "SearchMetaSystem/1.0 (contact: example@example.com)"
        }
        self.caller = caller or ResilientCaller()

    def search(self, search_terms: List[str], max_results: int) -> List[Dict[str, Any]]:
        per_term = max(1, min(10, max_results // max(1, len(search_terms))))
        docs = self.search_multiple_terms(search_terms, max_results_per_term=per_term)
        return docs[:max_results]

    def get_request_stats(self) -> dict[str, Any]:
        return self.caller.stats()

    def search_multiple_terms(
        self, search_terms: List[str], max_results_per_term: int = 10
    ) -> List[Dict[str, Any]]:
        all_documents: List[Dict[str, Any]] = []
        for term in search_terms:
            docs = self.caller.call(
                source=f"wikipedia.{self.lang}",
                term=term,
                cur_page=1,
                row_count=max_results_per_term,
                fields=["title", "snippet"],
                fetch=lambda t=term, n=max_results_per_term: self._search_single_term(t, max_results=n),
            )
            all_documents.extend(docs)
        return remove_duplicates(all_documents, key="title")

    def _search_single_term(self, term: str, max_results: int = 10) -> List[Dict[str, Any]]:
        try:
            search_params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": term,
                "srlimit": max_results,
                "srprop": "snippet|titlesnippet|size|wordcount|timestamp",
            }
            response = requests.get(self.base_url, params=search_params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            search_results = data.get("query", {}).get("search", [])
            documents = []
            for result in search_results:
                title = result.get("title", "")
                if title:
                    documents.append(self._get_page_info(title))
            return documents
        except Exception as e:
            logging.error("Wikipedia search failed: %s", e)
            return []

    def _get_page_info(self, title: str) -> Dict[str, Any]:
        domain = "ko" if self.lang == "ko" else "en"
        url = f"https://{domain}.wikipedia.org/wiki/{title.replace(' ', '_')}"
        return {
            "doc_id": title.replace(" ", "_"),
            "title": title,
            "abstract": f"Wikipedia article about {title}",
            "authors": "Wikipedia Contributors",
            "year": "",
            "url": url,
            "source": "Wikipedia",
        }


class WikipediaIntegration:
    def __init__(self, keyword_lang: str = "all"):
        lang = "ko" if keyword_lang == "korean" else "en"
        self.wikipedia_client = WikipediaAPIClient(lang=lang)
        logging.info("Wikipedia client initialized.")

    def search_with_wikipedia(
        self, query: str, search_terms: List[str], keywords: Dict[str, Any]
    ) -> Dict[str, Any]:
        start_time = time.time()
        try:
            documents = self.wikipedia_client.search(search_terms, max_results=40)
            return {
                "status": "success",
                "query": query,
                "keywords": keywords,
                "search_terms": search_terms,
                "documents": documents,
                "document_count": len(documents),
                "processing_time": time.time() - start_time,
                "sources": {"Wikipedia": len(documents)},
            }
        except Exception as e:
            logging.error("Wikipedia integration search failed: %s", e)
            return {
                "status": "error",
                "error_message": str(e),
                "query": query,
                "processing_time": time.time() - start_time,
            }
