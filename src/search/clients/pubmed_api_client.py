"""
PubMed API client and integration helper.
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

import requests
from src.search.base_client import BaseSearchClient
from src.search.resilient_caller import ResilientCaller
from src.utils.dedup import remove_duplicates


class PubMedAPIClient(BaseSearchClient):
    """PubMed E-utilities API client."""

    source_name = "PubMed"

    def __init__(
        self,
        credentials_path: Path,
        *,
        caller: ResilientCaller | None = None,
        per_term_max_results: int = 10,
        max_terms: int = 10,
    ):
        self.credentials = self._load_credentials(credentials_path)
        self.api_key = self.credentials.get("api_key", "")
        self.email = self.credentials.get("email", "")
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.rate_limit_delay = 0.1
        self.caller = caller or ResilientCaller()
        self.per_term_max_results = per_term_max_results
        self.max_terms = max_terms

        if not self.api_key:
            raise ValueError("PubMed API key is not configured.")

    def _load_credentials(self, credentials_path: Path) -> Dict[str, str]:
        try:
            if not credentials_path.exists():
                logging.warning("PubMed credentials file not found: %s", credentials_path)
                return {}
            with open(credentials_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error("Failed to load PubMed credentials: %s", e)
            return {}

    def search(self, search_terms: List[str], max_results: int) -> List[Dict[str, Any]]:
        docs = self.search_multiple_terms(search_terms, max_terms=self.max_terms)
        return docs[:max_results]

    def get_request_stats(self) -> dict[str, Any]:
        return self.caller.stats()

    def search_multiple_terms(
        self, search_terms: List[str], max_terms: int = 10
    ) -> List[Dict[str, Any]]:
        try:
            sorted_terms = sorted(search_terms, key=len, reverse=True)[:max_terms]
            all_documents: List[Dict[str, Any]] = []

            for term in sorted_terms:
                docs = self.caller.call(
                    source="pubmed",
                    term=term,
                    cur_page=1,
                    row_count=self.per_term_max_results,
                    fields=["pmid", "title", "abstract", "authors", "year", "url"],
                    fetch=lambda t=term: self._search_and_fetch_term(t, self.per_term_max_results),
                )
                all_documents.extend(docs)

            unique = remove_duplicates(all_documents, key="id")
            return [self._to_common_schema(d) for d in unique]
        except Exception as e:
            logging.error("PubMed multi-term search failed: %s", e)
            return []

    def search_single_term_test(
        self, search_term: str, max_results: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            docs = self._search_and_fetch_term(search_term, max_results=max_results)
            return [self._to_common_schema(d) for d in docs]
        except Exception as e:
            logging.error("PubMed single-term search failed: %s", e)
            return []

    def _search_and_fetch_term(
        self, term: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        try:
            formatted_term = self._format_search_term(term)

            search_params = {
                "db": "pubmed",
                "term": formatted_term,
                "retmax": str(max_results),
                "api_key": self.api_key,
            }
            if self.email:
                search_params["email"] = self.email

            search_response = requests.get(
                f"{self.base_url}/esearch.fcgi", params=search_params
            )
            search_response.raise_for_status()

            root = ET.fromstring(search_response.text)
            id_list = [id_elem.text for id_elem in root.findall(".//Id") if id_elem.text]
            if not id_list:
                return []

            time.sleep(self.rate_limit_delay)

            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "rettype": "abstract",
                "retmode": "xml",
                "api_key": self.api_key,
            }
            if self.email:
                fetch_params["email"] = self.email

            fetch_response = requests.get(
                f"{self.base_url}/efetch.fcgi", params=fetch_params
            )
            fetch_response.raise_for_status()
            return self._parse_pubmed_xml(fetch_response.text)
        except Exception as e:
            logging.error("PubMed search/fetch failed for term '%s': %s", term, e)
            return []

    def _format_search_term(self, term: str) -> str:
        if "|" in term:
            keywords = [kw.strip() for kw in term.split("|")]
            return " OR ".join(f'"{kw}"' for kw in keywords if kw)
        return f'"{term}"'

    def _parse_pubmed_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(xml_content)
            for article in root.findall(".//PubmedArticle"):
                try:
                    pmid_elem = article.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None and pmid_elem.text else ""

                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None and title_elem.text else ""

                    abstract_texts = article.findall(".//AbstractText")
                    abstract = " ".join(elem.text for elem in abstract_texts if elem.text)

                    authors = []
                    for author in article.findall(".//Author"):
                        last_name = author.find(".//LastName")
                        first_name = author.find(".//ForeName")
                        if (
                            last_name is not None
                            and last_name.text
                            and first_name is not None
                            and first_name.text
                        ):
                            authors.append(f"{first_name.text} {last_name.text}")

                    journal_elem = article.find(".//Journal/Title")
                    journal = (
                        journal_elem.text
                        if journal_elem is not None and journal_elem.text
                        else ""
                    )

                    pub_date_elem = article.find(".//PubDate/Year")
                    pub_year = (
                        pub_date_elem.text
                        if pub_date_elem is not None and pub_date_elem.text
                        else ""
                    )

                    documents.append(
                        {
                            "id": pmid,
                            "title": title or "",
                            "abstract": abstract,
                            "authors": ", ".join(authors),
                            "journal": journal,
                            "publication_year": pub_year,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            "source": "PubMed",
                            "content": f"{title}. {abstract}",
                            "score": 1.0,
                        }
                    )
                except Exception as e:
                    logging.warning("PubMed article parse failed: %s", e)
        except Exception as e:
            logging.error("PubMed XML parse failed: %s", e)
        return documents

    def _to_common_schema(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "doc_id": doc.get("id", ""),
            "title": doc.get("title", ""),
            "abstract": doc.get("abstract", ""),
            "authors": doc.get("authors", ""),
            "year": doc.get("publication_year", ""),
            "url": doc.get("url", ""),
            "source": "PubMed",
        }


class PubMedIntegration:
    """Integration layer used by SearchMetaSystem."""

    def __init__(self, pubmed_credentials_path: str = "configs/credentials/pubmed_api_credentials.json"):
        self.pubmed_client: PubMedAPIClient | None = None
        self.credentials_path = Path(pubmed_credentials_path)
        self._initialize_pubmed_client()

    def _initialize_pubmed_client(self) -> None:
        try:
            if self.credentials_path.exists():
                self.pubmed_client = PubMedAPIClient(self.credentials_path)
                logging.info("PubMed client initialized.")
            else:
                logging.warning("PubMed credentials file not found: %s", self.credentials_path)
        except Exception as e:
            logging.error("PubMed client initialization failed: %s", e)
            self.pubmed_client = None

    def search_with_pubmed_simple(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            if not self.pubmed_client:
                return {
                    "status": "error",
                    "error_message": "PubMed client is not initialized.",
                    "query": query,
                }

            search_term = query
            documents = self.pubmed_client.search_single_term_test(search_term, max_results=5)
            return {
                "status": "success",
                "query": query,
                "keywords": [search_term],
                "search_terms": [search_term],
                "documents": documents,
                "document_count": len(documents),
                "processing_time": time.time() - start_time,
                "sources": {"PubMed": len(documents)},
            }
        except Exception as e:
            logging.error("Simple PubMed search failed: %s", e)
            return {"status": "error", "error_message": str(e), "query": query}

    def search_with_pubmed(
        self,
        query: str,
        search_terms: List[str],
        keywords: Dict[str, Any],
        use_pubmed: bool = True,
    ) -> Dict[str, Any]:
        start_time = time.time()
        try:
            all_documents: List[Dict[str, Any]] = []
            if use_pubmed and self.pubmed_client:
                try:
                    pubmed_docs = self.pubmed_client.search_multiple_terms(search_terms)
                    all_documents.extend(pubmed_docs)
                except Exception as e:
                    logging.warning("PubMed search failed: %s", e)

            unique_documents = remove_duplicates(all_documents, key="title")
            return {
                "status": "success",
                "query": query,
                "keywords": keywords,
                "search_terms": search_terms,
                "documents": unique_documents,
                "document_count": len(unique_documents),
                "processing_time": time.time() - start_time,
                "sources": {
                    "PubMed": len(
                        [d for d in unique_documents if d.get("source") == "PubMed"]
                    )
                },
            }
        except Exception as e:
            logging.error("Integrated PubMed search failed: %s", e)
            return {"status": "error", "error_message": str(e), "query": query}
