"""Platform search clients: ScienceON / PubMed / Wikipedia.

Ported from ``shrag/search/clients/*`` + ``scienceon_adapter.py``. Each client
exposes ``search(search_terms, max_results) -> list[common-schema dict]`` with
keys: doc_id, title, abstract, source, url. ``acquire.py`` turns these into
LangChain Documents.

Two pragmatic improvements over the originals (both needed for a usable
keyless E2E): Wikipedia uses the real search snippet as the abstract (the
original stored a placeholder string), and PubMed works without an API key
(lower rate limit) instead of hard-failing.
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Protocol

import requests

from ..config import Settings
from .scienceon_client import ScienceONAPIClient  # ported low-level client

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_LAST_CALL: dict[str, float] = {}  # per-source last-request monotonic timestamp


def _polite_get(
    url: str,
    *,
    params: dict,
    headers: dict | None = None,
    timeout: int = 20,
    source: str = "",
    min_interval: float = 0.5,
    max_retries: int = 4,
) -> requests.Response:
    """GET with per-source throttling + retry on 429/5xx (exponential backoff).

    Lightweight stand-in for the original ``ResilientCaller`` — enough to keep
    the open Wikipedia/PubMed endpoints from rate-limiting us out.
    """
    resp = None
    for attempt in range(max_retries + 1):
        wait = min_interval - (time.monotonic() - _LAST_CALL.get(source, 0.0))
        if wait > 0:
            time.sleep(wait)
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        _LAST_CALL[source] = time.monotonic()
        if resp.status_code == 429 or resp.status_code >= 500:
            retry_after = resp.headers.get("Retry-After")
            sleep_s = (
                float(retry_after)
                if retry_after and retry_after.replace(".", "", 1).isdigit()
                else min(2.0**attempt, 30.0)
            )
            logger.warning("%s %s; retry %d/%d after %.1fs", source, resp.status_code, attempt + 1, max_retries, sleep_s)
            time.sleep(sleep_s)
            continue
        resp.raise_for_status()
        return resp
    assert resp is not None
    resp.raise_for_status()
    return resp


def _dedup(docs: list[dict], key: str = "doc_id") -> list[dict]:
    """Order-preserving dedup on ``key`` (case-insensitive), falling back to title."""
    seen: set[str] = set()
    out: list[dict] = []
    for doc in docs:
        norm = str(doc.get(key) or doc.get("title") or "").strip().lower()
        if not norm:
            out.append(doc)
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(doc)
    return out


class SearchClient(Protocol):
    source_name: str

    def search(self, search_terms: list[str], max_results: int) -> list[dict]: ...


# --------------------------------------------------------------------------- #
# Wikipedia
# --------------------------------------------------------------------------- #
class WikipediaClient:
    source_name = "Wikipedia"

    def __init__(self, lang: str = "en"):
        self.lang = lang if lang in {"ko", "en"} else "en"
        self.base_url = f"https://{self.lang}.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": "shrag_lc/0.1 (research)"}

    def search(self, search_terms: list[str], max_results: int) -> list[dict]:
        per_term = max(1, min(10, max_results // max(1, len(search_terms))))
        docs: list[dict] = []
        for term in search_terms:
            docs.extend(self._search_term(term, per_term))
        return _dedup(docs, key="title")[:max_results]

    def _search_term(self, term: str, limit: int) -> list[dict]:
        try:
            resp = _polite_get(
                self.base_url,
                params={
                    "action": "query",
                    "format": "json",
                    "list": "search",
                    "srsearch": term,
                    "srlimit": limit,
                    "srprop": "snippet",
                },
                headers=self.headers,
                source=f"wikipedia.{self.lang}",
                min_interval=0.5,
            )
            results = resp.json().get("query", {}).get("search", [])
        except Exception as e:  # noqa: BLE001
            logger.error("Wikipedia search failed (term=%s): %s", term, e)
            return []

        docs = []
        for r in results:
            title = r.get("title", "")
            if not title:
                continue
            snippet = _TAG_RE.sub("", r.get("snippet", "")).strip()
            docs.append(
                {
                    "doc_id": title.replace(" ", "_"),
                    "title": title,
                    "abstract": snippet or f"Wikipedia article: {title}",
                    "source": "Wikipedia",
                    "url": f"https://{self.lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                }
            )
        return docs


# --------------------------------------------------------------------------- #
# PubMed (E-utilities)
# --------------------------------------------------------------------------- #
class PubMedClient:
    source_name = "PubMed"

    def __init__(self, api_key: str = "", email: str = "", per_term: int = 10, max_terms: int = 10):
        self.api_key = api_key
        self.email = email
        self.per_term = per_term
        self.max_terms = max_terms
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def search(self, search_terms: list[str], max_results: int) -> list[dict]:
        terms = sorted(search_terms, key=len, reverse=True)[: self.max_terms]
        docs: list[dict] = []
        for term in terms:
            if len(docs) >= max_results:
                break
            docs.extend(self._search_and_fetch(term, self.per_term))
        return _dedup(docs, key="doc_id")[:max_results]

    def _params(self, extra: dict) -> dict:
        params = {"tool": "shrag_lc", **extra}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        return params

    def _search_and_fetch(self, term: str, retmax: int) -> list[dict]:
        try:
            formatted = self._format_term(term)
            interval = 0.1 if self.api_key else 0.34  # NCBI: 10/s with key, 3/s without
            sr = _polite_get(
                f"{self.base_url}/esearch.fcgi",
                params=self._params({"db": "pubmed", "term": formatted, "retmax": str(retmax)}),
                source="pubmed",
                min_interval=interval,
            )
            ids = [e.text for e in ET.fromstring(sr.text).findall(".//Id") if e.text]
            if not ids:
                return []
            fr = _polite_get(
                f"{self.base_url}/efetch.fcgi",
                params=self._params(
                    {"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "xml"}
                ),
                timeout=30,
                source="pubmed",
                min_interval=interval,
            )
            return self._parse_xml(fr.text)
        except Exception as e:  # noqa: BLE001
            logger.error("PubMed search/fetch failed (term=%s): %s", term, e)
            return []

    @staticmethod
    def _format_term(term: str) -> str:
        if "|" in term:
            kws = [kw.strip() for kw in term.split("|") if kw.strip()]
            return " OR ".join(f'"{kw}"' for kw in kws)
        return f'"{term}"'

    @staticmethod
    def _parse_xml(xml_content: str) -> list[dict]:
        docs: list[dict] = []
        try:
            root = ET.fromstring(xml_content)
        except Exception as e:  # noqa: BLE001
            logger.error("PubMed XML parse failed: %s", e)
            return docs
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None and pmid_el.text else ""
            title_el = article.find(".//ArticleTitle")
            title = title_el.text if title_el is not None and title_el.text else ""
            abstract = " ".join(
                el.text for el in article.findall(".//AbstractText") if el.text
            )
            docs.append(
                {
                    "doc_id": pmid,
                    "title": title or "",
                    "abstract": abstract,
                    "source": "PubMed",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )
        return docs


# --------------------------------------------------------------------------- #
# ScienceON
# --------------------------------------------------------------------------- #
_SCIENCEON_FIELDS = ["CN", "title", "abstract", "author", "year", "link"]
_SCIENCEON_ROW_COUNT = 10


class ScienceONClient:
    source_name = "ScienceON"

    def __init__(self, credentials_path: str | Path, max_pages: int = 5):
        self.client = ScienceONAPIClient(Path(credentials_path))
        self.max_pages = max_pages

    def search(self, search_terms: list[str], max_results: int) -> list[dict]:
        if not search_terms:
            return []
        docs: list[dict] = []
        for page in range(1, self.max_pages + 1):
            if len(docs) >= max_results:
                break
            page_docs = self._search_page(search_terms, page, max_results - len(docs))
            if not page_docs:
                break
            docs.extend(page_docs)
        return _dedup(docs, key="doc_id")[:max_results]

    def _search_page(self, terms: list[str], page: int, remaining: int) -> list[dict]:
        page_docs: list[dict] = []
        for term in terms:
            try:
                rows = self.client.search_articles(
                    query=term, cur_page=page, row_count=_SCIENCEON_ROW_COUNT, fields=_SCIENCEON_FIELDS
                ) or []
            except Exception as e:  # noqa: BLE001
                logger.error("ScienceON fetch failed (term=%s): %s", term, e)
                continue
            filtered = [self._to_common(r) for r in rows if self._is_quality(r)]
            page_docs.extend(filtered[: max(0, remaining - len(page_docs))])
            if len(page_docs) >= remaining:
                break
        return page_docs

    @staticmethod
    def _is_quality(row: dict) -> bool:
        title = str(row.get("title", "") or "").strip()
        if not title or len(title) < 5:
            return False
        abstract = str(row.get("abstract", "") or "").strip()
        if (not abstract or abstract == "없음") and len(title) < 20:
            return False
        return bool(str(row.get("CN", "") or "").strip())

    @staticmethod
    def _to_common(row: dict) -> dict:
        doc_id = str(row.get("CN", "") or "").strip()
        return {
            "doc_id": doc_id,
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "source": "ScienceON",
            "url": row.get("link", ""),
        }


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def create_search_client(source: str, settings: Settings, *, keyword_lang: str = "all") -> SearchClient:
    source = source.lower()
    if source == "wikipedia":
        return WikipediaClient(lang="ko" if keyword_lang == "korean" else "en")
    if source == "pubmed":
        api_key, email = settings.pubmed_api_key, settings.pubmed_email
        creds = Path(settings.pubmed_credentials_path)
        if not api_key and creds.exists():
            import json

            data = json.loads(creds.read_text(encoding="utf-8"))
            api_key, email = data.get("api_key", ""), data.get("email", "")
        return PubMedClient(api_key=api_key, email=email)
    if source == "scienceon":
        return ScienceONClient(settings.scienceon_credentials_path)
    raise ValueError(f"Unknown search source: {source!r}")
