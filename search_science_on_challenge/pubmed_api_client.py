"""
PubMed API 클라이언트
- E-utilities API 사용
- History Server 활용으로 rate limit 최적화
- 기존 ScienceON 구조와 호환
- 검색어 당 10개 문서 검색 제한
"""

import requests
import time
import xml.etree.ElementTree as ET
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

class PubMedAPIClient:
    """PubMed E-utilities API 클라이언트"""
    
    def __init__(self, credentials_path: Path):
        """
        PubMed API 클라이언트 초기화
        
        Args:
            credentials_path: PubMed API 자격증명 파일 경로
        """
        self.credentials = self._load_credentials(credentials_path)
        self.api_key = self.credentials.get('api_key', '')
        self.email = self.credentials.get('email', '')
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.rate_limit_delay = 0.1  # 초당 10회 제한
        
        # History Server 상태
        self.web_env = None
        self.query_keys = []
        
        if not self.api_key:
            raise ValueError("PubMed API 키가 설정되지 않았습니다.")
    
    def _load_credentials(self, credentials_path: Path) -> Dict[str, str]:
        """PubMed API 자격증명 로드"""
        try:
            if not credentials_path.exists():
                logging.warning(f"PubMed 자격증명 파일을 찾을 수 없습니다: {credentials_path}")
                return {}
            
            with open(credentials_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logging.error(f"PubMed 자격증명 로드 실패: {e}")
            return {}
        
    def search_multiple_terms(self, search_terms: List[str], max_terms: int = 10) -> List[Dict[str, Any]]:
        """
        여러 검색어로 PubMed 검색 (직접 검색 방식)
        
        Args:
            search_terms: 검색어 리스트
            max_terms: 최대 검색어 수
            
        Returns:
            문서 리스트 (ScienceON 호환 형식)
        """
        try:
            # 검색어를 길이 순으로 정렬하여 상위 N개만 선택
            sorted_terms = sorted(search_terms, key=len, reverse=True)[:max_terms]
            
            logging.info(f"PubMed 검색: {len(sorted_terms)}개 검색어 처리 (직접 검색)")
            
            all_documents = []
            
            # 각 검색어별로 개별 검색 및 fetch
            for i, term in enumerate(sorted_terms):
                try:
                    documents = self._search_and_fetch_term(term, max_results=10)
                    all_documents.extend(documents)
                    
                    logging.info(f"검색어 '{term}': {len(documents)}개 문서")
                    
                    # Rate limit 준수 (1초당 10회 제한)
                    time.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logging.warning(f"검색어 '{term}' 실패: {e}")
                    continue
            
            # 중복 제거 (ID 기준)
            unique_documents = self._remove_duplicates_by_id(all_documents)
            
            logging.info(f"PubMed 검색 완료: {len(unique_documents)}개 문서 (중복 제거 후)")
            return unique_documents
            
        except Exception as e:
            logging.error(f"PubMed 검색 실패: {e}")
            return []
    
    def _search_and_fetch_term(self, term: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        단일 검색어로 검색 및 즉시 fetch
        
        Args:
            term: 검색어
            max_results: 최대 결과 수
            
        Returns:
            문서 리스트
        """
        try:
            # 1단계: ESearch로 ID 목록 가져오기
            formatted_term = self._format_search_term(term)
            
            search_params = {
                'db': 'pubmed',
                'term': formatted_term,
                'retmax': str(max_results),
                'api_key': self.api_key
            }
            
            if self.email:
                search_params['email'] = self.email
            
            search_response = requests.get(f"{self.base_url}/esearch.fcgi", params=search_params)
            search_response.raise_for_status()
            
            # XML 파싱하여 ID 목록 추출
            root = ET.fromstring(search_response.text)
            id_list = [id_elem.text for id_elem in root.findall('.//Id')]
            
            if not id_list:
                return []
            
            # Rate limit 준수
            time.sleep(self.rate_limit_delay)
            
            # 2단계: EFetch로 상세 정보 가져오기
            id_string = ','.join(id_list)
            
            fetch_params = {
                'db': 'pubmed',
                'id': id_string,
                'rettype': 'abstract',
                'retmode': 'xml',
                'api_key': self.api_key
            }
            
            if self.email:
                fetch_params['email'] = self.email
            
            fetch_response = requests.get(f"{self.base_url}/efetch.fcgi", params=fetch_params)
            fetch_response.raise_for_status()
            
            # XML에서 문서 정보 추출
            documents = self._parse_pubmed_xml(fetch_response.text)
            
            return documents
            
        except Exception as e:
            logging.error(f"검색어 '{term}' 처리 실패: {e}")
            return []
    
    def _remove_duplicates_by_id(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ID 기준 중복 문서 제거"""
        seen_ids = set()
        unique_docs = []
        
        for doc in documents:
            doc_id = doc.get('id', '')
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
        
        return unique_docs
    
    def search_single_term_test(self, search_term: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        단일 검색어 테스트용 메서드 (새로운 직접 검색 방식 사용)
        
        Args:
            search_term: 단일 검색어
            max_results: 최대 결과 수
            
        Returns:
            문서 리스트
        """
        try:
            logging.info(f"PubMed 단일 검색어 테스트: '{search_term}', 최대 {max_results}개")
            
            # 새로운 직접 검색 방식 사용
            documents = self._search_and_fetch_term(search_term, max_results)
            
            logging.info(f"PubMed 단일 검색 완료: {len(documents)}개 문서")
            return documents
            
        except Exception as e:
            logging.error(f"PubMed 단일 검색 실패: {e}")
            return []
    
    # === 기존 History Server 방식 (사용 안 함) ===
        """
        단일 검색어 테스트용 메서드
        
        Args:
            search_term: 단일 검색어
            max_results: 최대 결과 수
            
        Returns:
            문서 리스트
        """
        try:
            logging.info(f"PubMed 단일 검색어 테스트: '{search_term}', 최대 {max_results}개")
            
            # 단일 검색어로 ESearch 실행
            formatted_term = self._format_search_term(search_term)
            
            params = {
                'db': 'pubmed',
                'term': formatted_term,
                'retmax': str(max_results),
                'api_key': self.api_key
            }
            
            if self.email:
                params['email'] = self.email
            
            response = requests.get(f"{self.base_url}/esearch.fcgi", params=params)
            response.raise_for_status()
            
            # XML 파싱하여 ID 목록 추출
            root = ET.fromstring(response.text)
            id_list = [id_elem.text for id_elem in root.findall('.//Id')]
            
            if not id_list:
                logging.warning("검색 결과가 없습니다.")
                return []
            
            logging.info(f"검색된 ID 수: {len(id_list)}")
            
            # EFetch로 상세 정보 가져오기
            id_string = ','.join(id_list)
            
            fetch_params = {
                'db': 'pubmed',
                'id': id_string,
                'rettype': 'abstract',
                'retmode': 'xml',
                'api_key': self.api_key
            }
            
            if self.email:
                fetch_params['email'] = self.email
            
            time.sleep(self.rate_limit_delay)
            
            fetch_response = requests.get(f"{self.base_url}/efetch.fcgi", params=fetch_params)
            fetch_response.raise_for_status()
            
            # XML에서 문서 정보 추출
            documents = self._parse_pubmed_xml(fetch_response.text)
            
            logging.info(f"PubMed 단일 검색 완료: {len(documents)}개 문서")
            return documents
            
        except Exception as e:
            logging.error(f"PubMed 단일 검색 실패: {e}")
            return []
    
    def _execute_searches(self, search_terms: List[str]):
        """여러 검색어로 ESearch 실행"""
        self.query_keys = []
        
        for i, term in enumerate(search_terms):
            try:
                # PubMed 검색어 형식으로 변환
                formatted_term = self._format_search_term(term)
                
                params = {
                    'db': 'pubmed',
                    'term': formatted_term,
                    'usehistory': 'y',
                    'retmax': '10',  # 각 검색어당 최대 10개로 줄임 (원래 10개)
                    'api_key': self.api_key
                }
                
                if self.email:
                    params['email'] = self.email
                
                response = requests.get(f"{self.base_url}/esearch.fcgi", params=params)
                response.raise_for_status()
                
                # XML 파싱
                root = ET.fromstring(response.text)
                
                # 첫 번째 검색에서 WebEnv 저장
                if i == 0:
                    web_env_elem = root.find('.//WebEnv')
                    if web_env_elem is not None:
                        self.web_env = web_env_elem.text
                
                # Query Key 저장
                query_key_elem = root.find('.//QueryKey')
                if query_key_elem is not None:
                    self.query_keys.append(query_key_elem.text)
                
                # Rate limit 준수
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logging.warning(f"검색어 '{term}' 실패: {e}")
                continue
    
    def _fetch_results(self) -> List[Dict[str, Any]]:
        """History Server에서 결과 Fetch"""
        all_documents = []
        
        if not self.web_env or not self.query_keys:
            return all_documents
        
        try:
            # 모든 query_key를 합쳐서 한 번에 Fetch
            for query_key in self.query_keys:
                params = {
                    'db': 'pubmed',
                    'WebEnv': self.web_env,
                    'query_key': query_key,
                    'rettype': 'abstract',
                    'retmode': 'xml',
                    'api_key': self.api_key
                }
                
                if self.email:
                    params['email'] = self.email
                
                response = requests.get(f"{self.base_url}/efetch.fcgi", params=params)
                response.raise_for_status()
                
                # XML에서 문서 정보 추출
                documents = self._parse_pubmed_xml(response.text)
                all_documents.extend(documents)
                
                # Rate limit 준수
                time.sleep(self.rate_limit_delay)
                
        except Exception as e:
            logging.error(f"PubMed Fetch 실패: {e}")
        
        return all_documents
    
    def _format_search_term(self, term: str) -> str:
        """검색어를 PubMed 형식으로 변환"""
        # 파이프(|)를 OR로 변환
        if '|' in term:
            keywords = [kw.strip() for kw in term.split('|')]
            return ' OR '.join(f'"{kw}"' for kw in keywords if kw)
        else:
            return f'"{term}"'
    
    def _parse_pubmed_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """PubMed XML을 ScienceON 호환 형식으로 변환"""
        documents = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    # PMID - None 체크 강화
                    pmid_elem = article.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None and pmid_elem.text else ""
                    
                    # 제목 - None 체크 강화
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None and title_elem.text else ""
                    
                    # 초록 - None 체크 강화
                    abstract_texts = article.findall('.//AbstractText')
                    abstract_parts = []
                    for elem in abstract_texts:
                        if elem.text:
                            abstract_parts.append(elem.text)
                    abstract = ' '.join(abstract_parts)
                    
                    # 저자 - None 체크 강화
                    authors = []
                    for author in article.findall('.//Author'):
                        last_name = author.find('.//LastName')
                        first_name = author.find('.//ForeName')
                        if (last_name is not None and last_name.text and 
                            first_name is not None and first_name.text):
                            authors.append(f"{first_name.text} {last_name.text}")
                    
                    # 저널 - None 체크 강화
                    journal_elem = article.find('.//Journal/Title')
                    journal = journal_elem.text if journal_elem is not None and journal_elem.text else ""
                    
                    # 출판일 - None 체크 강화
                    pub_date_elem = article.find('.//PubDate/Year')
                    pub_year = pub_date_elem.text if pub_date_elem is not None and pub_date_elem.text else ""
                    
                    # ScienceON 호환 형식으로 변환
                    document = {
                        'id': pmid,
                        'title': title or "",  # 빈 문자열 보장
                        'abstract': abstract,
                        'authors': ', '.join(authors),
                        'journal': journal,
                        'publication_year': pub_year,
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        'source': 'PubMed',
                        'content': f"{title}. {abstract}",  # 검색용 통합 텍스트
                        'score': 1.0  # 기본 점수
                    }
                    
                    documents.append(document)
                    
                except Exception as e:
                    logging.warning(f"PubMed 문서 파싱 실패: {e}")
                    continue
                    
        except Exception as e:
            logging.error(f"PubMed XML 파싱 실패: {e}")
        
        return documents

# SearchMetaSystem에 추가할 메서드들
class PubMedIntegration:
    """기존 SearchMetaSystem에 PubMed 기능 추가"""
    
    def __init__(self, search_meta_system, pubmed_credentials_path: str = "./configs/pubmed_api_credentials.json"):
        self.system = search_meta_system
        self.pubmed_client = None
        self.credentials_path = Path(pubmed_credentials_path)
        
        # PubMed 클라이언트 자동 초기화
        self._initialize_pubmed_client()
        
    def _initialize_pubmed_client(self):
        """PubMed API 클라이언트 초기화"""
        try:
            if self.credentials_path.exists():
                self.pubmed_client = PubMedAPIClient(self.credentials_path)
                logging.info("PubMed API 클라이언트 초기화 완료")
            else:
                logging.warning(f"PubMed 자격증명 파일이 없습니다: {self.credentials_path}")
        except Exception as e:
            logging.error(f"PubMed 클라이언트 초기화 실패: {e}")
            self.pubmed_client = None
    
    def search_with_pubmed_simple(self, query: str) -> Dict[str, Any]:
        """
        PubMed 간단 검색 (테스트용)
        
        Args:
            query: 검색 질문
            
        Returns:
            검색 결과
        """
        start_time = time.time()
        try:
            if not self.pubmed_client:
                return {
                    'status': 'error',
                    'error_message': 'PubMed 클라이언트가 초기화되지 않았습니다.',
                    'query': query
                }
            
            # 키워드 추출 또는 직접 질문 사용
            if hasattr(self.system, 'skip_keyword_extraction') and self.system.skip_keyword_extraction:
                # 키워드 추출을 건너뛰고 직접 질문 사용
                search_term = query
                logging.info(f"키워드 추출 생략, 직접 질문 사용: {query}")
            else:
                # 단일 검색어로 간단 검색
                search_term = query
            
            documents = self.pubmed_client.search_single_term_test(search_term, max_results=5)
            
            return {
                'status': 'success',
                'query': query,
                'keywords': [search_term],
                'search_terms': [search_term],
                'documents': documents,
                'document_count': len(documents),
                'processing_time': time.time() - start_time,
                'sources': {
                    'PubMed': len(documents)
                }
            }
            
        except Exception as e:
            logging.error(f"PubMed 간단 검색 실패: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'query': query
            }
    
    def search_with_pubmed(self, query: str, use_pubmed: bool = True) -> Dict[str, Any]:
        """
        PubMed 포함 통합 검색
        
        Args:
            query: 검색 질문
            use_pubmed: PubMed 검색 사용 여부
            use_scienceon: ScienceON 검색 사용 여부
            
        Returns:
            통합 검색 결과
        """
        start_time = time.time()
        try:
            # 키워드 추출 또는 직접 질문 사용
            if hasattr(self.system, 'skip_keyword_extraction') and self.system.skip_keyword_extraction:
                # 키워드 추출을 건너뛰고 직접 질문 사용
                keywords = {"english": [query], "korean": []}
                search_terms = [query]
                logging.info(f"키워드 추출 생략, 직접 질문 사용: {query}")
            else:
                # 기존 키워드 추출
                keywords = self.system.keyword_extractor.extract_keywords(query)
                search_terms = self.system.keyword_extractor.generate_search_terms(keywords)
            
            all_documents = []

            
            # PubMed 검색 (추가)
            if use_pubmed and self.pubmed_client:
                try:
                    pubmed_docs = self.pubmed_client.search_multiple_terms(search_terms)
                    all_documents.extend(pubmed_docs)
                    logging.info(f"PubMed 검색: {len(pubmed_docs)}개 문서")
                except Exception as e:
                    logging.warning(f"PubMed 검색 실패: {e}")
            
            # 중복 제거 (제목 기준)
            unique_documents = self._remove_duplicates(all_documents)
            
            return {
                'status': 'success',
                'query': query,
                'keywords': keywords,
                'search_terms': search_terms,
                'documents': unique_documents,
                'document_count': len(unique_documents),
                'processing_time': time.time() - start_time,
                'sources': {
                    'PubMed': len([d for d in unique_documents if d.get('source') == 'PubMed'])
                }
            }
            
        except Exception as e:
            logging.error(f"통합 검색 실패: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'query': query
            }
    
    def _remove_duplicates(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """제목 기준 중복 문서 제거"""
        seen_titles = set()
        unique_docs = []
        
        for doc in documents:
            title = doc.get('title', '')
            # None 체크 추가
            if title is None:
                title = ''
            title = title.strip().lower()
            
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)
        
        return unique_docs