"""
Wikipedia API 클라이언트
- Wikipedia Search API 사용
- 기존 ScienceON, PubMed 구조와 호환
- 키워드 기반 검색 지원
"""

import requests
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

class WikipediaAPIClient:
    """Wikipedia API 클라이언트"""
    
    def __init__(self):
        """
        Wikipedia API 클라이언트 초기화
        """
        self.base_url = "https://en.wikipedia.org/w/api.php"
        self.rate_limit_delay = 0.1  # Wikipedia API rate limiting
        self.headers = {
            'User-Agent': 'SearchMetaSystem/1.0 (https://github.com/example/searchmetasystem; contact@example.com)'
        }
        
    def search_multiple_terms(self, search_terms: List[str], max_results_per_term: int = 5) -> List[Dict[str, Any]]:
        """
        여러 검색어로 Wikipedia 검색
        
        Args:
            search_terms: 검색어 리스트
            max_results_per_term: 검색어당 최대 결과 수
            
        Returns:
            문서 리스트
        """
        all_documents = []
        
        for term in search_terms[:10]:  # 최대 10개 검색어로 제한
            try:
                documents = self._search_single_term(term, max_results_per_term)
                all_documents.extend(documents)
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logging.warning(f"Wikipedia 검색어 '{term}' 실패: {e}")
                continue
        
        # 중복 제거 (제목 기준)
        unique_documents = self._remove_duplicates_by_title(all_documents)
        
        return unique_documents
    
    def _search_single_term(self, term: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        단일 검색어로 Wikipedia 검색
        
        Args:
            term: 검색어
            max_results: 최대 결과 수
            
        Returns:
            문서 리스트
        """
        try:
            # Wikipedia Search API 사용
            search_params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': term,
                'srlimit': max_results,
                'srprop': 'snippet|titlesnippet|size|wordcount|timestamp'
            }
            
            response = requests.get(self.base_url, params=search_params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            search_results = data.get('query', {}).get('search', [])
            
            documents = []
            
            # 각 검색 결과에 대해 상세 정보 가져오기
            for result in search_results:
                try:
                    doc_info = self._get_page_info(result['title'])
                    if doc_info:
                        documents.append(doc_info)
                except Exception as e:
                    logging.warning(f"Wikipedia 페이지 '{result['title']}' 정보 가져오기 실패: {e}")
                    continue
            
            return documents
            
        except Exception as e:
            logging.error(f"Wikipedia 검색 실패: {e}")
            return []
    
    def _get_page_info(self, title: str) -> Dict[str, Any]:
        """
        Wikipedia 페이지 title 정보만 가져오기 (단순화)
        
        Args:
            title: 페이지 제목
            
        Returns:
            문서 정보 (title 위주)
        """
        try:
            # Wikipedia URL 생성
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            
            return {
                'title': title,
                'authors': 'Wikipedia Contributors',
                'abstract': f"Wikipedia article about {title}",
                'url': url,
                'publication_year': '',
                'journal': 'Wikipedia',
                'keywords': [title.lower()],
                'source': 'Wikipedia',
                'page_id': '',
                'last_modified': '',
                'word_count': 0
            }
            
        except Exception as e:
            logging.error(f"Wikipedia 페이지 정보 가져오기 실패: {e}")
            return None
    
    def _remove_duplicates_by_title(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """제목 기준 중복 문서 제거"""
        seen_titles = set()
        unique_documents = []
        
        for doc in documents:
            title = doc.get('title', '').lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_documents.append(doc)
        
        return unique_documents

class WikipediaIntegration:
    """SearchMetaSystem에 Wikipedia 기능 추가"""
    
    def __init__(self, search_meta_system):
        self.system = search_meta_system
        self.wikipedia_client = WikipediaAPIClient()
        logging.info("Wikipedia API 클라이언트 초기화 완료")
    
    def search_with_wikipedia(self, query: str) -> Dict[str, Any]:
        """
        Wikipedia 검색
        
        Args:
            query: 검색 질문
            
        Returns:
            검색 결과
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
            
            # 키워드와 검색어 로그 출력 (SingleQueryProcessor와 동일한 형식)
            print(f"✅ 질문: {query[:50]}...")
            
            # 키워드 표시
            korean_kw = keywords.get('korean', [])
            english_kw = keywords.get('english', [])
            print(f"   키워드 ({len(korean_kw + english_kw)}개):")
            if korean_kw:
                print(f"     한국어: {', '.join(korean_kw)}")
            if english_kw:
                print(f"     영어: {', '.join(english_kw)}")
            
            # 검색어 표시
            print(f"   검색어 ({len(search_terms)}개):")
            for i, term in enumerate(search_terms[:10], 1):  # 처음 10개만 표시
                print(f"     {i}. {term}")
            if len(search_terms) > 10:
                print(f"     ... 외 {len(search_terms) - 10}개")
            
            # Wikipedia 검색
            documents = self.wikipedia_client.search_multiple_terms(search_terms, max_results_per_term=4)
            
            print(f"   문서: {len(documents)}개")
            print()
            
            logging.info(f"Wikipedia 검색 완료: {len(documents)}개 문서")
            
            return {
                'status': 'success',
                'query': query,
                'keywords': keywords,
                'search_terms': search_terms,
                'documents': documents,
                'document_count': len(documents),
                'processing_time': time.time() - start_time,
                'sources': {
                    'Wikipedia': len(documents)
                }
            }
            
        except Exception as e:
            logging.error(f"Wikipedia 검색 실패: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'query': query,
                'processing_time': time.time() - start_time
            }