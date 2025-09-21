"""
단일 쿼리 처리기 모듈
- 하나의 질문에 대한 전체 처리 파이프라인
- 키워드 추출 → 검색어 생성 → 문서 검색 → 결과 정리
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from core.keyword_extractor import KeywordExtractor
from core.document_searcher import DocumentSearcher

class SingleQueryProcessor:
    """단일 쿼리 처리기"""
    
    def __init__(self, keyword_extractor: KeywordExtractor, document_searcher: DocumentSearcher):
        """
        단일 쿼리 처리기 초기화
        
        Args:
            keyword_extractor: 키워드 추출기
            document_searcher: 문서 검색기
        """
        self.keyword_extractor = keyword_extractor
        self.document_searcher = document_searcher
        
    def process_query(self, query: str, target_documents: int = 50) -> Dict[str, Any]:
        """
        단일 쿼리 처리
        
        Args:
            query: 처리할 질문
            target_documents: 목표 문서 수
            
        Returns:
            처리 결과 딕셔너리
        """
        start_time = datetime.now()
        
        try:
            # 1. 키워드 추출
            keywords = self.keyword_extractor.extract_keywords(query)
            
            # 2. 검색어 생성
            search_terms = self.keyword_extractor.generate_search_terms(keywords)
            
            if not search_terms:
                logging.warning("검색어 생성 실패")
                return self._create_error_result(query, "검색어 생성 실패")
            
            # 3. 언어 감지 및 검색어 우선순위 설정
            language_priority = self._detect_language(query)
            prioritized_terms = self._prioritize_search_terms(search_terms, language_priority)
            
            # 4. 목표 문서 수 설정
            self.document_searcher.set_target_documents(target_documents)
            
            # 5. 문서 검색
            documents = self.document_searcher.search_documents(prioritized_terms, query)
            
            # 6. 결과 정리
            result = self._create_success_result(
                query=query,
                keywords=keywords,
                search_terms=prioritized_terms,
                documents=documents,
                processing_time=datetime.now() - start_time
            )
            
            # 핵심 정보만 표시
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
            print(f"   검색어 ({len(prioritized_terms)}개):")
            for i, term in enumerate(prioritized_terms[:10], 1):  # 처음 10개만 표시
                print(f"     {i}. {term}")
            if len(prioritized_terms) > 10:
                print(f"     ... 외 {len(prioritized_terms) - 10}개")
            
            print(f"   문서: {len(documents)}개")
            print()
            
            return result
            
        except Exception as e:
            logging.error(f"질문 처리 실패: {e}")
            return self._create_error_result(query, str(e))
    
    def _detect_language(self, query: str) -> str:
        """질문 언어 감지"""
        korean_chars = sum(1 for char in query if '\uac00' <= char <= '\ud7af')
        english_chars = sum(1 for char in query if char.isalpha() and ord(char) < 128)
        
        if korean_chars > english_chars:
            return "korean"
        else:
            return "english"
    
    def _prioritize_search_terms(self, search_terms: List[str], language: str) -> List[str]:
        """언어에 따른 검색어 우선순위 설정"""
        if language == "korean":
            # 한국어 검색어를 앞에 배치
            korean_terms = [term for term in search_terms if any('\uac00' <= char <= '\ud7af' for char in term)]
            other_terms = [term for term in search_terms if term not in korean_terms]
            return korean_terms + other_terms
        else:
            # 영어 검색어를 앞에 배치
            english_terms = [term for term in search_terms if not any('\uac00' <= char <= '\ud7af' for char in term)]
            other_terms = [term for term in search_terms if term not in english_terms]
            return english_terms + other_terms
    
    def _create_success_result(self, query: str, keywords: Dict[str, List[str]], 
                             search_terms: List[str], documents: List[Dict[str, Any]], 
                             processing_time) -> Dict[str, Any]:
        """성공 결과 생성 (기존 형식에 맞춤)"""
        return {
            "question": query,
            "status": "success",
            "keywords": keywords,
            "search_queries": search_terms,
            "search_terms": search_terms,  # main.py에서 사용
            "documents": documents,
            "total_documents_found": len(documents),
            "document_count": len(documents),
            "processing_time": processing_time.total_seconds(),  # 누락된 키 추가
            "search_timestamp": datetime.now().isoformat(),
            "execution_mode": "batch"
        }
    
    def _create_error_result(self, query: str, error_message: str) -> Dict[str, Any]:
        """에러 결과 생성"""
        return {
            "query": query,
            "status": "error",
            "error_message": error_message,
            "documents": [],
            "document_count": 0,
            "timestamp": datetime.now().isoformat()
        }
