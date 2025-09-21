"""
문서 검색기 모듈
- ScienceON API를 통한 논문 검색
- 50개 문서 확보까지 반복 검색
- 품질 필터링
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

class DocumentSearcher:
    """ScienceON API를 사용한 문서 검색기"""
    
    def __init__(self, scienceon_client):
        """
        문서 검색기 초기화
        
        Args:
            scienceon_client: ScienceON API 클라이언트
        """
        self.scienceon_client = scienceon_client
        self.target_documents = 50  # 목표 문서 수
        self.max_pages = 5  # 최대 검색 페이지 수
        
    def search_documents(self, search_terms: List[str], query: str) -> List[Dict[str, Any]]:
        """
        검색어로 문서 검색 (50개 확보까지 반복)
        
        Args:
            search_terms: 검색어 리스트
            query: 원본 질문
            
        Returns:
            검색된 문서 리스트
        """
        all_documents = []
        used_terms = set()
        print(f"DEBUG - 검색어 리스트: {search_terms}")
        for page in range(1, self.max_pages + 1):
            if len(all_documents) >= self.target_documents:
                break
                
            page_documents = self._search_page(search_terms, page, used_terms)
            print(f"DEBUG - {page}페이지에서 {len(page_documents)}개 문서 검색됨")
            if not page_documents:
                break
            
            all_documents.extend(page_documents)
        
        # 중복 제거
        unique_documents = self._remove_duplicates(all_documents)
        
        # 품질 필터링
        filtered_documents = self._filter_quality(unique_documents)
        
        return filtered_documents[:self.target_documents]
    
    def _search_page(self, search_terms: List[str], page: int, used_terms: set) -> List[Dict[str, Any]]:
        """특정 페이지에서 검색 수행"""
        page_documents = []
        
        for term in search_terms:
            if term in used_terms:
                continue
                
            try:
                # ScienceON API 검색
                results = self.scienceon_client.search_articles(
                    query=term,
                    cur_page=page,
                    row_count=20
                )
                
                if results and isinstance(results, list):
                    documents = results
                    
                    # 품질 필터링 적용
                    filtered_docs = self._filter_quality(documents)
                    
                    if filtered_docs:
                        page_documents.extend(filtered_docs)
                        used_terms.add(term)
                        
                        # 목표 달성 시 중단
                        if len(page_documents) >= self.target_documents:
                            break
                
            except Exception as e:
                logging.error(f"검색어 '{term}' 검색 실패: {e}")
                continue
        
        return page_documents
    
    def _remove_duplicates(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 문서 제거"""
        seen_titles = set()
        unique_documents = []
        
        for doc in documents:
            title = doc.get('title', '').strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_documents.append(doc)
        
        return unique_documents
    
    def _filter_quality(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """문서 품질 필터링 및 source 필드 추가"""
        filtered = []
        
        for doc in documents:
            if self._is_quality_document(doc):
                # source 필드 추가 (기존 형식에 맞춤)
                cn = doc.get('CN', '')
                if cn:
                    doc['source'] = f"http://click.ndsl.kr/servlet/OpenAPIDetailView?keyValue={cn}&target=NART&cn={cn}"
                filtered.append(doc)
        
        return filtered
    
    def _is_quality_document(self, doc: Dict[str, Any]) -> bool:
        """문서 품질 검사"""
        # 제목이 있는지 확인
        title = doc.get('title', '').strip()
        if not title or len(title) < 5:
            return False
        
        # 초록이 있는지 확인 (선택적)
        abstract = doc.get('abstract', '').strip()
        if not abstract or abstract == '없음':
            # 초록이 없어도 제목이 충분히 길면 허용
            if len(title) < 20:
                return False
        
        # CN (논문 번호)이 있는지 확인 (ScienceON API의 경우)
        cn = doc.get('CN', '').strip()
        if not cn:
            return False
        
        return True
    
    def set_target_documents(self, target: int):
        """목표 문서 수 설정"""
        self.target_documents = target
    
    def set_max_pages(self, max_pages: int):
        """최대 검색 페이지 수 설정"""
        self.max_pages = max_pages
