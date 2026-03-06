"""
키워드 추출기 베이스 클래스
모든 키워드 추출기는 이 클래스를 상속받아 구현합니다.
"""

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseKeywordExtractor(ABC):
    """키워드 추출기 베이스 클래스"""
    
    @abstractmethod
    def extract_keywords(self, query: str) -> List[str]:
        """
        주어진 쿼리에서 키워드를 추출합니다.
        
        Args:
            query: 키워드를 추출할 질문이나 텍스트
            
        Returns:
            추출된 키워드 리스트
        """
        pass
    
    @abstractmethod
    def get_extractor_info(self) -> Dict[str, str]:
        """
        추출기 정보를 반환합니다.
        
        Returns:
            추출기 정보 딕셔너리
        """
        pass
    
    def extract_keywords_batch(self, queries: List[str]) -> List[List[str]]:
        """
        여러 쿼리에 대해 배치로 키워드를 추출합니다.
        기본 구현은 순차적으로 처리합니다.
        
        Args:
            queries: 키워드를 추출할 쿼리 리스트
            
        Returns:
            각 쿼리별 키워드 리스트의 리스트
        """
        return [self.extract_keywords(query) for query in queries]