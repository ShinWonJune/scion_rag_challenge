"""
키워드 추출기 모듈
- Gemini API를 사용한 키워드 추출
- 한국어/영어 키워드 분리
- 검색어 생성
"""

import re
import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai

class KeywordExtractor:
    """Gemini API를 사용한 키워드 추출기"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", language: str = "all"):
        """
        키워드 추출기 초기화
        
        Args:
            api_key: Google API 키
            model_name: 사용할 Gemini 모델명
            language: 키워드 추출 언어 ("korean", "english", "all")
        """
        self.api_key = api_key
        self.model_name = model_name
        self.language = language
        self.model = self._init_gemini()
        
    def _init_gemini(self) -> genai.GenerativeModel:
        """Gemini 모델 초기화"""
        genai.configure(api_key=self.api_key)
        generation_config = genai.GenerationConfig(
            temperature=0.2,
            candidate_count=1,
        )
        
        return genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
        )
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        질문에서 키워드 추출
        
        Args:
            query: 검색할 질문
            
        Returns:
            {'korean': [...], 'english': [...]} 형태의 키워드 딕셔너리
        """
        try:
            korean_keywords = []
            english_keywords = []
            
            if self.language in ["korean", "all"]:
                # 한국어 키워드 추출
                korean_keywords = self._extract_korean_keywords(query)
            
            if self.language in ["english", "all"]:
                # 영어 키워드 추출
                english_keywords = self._extract_english_keywords(query)
            
            return {
                'korean': korean_keywords,
                'english': english_keywords
            }
            
        except Exception as e:
            logging.error(f"키워드 추출 실패: {e}")
            return {'korean': [], 'english': []}
    
    def _extract_korean_keywords(self, query: str) -> List[str]:
        """한국어 키워드 추출"""
        prompt = f"""
당신은 논문 검색을 위한 키워드 추출 전문가입니다. 주어진 질문에서 가장 중요하고 검색에 유용한 키워드를 한국어로 추출해주세요.

질문: "{query}"

요구사항:
- 전문용어와 기술용어를 우선적으로 선택
- 축약어의 경우에는 축약어와 전체단어 모두 사용 키워드로 만드세요. 
- 각 키워드는 1-20자 이내로 간결하게
- 3-7개 단어
- 중요도가 높은 순서대로 나열


출력 형식:키워드1, 키워드2, 키워드3, 키워드4


키워드:
"""
        
        try:
            response = self.model.generate_content(prompt)
            keywords_text = response.text.strip()
            
            # 불필요한 접두사 제거
            keywords_text = keywords_text.replace("키워드:", "").strip()
            
            # 쉼표로 분리하고 정리
            keywords = [kw.strip() for kw in keywords_text.split(',')]
            keywords = [kw for kw in keywords if kw and len(kw) > 1]
            
            return keywords[:5]  # 최대 5개
            
        except Exception as e:
            logging.error(f"한국어 키워드 추출 실패: {e}")
            return []
    
    def _extract_english_keywords(self, query: str) -> List[str]:
        """영어 키워드 추출"""
        prompt = f"""
You are an expert in extracting keywords for academic paper searches. From the given query, please extract the most important and useful keywords for the search in English.

Query: "{query}"

Requirements:
1. Extract ONLY English keywords
2. Prioritize technical terms and scientific jargon
3. In the case of abbreviations, use both the abbreviation and the full term as keywords
4. Each keyword should be concise, within 1-20 characters
5. Extract 3-7 keywords
6. List them in descending order of importance

Output Format: keyword1, keyword2, keyword3, keyword4

Keywords:
"""
        
        try:
            response = self.model.generate_content(prompt)
            keywords_text = response.text.strip()
            
            # 불필요한 접두사 제거
            keywords_text = keywords_text.replace("English Keywords:", "").strip()
            
            # 쉼표로 분리하고 정리
            keywords = [kw.strip() for kw in keywords_text.split(',')]
            keywords = [kw for kw in keywords if kw and len(kw) > 1]
            
            return keywords[:5]  # 최대 5개
            
        except Exception as e:
            logging.error(f"영어 키워드 추출 실패: {e}")
            return []
    
    def generate_search_terms(self, keywords: Dict[str, List[str]]) -> List[str]:
        """
        키워드로부터 검색어 생성
        
        Args:
            keywords: 추출된 키워드 딕셔너리
            
        Returns:
            검색어 리스트
        """
        try:
            korean_kw = keywords.get('korean', [])
            english_kw = keywords.get('english', [])
            
            # 검색어 조합 생성
            search_terms = []
            
            # 한국어 검색어 (파이프로 구분)
            if korean_kw:
                korean_term = '|'.join(korean_kw)
                search_terms.append(korean_term)
            
            # 영어 검색어 (파이프로 구분)
            if english_kw:
                english_term = '|'.join(english_kw)
                search_terms.append(english_term)
            
            # 혼합 검색어 생성
            mixed_terms = self._generate_mixed_terms(korean_kw, english_kw)
            search_terms.extend(mixed_terms)
            
            # 중복 제거 및 정리
            search_terms = list(set(search_terms))
            search_terms = [term for term in search_terms if term.strip()]
            
            return search_terms[:15]  # 최대 15개 검색어
            
        except Exception as e:
            logging.error(f"검색어 생성 실패: {e}")
            return []
    
    def _generate_mixed_terms(self, korean_kw: List[str], english_kw: List[str]) -> List[str]:
        """혼합 검색어 생성 (더 많은 조합 생성)"""
        mixed_terms = []
        
        # 한국어 + 영어 조합 (더 많은 조합)
        if korean_kw and english_kw:
            for kr in korean_kw[:3]:  # 상위 3개
                for en in english_kw[:3]:  # 상위 3개
                    mixed_terms.append(f"{kr}|{en}")
        
        # 부분 조합 (더 긴 조합)
        if len(korean_kw) > 1:
            mixed_terms.append('|'.join(korean_kw[:4]))  # 4개까지
            if len(korean_kw) > 2:
                mixed_terms.append('|'.join(korean_kw[:2]))  # 2개 조합도 추가
        
        if len(english_kw) > 1:
            mixed_terms.append('|'.join(english_kw[:4]))  # 4개까지
            if len(english_kw) > 2:
                mixed_terms.append('|'.join(english_kw[:2]))  # 2개 조합도 추가
        
        # 개별 키워드도 검색어로 추가
        mixed_terms.extend(korean_kw[:3])  # 상위 3개 한국어 키워드
        mixed_terms.extend(english_kw[:3])  # 상위 3개 영어 키워드
        
        return mixed_terms
