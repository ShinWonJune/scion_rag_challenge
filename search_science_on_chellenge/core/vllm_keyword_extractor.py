"""
vLLM 키워드 추출기 모듈
- vLLM OpenAI 호환 API를 사용한 키워드 추출
- 한국어/영어 키워드 분리
- 검색어 생성
"""

import re
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from typing import List

class VLLMKeywordExtractor:
    """vLLM OpenAI 호환 API를 사용한 키워드 추출기"""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "openai/gpt-oss-120B"):
        """
        vLLM 키워드 추출기 초기화
        
        Args:
            base_url: vLLM 서버 URL
            model: 사용할 모델명
        """
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key="dummy-key")
        
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        질문에서 키워드 추출
        
        Args:
            query: 검색할 질문
            
        Returns:
            {'korean': [...], 'english': [...]} 형태의 키워드 딕셔너리
        """
        try:
            # 한국어 키워드 추출
            korean_keywords = self._extract_korean_keywords(query)
            # print(f"DEBUG - 한국어 키워드: {korean_keywords}")
            # 영어 키워드 추출
            english_keywords = self._extract_english_keywords(query)
            # print(f"DEBUG - 영어 키워드: {english_keywords}")
            return {
                'korean': korean_keywords,
                'english': english_keywords
            }
            
        except Exception as e:
            logging.error(f"키워드 추출 실패: {e}")
            return {'korean': [], 'english': []}

    def _extract_keywords_from_reasoning(self, reasoning_content: str) -> List[str]:
        """
        reasoning_content에서 키워드 추출
        
        Args:
            reasoning_content: vLLM의 reasoning_content 문자열
            
        Returns:
            중복 제거된 키워드 리스트 (순서 유지)
        """
        if not reasoning_content:
            return []
        
        # 키워드 패턴 찾기: "Keywords:", "maybe:", "keywords:", "Probably:" 이후 마침표까지
        patterns = [
            r'Keywords:\s*([^.]+)\.',
            r'maybe:\s*([^.]+)\.',
            r'keywords:\s*([^.]+)\.',
            r'Probably:\s*([^.]+)\.'
        ]
        
        all_keywords = []
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning_content, re.IGNORECASE)
            for match in matches:
                # 콤마로 분리하고 각 키워드 정리
                keywords = [kw.strip() for kw in match.split(',')]
                keywords = [kw for kw in keywords if kw and len(kw) > 1]
                all_keywords.extend(keywords)
        
        # 중복 제거하되 순서 유지 (OrderedDict 대신 dict 사용 - Python 3.7+에서 순서 보장)
        seen = set()
        unique_keywords = []
        for keyword in all_keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:5]  # 최대 5개
        
    def _extract_korean_keywords(self, query: str) -> List[str]:
        """한국어 키워드 추출"""
        prompt = f"""
당신은 논문 검색을 위한 키워드 추출 전문가입니다. 주어진 질문에서 ScienceON API 검색에 최적화된 3-5개의 핵심 키워드들을 한국어로 추출해주세요.

질문: "{query}"

다음 형식으로 키워드를 추출하세요:

1. 한국어 키워드: 3-5개의 핵심 키워드를 쉼표로 구분


규칙:
- 전문용어와 기술용어를 우선적으로 선택
- 축약어, 전체용어를 모두 알 경우, 모두 사용 키워드로 만드세요
- 각 키워드는 1-20자 이내로 간결하게

출력 형식:키워드1, 키워드2, 키워드3, 키워드4


키워드:
"""
        
        try:
            messages = [
            {"role": "system", "content": "Reasoning: low\nDo not output analysis. Return only the final answer."},
            {"role": "user", "content": prompt},
            ] 
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=150,
                temperature=0.1
            )
            # print(f"DEBUG - 한국어 전체 응답: {response}")
            content = response.choices[0].message.content
            # print(f"DEBUG - vLLM 한국어 응답: '{content}'")

            # content가 비어있거나 'None'인 경우 reasoning_content에서 키워드 추출 시도
            if not content or content.strip() == '' or content == 'None':
                reasoning_content = getattr(response.choices[0].message, 'reasoning_content', '')
                
                if reasoning_content:
                    keywords = self._extract_keywords_from_reasoning(reasoning_content)
                    # print(f"DEBUG - 추출된 키워드: {keywords}")
                    return keywords
                else:
                    return []
            
            keywords_text = content.strip()
            
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
당신은 논문 검색을 위한 키워드 추출 전문가입니다. 주어진 질문에서 ScienceON API 검색에 최적화된 핵심 키워드들을 영어로 추출해주세요.

질문: "{query}"


영어 키워드: 3-5개의 핵심 키워드를 쉼표로 구분 (textbook, artificial intelligence)

규칙:
- 전문용어와 기술용어를 우선적으로 선택
- 축약어, 전체용어를 모두 알 경우, 모두 사용 키워드로 만드세요. 전문용어가 전체용어로 질문에 들어온 경우 확실하게 키워드로 만드세요. (예: SVM, DTG, NLP, artificial intelligence, Warehouse Management System)
- 각 키워드는 1-20자 이내로 간결하게


출력 형식:keyword1, keyword2, keyword3, keyword4



키워드:"""

        try:
            messages = [
            {"role": "system", "content": "Reasoning: low\nDo not output analysis. Return only the final answer."},
            {"role": "user", "content": prompt},
            ]   
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=150,
                temperature=0.2
            )
            # print(f"DEBUG - 영어 전체 응답: {response}")
            content = response.choices[0].message.content
            # print(f"DEBUG - vLLM 영어 응답: '{content}'")
            
            # content가 비어있거나 'None'인 경우 reasoning_content에서 키워드 추출 시도
            if not content or content.strip() == '' or content == 'None':
                reasoning_content = getattr(response.choices[0].message, 'reasoning_content', '')
                
                if reasoning_content:
                    keywords = self._extract_keywords_from_reasoning(reasoning_content)
                    # print(f"DEBUG - 추출된 키워드: {keywords}")
                    return keywords
                else:
                    return []
            
            keywords_text = content.strip()
            
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
            # print(f"DEBUG - 한국어 키워드: {korean_kw}")
            # print(f"DEBUG - 영어 키워드: {english_kw}")
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