"""
    Wikipedia를 위한 extractor


검색어에 한국어 영어 혼용하지 않음
키워드에 공백 제거 (영어 키워드만)

AND 조합 추가
self._add_operator(search_terms, number_of_operators=1) -> AND 1개 추가
가장 뒤의 | 부터 + 로 변경
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
        if not api_key:
            raise ValueError("API 키가 필요합니다. api_key 매개변수가 None이거나 비어있습니다.")
        
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
당신은 검색을 위한 키워드 추출 전문가입니다. 주어진 질문에서 가장 중요하고 검색에 유용한 키워드를 한국어로 추출해주세요.

질문: "{query}"

요구사항:
1. 한국어와 숫자 용어로 된 키워드만 추출하세요.
2. 전문용어와 기술용어를 우선적으로 선택하세요.
3. 축약어의 경우에는 축약어와 전체단어 모두 사용 키워드로 만드세요.
4. 2~5개 단어 추출하세요.
5. 중요도가 높은 순서대로 나열하세요.


출력 형식:키워드1, 키워드2, 키워드3, 키워드4


키워드:
"""
        
        try:
            response = self.model.generate_content(prompt)
            # print(f"한국어 전체 응답: {response}")
            keywords_text = response.text.strip()
            
            # 불필요한 접두사 제거
            keywords_text = keywords_text.replace("키워드:", "").strip()
            
            # 쉼표로 분리하고 정리
            keywords = [kw.strip() for kw in keywords_text.split(',')]
            keywords = [kw for kw in keywords if kw and len(kw) > 1]

            final_keywords = []
            seen_keywords = set()  # 중복 제거용 set
            
            for kw in keywords:
                if ' ' in kw:
                    # 공백 기준으로 분리하여 각각 추가
                    split_keywords = [word.strip() for word in kw.split() if word.strip() and len(word.strip()) > 1]
                    for split_kw in split_keywords:
                        # 대소문자 구분 없이 중복 체크
                        if split_kw.lower() not in seen_keywords:
                            final_keywords.append(split_kw)
                            seen_keywords.add(split_kw.lower())
                else:
                    # 공백이 없으면 그대로 추가 (중복 체크)
                    if kw.lower() not in seen_keywords:
                        final_keywords.append(kw)
                        seen_keywords.add(kw.lower())

            return final_keywords[:10]
            
        except Exception as e:
            logging.error(f"한국어 키워드 추출 실패: {e}")
            return []
    
    def _extract_english_keywords(self, query: str) -> List[str]:
        """영어 키워드 추출"""
        prompt = f"""
You are an expert in extracting keywords for academic paper searches. From the given query, please extract the most important and useful keywords for the search in English.

Query: "{query}"

Requirements:
1. Extract in English keywords or number terms only
2. Prioritize technical terms and scientific jargon
3. In the case of abbreviations, use both the abbreviation and the full term as keywords
4. After extracting the keywords, list 2~5 in order of importance
5. List them in descending order of importance

Output Format: keyword1, keyword2, keyword3, keyword4

Keywords:
"""
        
        try:
            response = self.model.generate_content(prompt)
            # print(f"영어 전체 응답: {response}")
            keywords_text = response.text.strip()
            
            # 불필요한 접두사 제거
            keywords_text = keywords_text.replace("Keywords:", "").strip()
            
            # 쉼표로 분리하고 정리
            keywords = [kw.strip() for kw in keywords_text.split(',')]
            keywords = [kw for kw in keywords if kw and len(kw) > 1]

            # 공백이 포함된 키워드를 분리하여 개별 키워드로 추가
            final_keywords = []
            seen_keywords = set()  # 중복 제거용 set
            
            for kw in keywords:
                if ' ' in kw:
                    # 공백 기준으로 분리하여 각각 추가
                    split_keywords = [word.strip() for word in kw.split() if word.strip() and len(word.strip()) > 1]
                    for split_kw in split_keywords:
                        # 대소문자 구분 없이 중복 체크
                        if split_kw.lower() not in seen_keywords:
                            final_keywords.append(split_kw)
                            seen_keywords.add(split_kw.lower())
                else:
                    # 공백이 없으면 그대로 추가 (중복 체크)
                    if kw.lower() not in seen_keywords:
                        final_keywords.append(kw)
                        seen_keywords.add(kw.lower())

            return final_keywords[:10]

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
                for i in range(len(korean_kw)):
                    korean_term = '|'.join(korean_kw)
                    search_terms.append(korean_term)
                    korean_kw = korean_kw[:-1]

            # 영어 검색어 (파이프로 구분)
            if english_kw:
                for i in range(len(english_kw)):
                    english_term = '|'.join(english_kw)
                    search_terms.append(english_term)
                    english_kw = english_kw[:-1]

            # AND 조합 추가
            search_terms = self._add_operator(search_terms, number_of_operators=0)
            # print(search_terms)
            
            # 혼합 검색어 생성
            # mixed_terms = self._generate_mixed_terms(korean_kw, english_kw)
            # search_terms.extend(mixed_terms)

            
            
            

            # 중복 제거 및 정리
            search_terms = list(set(search_terms))
            search_terms = [term for term in search_terms if term.strip()]

            return search_terms  # 최대 15개 검색어

        except Exception as e:
            logging.error(f"검색어 생성 실패: {e}")
            return []
    
    # def _generate_mixed_terms(self, korean_kw: List[str], english_kw: List[str]) -> List[str]:
    #     """혼합 검색어 생성 (더 많은 조합 생성)"""
    #     mixed_terms = []
        
    #     # 한국어 + 영어 조합 (더 많은 조합)
    #     if korean_kw and english_kw:
    #         for kr in korean_kw[:3]:  # 상위 3개
    #             for en in english_kw[:3]:  # 상위 3개
    #                 mixed_terms.append(f"{kr}|{en}")
        

        
    #     # 개별 키워드도 검색어로 추가
    #     mixed_terms.extend(korean_kw[:3])  # 상위 3개 한국어 키워드
    #     mixed_terms.extend(english_kw[:3])  # 상위 3개 영어 키워드
        
    #     return mixed_terms
    
    def _generate_keyword_rotations(self, keywords: List[str]) -> List[str]:
        """
        각 키워드가 맨 앞으로 한번씩 오는 검색어 조합 생성
        
        Args:
            keywords: 키워드 리스트
            
        Returns:
            각 키워드를 맨 앞으로 한 조합들의 리스트
        """
        rotations = []
        
        if len(keywords) <= 1:
            return rotations
            
        # 원본 순서 (이미 포함되어 있으므로 주석 처리)
        # rotations.append('|'.join(keywords))
        
        # 각 키워드를 맨 앞으로 한 조합 (첫 번째 제외, 이미 원본에 포함)
        for i in range(1, len(keywords)):
            # i번째 키워드를 맨 앞으로, 나머지는 원본 순서 유지
            rotated = [keywords[i]] + keywords[:i] + keywords[i+1:]
            rotations.append('|'.join(rotated))
        
        return rotations

    def _add_operator(self, search_terms: List[str], number_of_operators: int = 0) -> List[str]:
        """
        검색어의 "|" 연산자를 "+"로 변경
        
        Args:
            search_terms: 검색어 리스트
            number_of_operators: 뒤에서부터 변경할 "|" 연산자의 개수
            
        Returns:
            연산자가 변경된 검색어 리스트
        """
        if number_of_operators <= 0:
            return search_terms
        
        modified_terms = []
        
        for term in search_terms:
            if "|" not in term:
                # "|"가 없으면 그대로 추가
                modified_terms.append(term)
                continue
            
            # "|"로 분리
            parts = term.split("|")
            
            if len(parts) <= number_of_operators:
                # 모든 "|"를 "+"로 변경
                modified_term = "+".join(parts)
            else:
                # 뒤에서부터 number_of_operators 개수만큼 "+"로 변경
                front_parts = parts[:-number_of_operators]
                back_parts = parts[-number_of_operators:]
                
                # 앞부분은 "|"로, 뒷부분은 "+"로 연결
                front_term = "|".join(front_parts) if front_parts else ""
                back_term = "+".join(back_parts) if back_parts else ""
                
                if front_term and back_term:
                    modified_term = front_term + "+" + back_term
                else:
                    modified_term = front_term + back_term
            
            modified_terms.append(modified_term)
        
        return modified_terms        