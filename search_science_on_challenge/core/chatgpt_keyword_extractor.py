"""
ChatGPT API 기반 키워드 추출기
한국어, 영어, 전체 언어에 대한 키워드 추출을 지원합니다.
"""

import json
import os
import re
from typing import List, Dict, Optional
from openai import OpenAI

from .base_extractor import BaseKeywordExtractor


class ChatGPTKeywordExtractor(BaseKeywordExtractor):
    """ChatGPT API를 사용한 키워드 추출기"""
    
    def __init__(self, credentials_file: str = None, api_key: str = None, model: str = "gpt-4o-mini", language: str = "all"):
        """
        ChatGPT 키워드 추출기 초기화
        
        Args:
            credentials_file: ChatGPT API 자격증명 파일 경로 (api_key가 없을 때만 사용)
            api_key: OpenAI API 키 (직접 전달시 사용)
            model: 사용할 ChatGPT 모델 (gpt-4o-mini, gpt-4, gpt-3.5-turbo 등)
            language: 키워드 추출 언어 ("korean", "english", "all")
        """
        self.model = model
        self.language = language
        
        # API 키 가져오기
        if api_key:
            # 직접 전달받은 API 키 사용
            openai_api_key = api_key
        elif credentials_file:
            # 자격증명 파일에서 로드
            with open(credentials_file, 'r', encoding='utf-8') as f:
                credentials = json.load(f)
            openai_api_key = credentials["api_key"]
        else:
            raise ValueError("api_key 또는 credentials_file 중 하나는 필수입니다.")
        
        self.client = OpenAI(api_key=openai_api_key)
        
        # 언어별 프롬프트 설정
        self.prompts = {
            "korean": {
                "system": "당신은 과학 논문과 의학 문헌의 한국어 키워드 추출 전문가입니다. 주어진 질문이나 텍스트에서 가장 중요하고 검색에 유용한 한국어 키워드만을 추출해주세요.",
                "user_template": """
다음 질문에서 한국어 키워드만을 추출해주세요:

질문: "{query}"

요구사항:
1. 한국어 단어만 추출
2. 의학, 과학 용어 우선
3. 검색에 유용한 핵심 키워드만
4. 3-7개 단어
5. 쉼표로 구분

키워드:"""
            },
            "english": {
                "system": "You are an expert in extracting English keywords from scientific and medical literature. Extract only the most important and search-relevant English keywords from the given text.",
                "user_template": """
Extract English keywords from the following question:

Question: "{query}"

Requirements:
1. Extract only English words
2. Prioritize medical and scientific terms
3. Focus on core search-relevant keywords
4. 3-7 keywords maximum
5. Separate with commas

Keywords:"""
            },
            "all": {
                "system": "당신은 과학 논문과 의학 문헌의 키워드 추출 전문가입니다. 주어진 질문이나 텍스트에서 가장 중요하고 검색에 유용한 키워드를 한국어와 영어 모두 추출해주세요.",
                "user_template": """
다음 질문에서 중요한 키워드를 한국어와 영어로 추출해주세요:

질문: "{query}"

요구사항:
1. 한국어와 영어 키워드 모두 추출
2. 의학, 과학 용어 우선
3. 검색에 유용한 핵심 키워드만
4. 총 5-10개 단어
5. 쉼표로 구분

키워드:"""
            }
        }
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        주어진 쿼리에서 키워드를 추출합니다.
        
        Args:
            query: 키워드를 추출할 질문이나 텍스트
            
        Returns:
            추출된 키워드 딕셔너리 {'korean': [...], 'english': [...]}
        """
        # 언어별 프롬프트 선택
        prompt_config = self.prompts[self.language]
        
        # ChatGPT API 호출
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt_config["system"]},
                {"role": "user", "content": prompt_config["user_template"].format(query=query)}
            ],
            max_tokens=150,
            temperature=0.3
        )
        
        # 응답에서 키워드 추출
        keywords_text = response.choices[0].message.content.strip()
        
        # 키워드 파싱 및 정제
        keywords_list = self._parse_keywords(keywords_text)
        
        # 언어별로 분류하여 딕셔너리 형태로 반환
        return self._categorize_keywords(keywords_list)
    
    def _parse_keywords(self, keywords_text: str) -> List[str]:
        """
        ChatGPT 응답에서 키워드를 파싱합니다.
        
        Args:
            keywords_text: ChatGPT 응답 텍스트
            
        Returns:
            정제된 키워드 리스트
        """
        # 다양한 구분자로 분리
        keywords = re.split(r'[,，、\n\r]+', keywords_text)
        
        # 키워드 정제
        cleaned_keywords = []
        for keyword in keywords:
            # 앞뒤 공백 제거
            keyword = keyword.strip()
            
            # 숫자나 특수문자로만 된 키워드 제외
            if not keyword or keyword.isdigit():
                continue
                
            # 너무 짧거나 긴 키워드 제외
            if len(keyword) < 2 or len(keyword) > 50:
                continue
                
            # 불필요한 문구 제거
            unwanted_phrases = ['키워드:', 'Keywords:', '답변:', 'Answer:', '-', '•', '*']
            if any(phrase in keyword for phrase in unwanted_phrases):
                continue
                
            cleaned_keywords.append(keyword)
        
        # 중복 제거 및 최대 개수 제한
        unique_keywords = list(dict.fromkeys(cleaned_keywords))  # 순서 유지하며 중복 제거
        return unique_keywords[:10]  # 최대 10개
    

    
    def _categorize_keywords(self, keywords_list: List[str]) -> Dict[str, List[str]]:
        """
        키워드 리스트를 언어별로 분류
        
        Args:
            keywords_list: 키워드 리스트
            
        Returns:
            언어별 키워드 딕셔너리
        """
        korean_keywords = []
        english_keywords = []
        
        for keyword in keywords_list:
            # 한국어 문자가 포함된 경우
            if re.search(r'[가-힣]', keyword):
                korean_keywords.append(keyword)
            # 영어 문자만 포함된 경우
            elif re.search(r'^[a-zA-Z\s\-]+$', keyword):
                english_keywords.append(keyword)
        
        # 언어 설정에 따라 결과 필터링
        if self.language == "korean":
            return {"korean": korean_keywords, "english": []}
        elif self.language == "english":
            return {"korean": [], "english": english_keywords}
        else:  # all
            return {"korean": korean_keywords, "english": english_keywords}
    

    
    def extract_keywords_batch(self, queries: List[str]) -> List[Dict[str, List[str]]]:
        """
        여러 쿼리에 대해 배치로 키워드를 추출합니다.
        
        Args:
            queries: 키워드를 추출할 쿼리 리스트
            
        Returns:
            각 쿼리별 키워드 딕셔너리의 리스트
        """
        return [self.extract_keywords(query) for query in queries]
    
    def generate_search_terms(self, keywords: Dict[str, List[str]]) -> List[str]:
        """
        키워드로부터 검색어 생성
        
        Args:
            keywords: 추출된 키워드 딕셔너리
            
        Returns:
            검색어 리스트
        """
        search_terms = []
        
        korean_kw = keywords.get('korean', [])
        english_kw = keywords.get('english', [])
        
        # 한국어 검색어 (파이프로 구분)
        if korean_kw:
            korean_term = '|'.join(korean_kw)
            search_terms.append(korean_term)
        
        # 영어 검색어 (파이프로 구분)
        if english_kw:
            english_term = '|'.join(english_kw)
            search_terms.append(english_term)
        
        return search_terms
    
    def get_extractor_info(self) -> Dict[str, str]:
        """
        추출기 정보를 반환합니다.
        
        Returns:
            추출기 정보 딕셔너리
        """
        return {
            "name": "ChatGPT 키워드 추출기",
            "model": self.model,
            "language": self.language,
            "description": f"ChatGPT {self.model}을 사용한 {self.language} 키워드 추출"
        }