#!/usr/bin/env python3
"""
PubMed 단일 검색어 테스트 스크립트
"""

import sys
import logging
from pathlib import Path
from pubmed_api_client import PubMedAPIClient

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_single_search():
    """단일 검색어 테스트"""
    
    # PubMed 클라이언트 초기화
    credentials_path = Path("./configs/pubmed_api_credentials.json")
    
    try:
        client = PubMedAPIClient(credentials_path)
        print("✅ PubMed 클라이언트 초기화 성공")
    except Exception as e:
        print(f"❌ PubMed 클라이언트 초기화 실패: {e}")
        return
    
    # 테스트 검색어
    test_terms = [
        "covid-19",
        "coronavirus",
        "machine learning"
    ]
    
    for term in test_terms:
        print(f"\n🔍 테스트 검색어: '{term}'")
        print("=" * 50)
        
        try:
            # 단일 검색어 테스트 (최대 3개 결과)
            documents = client.search_single_term_test(term, max_results=3)
            
            print(f"✅ 검색 완료: {len(documents)}개 문서 발견")
            
            # 결과 출력
            for i, doc in enumerate(documents, 1):
                print(f"\n📄 문서 {i}:")
                print(f"   ID: {doc.get('id', 'N/A')}")
                print(f"   제목: {doc.get('title', 'N/A')[:100]}...")
                print(f"   저자: {doc.get('authors', 'N/A')[:50]}...")
                print(f"   저널: {doc.get('journal', 'N/A')}")
                print(f"   연도: {doc.get('publication_year', 'N/A')}")
                
        except Exception as e:
            print(f"❌ 검색 실패: {e}")
            logging.error(f"검색어 '{term}' 실패: {e}")

if __name__ == "__main__":
    test_single_search()