#!/usr/bin/env python3
"""
생성된 JSON 파일 분석 및 검증
"""

import json

def analyze_json_file(filepath):
    """JSON 파일 분석"""
    print(f"\n📊 {filepath} 분석:")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_queries = len(data)
    total_documents = sum(len(query['documents']) for query in data)
    
    print(f"   총 쿼리: {total_queries}개")
    print(f"   총 문서: {total_documents}개")
    print(f"   평균 문서/쿼리: {total_documents/total_queries:.1f}개")
    
    # 문서 수별 분포
    doc_counts = [len(query['documents']) for query in data]
    from collections import Counter
    distribution = Counter(doc_counts)
    
    print(f"   문서 수 분포:")
    for count, queries in sorted(distribution.items()):
        print(f"      {count}개 문서: {queries}개 쿼리")
    
    # 샘플 확인 (다중 문서가 있는 쿼리)
    multi_doc_queries = [q for q in data if len(q['documents']) > 1]
    if multi_doc_queries:
        print(f"\n   🔍 다중 문서 쿼리 예시:")
        for i, query in enumerate(multi_doc_queries[:3]):
            print(f"      {i+1}. Query: {query['query']}")
            print(f"         Documents ({len(query['documents'])}개):")
            for doc in query['documents']:
                print(f"            - {doc['doc_id']}: {doc['title']}")
    
    return data

def main():
    """메인 분석 함수"""
    print("🔍 JSON 파일 분석...")
    
    # 한국어 JSON 분석
    ko_data = analyze_json_file('miracl_ko_query_documents.json')
    
    # 영어 JSON 분석
    en_data = analyze_json_file('miracl_en_query_documents.json')
    
    print(f"\n📈 전체 요약:")
    print(f"   한국어: {len(ko_data)}개 쿼리, {sum(len(q['documents']) for q in ko_data)}개 문서")
    print(f"   영어: {len(en_data)}개 쿼리, {sum(len(q['documents']) for q in en_data)}개 문서")

if __name__ == "__main__":
    main()