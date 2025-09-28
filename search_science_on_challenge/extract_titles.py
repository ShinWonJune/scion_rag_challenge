#!/usr/bin/env python3
"""
Wikipedia 검색 결과에서 title만 추출하여 CSV 파일 생성
"""

import json
import csv
from pathlib import Path

def extract_titles_to_csv(json_path: str, output_csv: str):
    """
    JSON 파일에서 title들을 추출하여 CSV로 저장
    
    Args:
        json_path: 입력 JSON 파일 경로
        output_csv: 출력 CSV 파일 경로
    """
    try:
        # JSON 파일 읽기
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # title 추출
        titles = []
        query_count = 0
        
        # 결과에서 각 쿼리의 문서들을 순회
        for result in data.get('results', []):
            query = result.get('query', '')
            query_count += 1
            
            # 각 쿼리의 문서들에서 title 추출
            for doc in result.get('documents', []):
                title = doc.get('title', '')
                if title:  # 빈 title은 제외
                    titles.append({
                        'query_number': query_count,
                        'query': query,
                        'title': title,
                        'url': doc.get('url', ''),
                        'source': doc.get('source', 'Wikipedia')
                    })
        
        # CSV 파일로 저장
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['query_number', 'query', 'title', 'url', 'source']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 헤더 작성
            writer.writeheader()
            
            # 데이터 작성
            for title_info in titles:
                writer.writerow(title_info)
        
        print(f"✅ Title 추출 완료!")
        print(f"   총 쿼리: {query_count}개")
        print(f"   추출된 title: {len(titles)}개")
        print(f"   출력 파일: {output_csv}")
        
        return len(titles)
        
    except Exception as e:
        print(f"❌ Title 추출 실패: {e}")
        return 0

if __name__ == "__main__":
    # 입력 및 출력 파일 경로
    json_file = "outputs/search_meta_results_20250928_022254.json"
    csv_file = "wikipedia_titles_extracted.csv"
    
    # Title 추출 실행
    extract_titles_to_csv(json_file, csv_file)