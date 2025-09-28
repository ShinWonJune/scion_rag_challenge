#!/usr/bin/env python3
"""
MIRACL 영어 쿼리 문서에서 title만 추출하여 CSV 파일 생성
"""

import json
import csv
from pathlib import Path

def extract_miracl_titles_to_csv(json_path: str, output_csv: str):
    """
    MIRACL JSON 파일에서 title들을 추출하여 CSV로 저장
    
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
        
        # 각 쿼리를 순회
        for query_data in data:
            query_id = query_data.get('query_id', '')
            query = query_data.get('query', '')
            
            # 각 쿼리의 문서들에서 title 추출
            for doc in query_data.get('documents', []):
                doc_id = doc.get('doc_id', '')
                title = doc.get('title', '')
                
                if title:  # 빈 title은 제외
                    titles.append({
                        'query_id': query_id,
                        'query': query,
                        'doc_id': doc_id,
                        'title': title
                    })
        
        # CSV 파일로 저장
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['query_id', 'query', 'doc_id', 'title']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 헤더 작성
            writer.writeheader()
            
            # 데이터 작성
            for title_info in titles:
                writer.writerow(title_info)
        
        # 유니크한 title만 추출
        unique_titles = list(set([t['title'] for t in titles]))
        unique_csv = output_csv.replace('.csv', '_unique_titles.csv')
        
        with open(unique_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['title'])  # 헤더
            for title in sorted(unique_titles):
                writer.writerow([title])
        
        print(f"✅ MIRACL Title 추출 완료!")
        print(f"   총 쿼리: {len(data)}개")
        print(f"   추출된 title (중복 포함): {len(titles)}개")
        print(f"   유니크 title: {len(unique_titles)}개")
        print(f"   상세 파일: {output_csv}")
        print(f"   유니크 title 파일: {unique_csv}")
        
        return len(titles), len(unique_titles)
        
    except Exception as e:
        print(f"❌ Title 추출 실패: {e}")
        return 0, 0

if __name__ == "__main__":
    # 입력 및 출력 파일 경로
    json_file = "/app/data/miracl/questions/miracl_en_query_documents.json"
    csv_file = "miracl_titles_extracted.csv"
    
    # Title 추출 실행
    extract_miracl_titles_to_csv(json_file, csv_file)