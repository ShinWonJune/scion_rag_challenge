#!/usr/bin/env python3
"""
MIRACL title과 Wikipedia title의 교집합 찾기
"""

import csv
from pathlib import Path

def compare_titles(miracl_csv: str, wikipedia_csv: str):
    """
    MIRACL title과 Wikipedia title의 교집합 찾기
    
    Args:
        miracl_csv: MIRACL title CSV 파일 경로
        wikipedia_csv: Wikipedia title CSV 파일 경로
    """
    try:
        # MIRACL titles 읽기
        miracl_titles = set()
        with open(miracl_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                miracl_titles.add(row['title'].strip())
        
        # Wikipedia titles 읽기
        wikipedia_titles = set()
        with open(wikipedia_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                wikipedia_titles.add(row['title'].strip())
        
        # 교집합 찾기
        common_titles = miracl_titles.intersection(wikipedia_titles)
        
        # MIRACL에만 있는 titles
        miracl_only = miracl_titles - wikipedia_titles
        
        # Wikipedia에만 있는 titles (너무 많으므로 개수만)
        wikipedia_only_count = len(wikipedia_titles - miracl_titles)
        
        # 결과 출력
        print(f"📊 MIRACL vs Wikipedia Title 비교 결과")
        print("=" * 50)
        print(f"MIRACL 전체 title: {len(miracl_titles)}개")
        print(f"Wikipedia 전체 title: {len(wikipedia_titles)}개")
        print(f"공통 title: {len(common_titles)}개")
        print(f"MIRACL 전용 title: {len(miracl_only)}개")
        print(f"Wikipedia 전용 title: {wikipedia_only_count}개")
        print()
        
        # 겹치는 비율
        overlap_rate = (len(common_titles) / len(miracl_titles)) * 100 if miracl_titles else 0
        print(f"📈 MIRACL 기준 겹치는 비율: {overlap_rate:.1f}%")
        print()
        
        # 공통 titles 출력
        if common_titles:
            print(f"🎯 공통 Title 목록 ({len(common_titles)}개):")
            for title in sorted(common_titles):
                print(f"   - {title}")
            print()
        
        # MIRACL 전용 titles 출력
        if miracl_only:
            print(f"📝 MIRACL 전용 Title 목록 ({len(miracl_only)}개):")
            for title in sorted(miracl_only):
                print(f"   - {title}")
            print()
        
        # 결과를 CSV로 저장
        with open('title_comparison_result.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['title', 'in_miracl', 'in_wikipedia', 'status'])
            
            # 공통 titles
            for title in sorted(common_titles):
                writer.writerow([title, 'Yes', 'Yes', 'Common'])
            
            # MIRACL 전용
            for title in sorted(miracl_only):
                writer.writerow([title, 'Yes', 'No', 'MIRACL_Only'])
        
        print(f"💾 결과가 'title_comparison_result.csv'에 저장되었습니다.")
        
        return len(common_titles), len(miracl_only), overlap_rate
        
    except Exception as e:
        print(f"❌ 비교 실패: {e}")
        return 0, 0, 0

if __name__ == "__main__":
    # 파일 경로
    miracl_file = "miracl_titles_extracted_unique_titles.csv"
    wikipedia_file = "wikipedia_titles_unique.csv"
    
    # 비교 실행
    compare_titles(miracl_file, wikipedia_file)