#!/usr/bin/env python3
"""
SciFact 답변 결과를 CSV 파일로 변환하는 도구

사용법:
    python final_result_scifact.py --input_dir /app/results/final_answers --output_dir /app/results/scifact_final_answers
"""

import os
import json
import csv
import argparse
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional


def extract_abstracts_from_retrieval(retrieval_data: Dict) -> Dict[str, str]:
    """
    retrieval 데이터에서 abstract들을 추출하여 딕셔너리로 반환
    
    Args:
        retrieval_data: retrieval 결과 딕셔너리
        
    Returns:
        Dict[str, str]: {'retrieve_docs_1': 'abstract1', 'retrieve_docs_2': 'abstract2', ...}
    """
    abstracts = {}
    
    if not retrieval_data or 'retrieval_results' not in retrieval_data:
        return abstracts
    
    for result in retrieval_data.get('retrieval_results', []):
        if 'hits' not in result:
            continue
            
        for i, hit in enumerate(result['hits'], 1):
            abstract_key = f"retrieve_docs_{i}"
            abstract_text = hit.get('abstract', '').strip()
            
            # abstract가 너무 길면 일정 길이로 자르기 (CSV 가독성을 위해)
            if len(abstract_text) > 500:
                abstract_text = abstract_text[:500] + "..."
            
            abstracts[abstract_key] = abstract_text
    
    return abstracts


def process_json_file(filepath: str) -> Optional[Dict[str, Any]]:
    """
    개별 JSON 파일을 처리하여 CSV 행 데이터를 생성
    
    Args:
        filepath: JSON 파일 경로
        
    Returns:
        Dict[str, Any]: CSV 행 데이터 또는 None (오류 시)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        row_data = {}
        
        # 기본 정보 추출
        row_data['id'] = data.get('id', '')
        
        # result에서 determination 추출
        result = data.get('result', {})
        row_data['label'] = result.get('determination', '')
        
        # retrieval에서 query와 abstracts 추출
        retrieval = data.get('retrival', {})  # 오타 있음 ('retrival' -> 'retrieval')
        
        # query 추출
        query = ""
        if 'retrieval_results' in retrieval:
            for result in retrieval.get('retrieval_results', []):
                if 'query' in result:
                    query = result['query']
                    break
        row_data['question'] = query
        
        # abstracts 추출
        abstracts = extract_abstracts_from_retrieval(retrieval)
        row_data.update(abstracts)
        
        return row_data
        
    except Exception as e:
        print(f"⚠️ 파일 처리 중 오류 발생 {filepath}: {e}")
        return None


def determine_max_docs(input_dir: str) -> int:
    """
    모든 JSON 파일을 스캔하여 최대 문서 수를 결정
    
    Args:
        input_dir: 입력 디렉토리 경로
        
    Returns:
        int: 최대 문서 수
    """
    max_docs = 0
    json_files = glob.glob(os.path.join(input_dir, "row_*.json"))
    
    for filepath in json_files[:10]:  # 샘플링을 위해 처음 10개 파일만 확인
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            retrieval = data.get('retrival', {})
            for result in retrieval.get('retrieval_results', []):
                if 'hits' in result:
                    max_docs = max(max_docs, len(result['hits']))
                    
        except Exception:
            continue
    
    return min(max_docs, 20)  # 최대 20개로 제한


def create_csv_from_json_files(input_dir: str, output_dir: str) -> str:
    """
    JSON 파일들을 CSV로 변환
    
    Args:
        input_dir: JSON 파일들이 있는 디렉토리
        output_dir: CSV 파일을 저장할 디렉토리
        
    Returns:
        str: 생성된 CSV 파일 경로
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 최대 문서 수 결정
    max_docs = determine_max_docs(input_dir)
    print(f"📄 최대 문서 수: {max_docs}")
    
    # CSV 헤더 생성
    headers = ['id', 'question', 'label']
    for i in range(1, max_docs + 1):
        headers.append(f'retrieve_docs_{i}')
    
    # 출력 파일 경로 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"scifact_predictions_{timestamp}.csv"
    output_filepath = os.path.join(output_dir, output_filename)
    
    # JSON 파일 목록 가져오기
    json_files = sorted(glob.glob(os.path.join(input_dir, "row_*.json")))
    
    if not json_files:
        raise ValueError(f"❌ JSON 파일을 찾을 수 없습니다: {input_dir}/row_*.json")
    
    print(f"📁 {len(json_files)}개의 JSON 파일을 찾았습니다.")
    
    # CSV 파일 생성
    processed_count = 0
    with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for json_file in json_files:
            row_data = process_json_file(json_file)
            if row_data:
                # 모든 헤더에 대해 값이 없으면 빈 문자열로 채우기
                for header in headers:
                    if header not in row_data:
                        row_data[header] = ''
                
                writer.writerow(row_data)
                processed_count += 1
    
    print(f"✅ {processed_count}개 레코드를 처리했습니다.")
    print(f"📄 CSV 파일 생성 완료: {output_filepath}")
    
    return output_filepath


def main():
    parser = argparse.ArgumentParser(
        description="SciFact 답변 결과를 CSV 파일로 변환합니다."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="JSON 답변 파일들이 있는 디렉토리 경로 (예: /app/results/final_answers)"
    )
    parser.add_argument(
        "--output_dir",
        default="/app/results/scifact_final_answers",
        help="CSV 파일을 저장할 디렉토리 경로 (기본값: /app/results/scifact_final_answers)"
    )
    
    args = parser.parse_args()
    
    # 입력 디렉토리 검증
    if not os.path.exists(args.input_dir):
        print(f"❌ 입력 디렉토리가 존재하지 않습니다: {args.input_dir}")
        return 1
    
    try:
        output_file = create_csv_from_json_files(args.input_dir, args.output_dir)
        print(f"\n🎉 변환 완료!")
        print(f"📍 출력 파일: {output_file}")
        
        # 파일 크기와 행 수 출력
        file_size = os.path.getsize(output_file)
        with open(output_file, 'r', encoding='utf-8') as f:
            row_count = sum(1 for _ in f) - 1  # 헤더 제외
        
        print(f"📊 파일 정보:")
        print(f"   - 크기: {file_size:,} bytes")
        print(f"   - 행 수: {row_count:,} rows")
        
        return 0
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    exit(main())