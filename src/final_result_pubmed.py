"""
final_answers 디렉토리의 JSON 파일들을 PubMedQA 형태의 CSV로 변환하는 스크립트
- JSON 파일의 result에서 long_answer와 final_decision 분리
- abstract들을 context로 결합
- pubmedqa_complete.csv와 비교 가능한 형태로 변환
"""

import os
import json
import pandas as pd
import argparse
import glob
import re
import datetime
from typing import Dict, List, Optional

def extract_final_decision(result_text: str) -> tuple[str, str]:
    """
    result 텍스트에서 long_answer와 final_decision을 분리합니다.
    
    Args:
        result_text: JSON의 result 필드 텍스트
        
    Returns:
        (long_answer, final_decision) 튜플
    """
    # "Final decision:" 또는 "final decision:" 패턴 찾기 (대소문자 구분 없음)
    pattern = r'(?i)final\s+decision\s*:\s*(.+?)(?:\.|$)'
    match = re.search(pattern, result_text)
    
    if match:
        # final decision 부분 추출
        final_decision = match.group(1).strip()
        
        # 마크다운 형식 제거 (**텍스트** -> 텍스트)
        final_decision = re.sub(r'\*\*([^*]+)\*\*', r'\1', final_decision)
        final_decision = re.sub(r'\*\*', '', final_decision)
        final_decision = final_decision.strip()
        
        # final decision 부분 이전까지를 long_answer로 사용
        decision_start = match.start()
        long_answer = result_text[:decision_start].strip()
        
        # long_answer에서도 마크다운 형식 제거 (**텍스트** -> 텍스트)
        long_answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', long_answer)
        long_answer = re.sub(r'\*\*', '', long_answer)
        
        # 마지막에 불완전한 문장이 있다면 정리
        long_answer = re.sub(r'\s+$', '', long_answer)
        
    else:
        # "Final decision:"이 없는 경우 전체를 long_answer로 사용
        long_answer = result_text.strip()
        
        # long_answer에서도 마크다운 형식 제거
        long_answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', long_answer)
        long_answer = re.sub(r'\*\*', '', long_answer)
        
        final_decision = "unknown"
    
    return long_answer, final_decision

def extract_abstracts_from_json(json_data: Dict) -> str:
    """
    JSON 데이터에서 모든 abstract를 추출하여 context로 결합합니다.
    
    Args:
        json_data: JSON 파일 데이터
        
    Returns:
        결합된 context 문자열
    """
    abstracts = []
    
    # retrieval_results에서 hits의 abstract 추출
    if 'retrival' in json_data and 'retrieval_results' in json_data['retrival']:
        for result in json_data['retrival']['retrieval_results']:
            hits = result.get('hits', [])
            for hit in hits:
                abstract = hit.get('abstract', '').strip()
                if abstract and abstract != "" and abstract not in abstracts:
                    abstracts.append(abstract)
    
    # 단순히 공백으로 연결 (따옴표와 콤마 제거)
    if abstracts:
        context = ' '.join(abstracts)
    else:
        context = ""
    
    return context

def get_question_from_json(json_data: Dict) -> str:
    """
    JSON 데이터에서 질문을 추출합니다.
    
    Args:
        json_data: JSON 파일 데이터
        
    Returns:
        질문 문자열
    """
    # retrival -> retrieval_results에서 original 타입의 query 찾기
    if 'retrival' in json_data and 'retrieval_results' in json_data['retrival']:
        for result in json_data['retrival']['retrieval_results']:
            if result.get('query_meta', {}).get('type') == 'original':
                return result.get('query', '')
    
    # 못 찾으면 빈 문자열 반환
    return ""

def load_pubmedqa_mapping(pubmedqa_path: str) -> Dict[str, int]:
    """
    pubmedqa_complete.csv에서 질문-pubid 매핑을 로드합니다.
    
    Args:
        pubmedqa_path: pubmedqa_complete.csv 파일 경로
        
    Returns:
        질문을 키로 하는 pubid 매핑 딕셔너리
    """
    try:
        df = pd.read_csv(pubmedqa_path)
        return dict(zip(df['question'], df['pubid']))
    except Exception as e:
        print(f"Warning: Could not load pubmedqa_complete.csv: {e}")
        return {}

def process_json_files(input_dir: str, output_path: str, pubmedqa_path: Optional[str] = None):
    """
    input_dir의 모든 JSON 파일을 처리하여 CSV로 변환합니다.
    
    Args:
        input_dir: JSON 파일들이 있는 디렉토리
        output_path: 출력 CSV 파일 경로
        pubmedqa_path: pubmedqa_complete.csv 파일 경로 (옵션)
    """
    # JSON 파일 패턴 검색
    json_pattern = os.path.join(input_dir, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"Error: No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    
    # pubmedqa 매핑 로드
    pubmedqa_mapping = {}
    if pubmedqa_path and os.path.exists(pubmedqa_path):
        pubmedqa_mapping = load_pubmedqa_mapping(pubmedqa_path)
        print(f"Loaded {len(pubmedqa_mapping)} question-pubid mappings")
    
    # 결과 데이터 리스트
    results = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 각 필드 추출
            question = get_question_from_json(json_data)
            context = extract_abstracts_from_json(json_data)
            result_text = json_data.get('result', '')
            long_answer, final_decision = extract_final_decision(result_text)
            
            # pubid 매핑에서 찾기
            pubid = pubmedqa_mapping.get(question, '')
            
            result_row = {
                'question': question,
                'context': context,
                'long_answer': long_answer,
                'final_decision': final_decision,
                'pubid': pubid
            }
            
            results.append(result_row)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    # DataFrame 생성 및 저장
    if results:
        df = pd.DataFrame(results)
        
        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # CSV 저장
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully saved {len(results)} records to {output_path}")
        
        # 간단한 통계 출력
        print(f"\nSummary:")
        print(f"- Total questions: {len(results)}")
        print(f"- Questions with pubid: {sum(1 for r in results if r['pubid'])}")
        print(f"- Questions with context: {sum(1 for r in results if r['context'])}")
        print(f"- Final decisions distribution:")
        decision_counts = df['final_decision'].value_counts()
        for decision, count in decision_counts.items():
            print(f"  - {decision}: {count}")
    else:
        print("No valid data found to save")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Convert final_answers JSON files to PubMedQA CSV format"
    )
    
    parser.add_argument(
        "--input_dir",
        default="../results/final_answers",
        help="Directory containing JSON files (default: ../results/final_answers)"
    )
    
    parser.add_argument(
        "--output_path",
        default=None,
        help="Output CSV file path. If not specified, uses timestamp-based filename."
    )
    
    parser.add_argument(
        "--output_dir",
        default="../results/pubmedqa_final_answers",
        help="Output directory (default: ../results/pubmedqa_final_answers)"
    )
    
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output CSV filename. If not specified, uses timestamp-based filename."
    )
    
    parser.add_argument(
        "--pubmedqa_path",
        default="../pubmedqa/data/evaluation/pubmedqa_answer.csv",
        help="Path to pubmedqa_answer.csv for pubid mapping (default: ../pubmedqa/data/evaluation/pubmedqa_answer.csv)"
    )
    
    args = parser.parse_args()
    
    # 현재 날짜와 시간을 가져옵니다.
    now = datetime.datetime.now()
    # 원하는 형식(yymmdd_hhmmss)으로 문자열을 만듭니다.
    timestamp_str = now.strftime("%y%m%d_%H%M%S")
    
    # 출력 경로 설정
    if args.output_path:
        # output_path가 지정된 경우 그대로 사용
        output_path = args.output_path
    else:
        # 출력 디렉토리 설정
        output_dir = args.output_dir
        if not output_dir.endswith('/'):
            output_dir += '/'
            
        # 출력 파일명 설정
        if args.output_file:
            output_file = args.output_file
        else:
            output_file = f"pubmedqa_final_predictions_{timestamp_str}.csv"
        
        output_path = os.path.join(output_dir, output_file)
    
    # 경로 검증
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    print(f"Processing JSON files from: {args.input_dir}")
    print(f"Output CSV will be saved to: {output_path}")
    print(f"Timestamp: {timestamp_str}")
    
    if args.pubmedqa_path and os.path.exists(args.pubmedqa_path):
        print(f"Using pubmedqa mapping from: {args.pubmedqa_path}")
    else:
        print("Warning: pubmedqa_complete.csv not found, pubid will be empty")
        args.pubmedqa_path = None
    
    # 처리 실행
    process_json_files(args.input_dir, output_path, args.pubmedqa_path)

if __name__ == "__main__":
    main()