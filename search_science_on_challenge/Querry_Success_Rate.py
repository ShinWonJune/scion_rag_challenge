
"""
Wiki Search Evaluation (QSR) Script
===================================

이 스크립트는 검색 결과(Search Results)와 정답 데이터(Ground Truth: miracl_answer_documents_*.json)를 비교하여
검색 시스템의 성능 지표인 QSR(Query Success Rate)을 측정합니다.
QSR은 miracl 데이터셋에 한정하여 사용됩니다.

평가 방식:
    1. 검색된 문서 제목과 정답 문서 제목을 비교합니다.
    2. 문서 제목 비교 시 대소문자는 구분하지 않으며, 집합(Set)의 교집합(intersection) 연산을 통해 
       문자열이 정확히 일치하는지 판별합니다.
    3. 교집합인 문서가 하나 이상 존재하면 해당 쿼리는 '성공'으로 간주합니다.

Metric:
    QSR = (성공한 쿼리 수 / 전체 쿼리 수) * 100


사용법
argument로 쿼리 검색 결과 파일 경로를 전달하여 정답 문서와 비교 평가를 수행합니다.
정답문서는 영어 또는 한국어가 존재하며 기본적으로 영어로 설정되어 있습니다. --lang 옵션으로 변경 가능합니다. (eng 또는 ko)

python Querry_Success_Rate.py search_meta_results_*.json

or

python Querry_Success_Rate.py search_meta_results_*.json --lang ko

"""

import json
import argparse
import os
import logging
from typing import Dict, List, Set, Any, Tuple

def load_json(filepath: str) -> Any:
    """JSON 파일을 로드합니다."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다 - {filepath}")
        return []

def evaluate_search_performance(search_path: str, gt_path: str) -> Dict[str, Any]:
    """
    검색 결과와 정답 문서를 비교하여 평가 지표를 계산합니다.
    """
    # 0. 경로 보정 (입력된 파일이 없으면 outputs 폴더 확인)
    if not os.path.exists(search_path):
        candidate_path = os.path.join("outputs", search_path)
        if os.path.exists(candidate_path):
            search_path = candidate_path
            print(f"[Info] '{search_path}'에서 파일을 찾았습니다.")

    # 1. 데이터 로드
    search_data = load_json(search_path)
    miracl_data = load_json(gt_path)

    if not search_data or not miracl_data:
        return {}

    # 2. 정답 데이터 준비 (딕셔너리로 변환)
    gt_map = {item["query"]: item for item in miracl_data} # {query_text: {id, query, documents[]}}

    overlap_results = []
    queries_with_no_overlap = []
    total_overlap_count = 0
    successful_query_count = 0

    print("=== 쿼리별 문서 제목 겹침 분석 ===\n")

    for result in search_data.get("results", []):  # 모든 쿼리에 대해 반복
        query_text = result["query"] 
        
        
        gt_item = gt_map.get(query_text)  # 딕셔너리 기반 정답 데이터, {id, query, documents[]} 
        if not gt_item:
            print(f"⚠️  Skip: MIRACL 데이터셋에 없는 쿼리입니다: '{query_text}'")
            continue

        # 3. 문서 제목 정규화 (소문자 변환) 및 집합 생성
        search_titles = {doc["title"].lower() for doc in result.get("documents", [])}
        gt_titles = {doc["title"].lower() for doc in gt_item.get("documents", [])}

        # 4. 교집합(Intersection) 계산
        overlapping_titles = search_titles.intersection(gt_titles)
        overlap_count = len(overlapping_titles)
        
        # 통계 집계
        is_success = overlap_count > 0 # 교집합이 존재하면 '성공 쿼리'  로 간주
        total_overlap_count += overlap_count
        if is_success:
            successful_query_count += 1
        
        overlap_info = {
            "query": query_text,
            "query_id": gt_item["query_id"],
            "search_doc_count": len(search_titles),
            "miracl_doc_count": len(gt_titles),
            "overlap_count": overlap_count,
            "overlapping_titles": list(overlapping_titles),
            "is_success": is_success
        }
        overlap_results.append(overlap_info)

        if not is_success:
            queries_with_no_overlap.append(overlap_info)

        # 개별 로그 출력
        _print_single_query_log(gt_item['query_id'], query_text, len(search_titles), len(gt_titles), overlap_count, overlapping_titles)

    return {
        "overlap_results": overlap_results,
        "queries_with_no_overlap": queries_with_no_overlap,
        "total_queries": len(overlap_results),
        "successful_query_count": successful_query_count,
        "failed_query_count": len(queries_with_no_overlap),
        "total_overlap_count": total_overlap_count
    }

def _print_single_query_log(qid: str, text: str, n_search: int, n_gt: int, n_overlap: int, titles: Set[str]):
    """개별 쿼리 분석 결과를 출력하는 헬퍼 함수"""
    print(f"Query ID {qid}: {text}")
    print(f"  - 겹치는 문서 수: {n_overlap}")
    # if n_overlap > 0:
    #     print(f"  - 매칭된 제목: {', '.join(titles)}")
    print("-" * 30)

def print_evaluation_report(stats: Dict[str, Any]):
    """최종 평가 리포트를 출력합니다."""
    if not stats:
        print("평가할 데이터가 없습니다.")
        return

    total = stats["total_queries"]
    success = stats["successful_query_count"]
    
    # QSR 계산
    qsr = (success / total * 100) if total > 0 else 0.0

    print("\n" + "=" * 10)
    print("=== 📊 최종 평가 리포트 ===")
    print("=" * 10)
    print(f"총 분석 쿼리 수  : {total}")
    print(f"성공한 쿼리 수   : {success} (하나 이상 정답 포함)")
    print(f"실패한 쿼리 수   : {stats['failed_query_count']}")
    print(f"평균 정답 문서 수: {stats['total_overlap_count'] / total:.2f}개")
    print("-" * 10)
    print(f"✅ Query Success Rate (QSR): {qsr:.2f}%")
    print("=" * 10)

    # 실패한 쿼리 목록 출력
    if stats["queries_with_no_overlap"]:
        print(f"\n[참고] 실패한 쿼리 목록 ({len(stats['queries_with_no_overlap'])}개):")
        for item in stats["queries_with_no_overlap"]:
            print(f"  - [ID:{item['query_id']}] {item['query']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wiki Search Evaluation (QSR) Script")
    parser.add_argument("search_result_path", type=str, help="Path to the search result JSON file")
    parser.add_argument("--lang", type=str, choices=["en", "ko"], default="en", help="Language code (en or ko)")
    
    args = parser.parse_args()
    
    # 정답 파일 경로 구성
    gt_filename = f"miracl_answer_documents_{args.lang}.json"
    
    # 유연한 경로 확인: ../data (하위 폴더 실행 시) 또는 data (루트 실행 시)
    possible_paths = [
        os.path.join("..", "data", "miracl", "answer_docs", gt_filename),
        os.path.join("data", "miracl", "answer_docs", gt_filename)
    ]
    
    gt_path = possible_paths[0] # Default
    for path in possible_paths:
        if os.path.exists(path):
            gt_path = path
            break
            
    print(f"[Info] Target Search Result: {args.search_result_path}")
    print(f"[Info] Target Ground Truth: {gt_path}")

    # 평가 실행
    evaluation_stats = evaluate_search_performance(args.search_result_path, gt_path)
    
    # 리포트 출력
    print_evaluation_report(evaluation_stats)