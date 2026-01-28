"""
이 스크립트는 search_meta_results(쿼리 검색 결과) 와 miracl_en_query_documents(쿼리 답변을 위한 정답 문서 리스트)
간의 문서 제목 겹침을 분석하기 위하여 Query Success Rate(QSR)을 도출합니다.

주어진 쿼리 집합에 대하여 

QSR은 쿼리 집합에 대하여 각 쿼리에 대해 `검색된 문서 제목`과 
MIRACL 데이터셋에서 제공된 각 쿼리별 `정답 문서 제목` 간의 교집합을 계산하여,
겹치는 문서가 하나 이상 있는 쿼리의 비율을 나타냅니다.

문서 제목 비교 시 대소문자는 구분하지 않으며, 집합(Set) intersection 메서드 연산을 통해 문자열이 정확히 일치하는지 판별합니다.


"""


import json
from typing import Dict, List, Set

def analyze_document_overlap():
    """
    search_meta_results와 miracl_en_query_documents의 문서 제목 겹침을 분석합니다.
    """
    
    # 파일 경로
    search_results_path = "./outputs/search_meta_results_20251021_120731.json"
    answer_docs_file = "../data/miracl/questions/miracl_ko_query_documents.json"
    # answer_docs_file = "../data/miracl/questions/miracl_en_query_documents.json" # 영어 질의 사용시
    
    # 파일 로드
    with open(search_results_path, 'r', encoding='utf-8') as f:
        search_results = json.load(f)
    
    with open(answer_docs_file, 'r', encoding='utf-8') as f:
        miracl_data = json.load(f)
    
    # MIRACL 데이터를 쿼리별로 매핑 (query_id를 키로 사용)
    miracl_by_query_id = {}
    for item in miracl_data:
        miracl_by_query_id[item["query_id"]] = item
    # id : {id, query, documents[]} 형태

    # 결과 저장용 변수들
    overlap_results = []
    queries_with_no_overlap = []
    total_overlap_count = 0
    
    print("=== 쿼리별 문서 제목 겹침 분석 ===\n")
    
    # search_results의 각 쿼리에 대해 분석
    # search_results의 list에 있는 dict 객체를 result로 할당
    for result in search_results["results"]:
        query_text = result["query"]
        
        
        # 해당 쿼리와 매칭되는 MIRACL 데이터 찾기
        miracl_match = None
        for query_id, miracl_item in miracl_by_query_id.items():
            if miracl_item["query"] == query_text:
                miracl_match = miracl_item # {id, query, documents[]}
                break
        
        if not miracl_match:
            print(f"⚠️  MIRACL에서 매칭되는 쿼리를 찾을 수 없음: '{query_text}'")
            continue
        
        # 문서 제목 추출 (대소문자 구분 없이 비교)
        search_titles = set(doc["title"].lower() for doc in result["documents"]) 
        miracl_titles = set(doc["title"].lower() for doc in miracl_match["documents"])
        
        # 겹치는 제목 찾기
        overlapping_titles = search_titles.intersection(miracl_titles)
        overlap_count = len(overlapping_titles)
        
        # 결과 저장
        overlap_info = {
            "query": query_text,
            "query_id": miracl_match["query_id"],
            "search_doc_count": len(search_titles),
            "miracl_doc_count": len(miracl_titles),
            "overlap_count": overlap_count,
            "overlapping_titles": list(overlapping_titles)
        }
        overlap_results.append(overlap_info)
        total_overlap_count += overlap_count
        
        # 겹치지 않는 쿼리 체크
        if overlap_count == 0:
            queries_with_no_overlap.append(overlap_info)
        
        # 개별 결과 출력
        # print(f"Query ID {miracl_match['query_id']}: {query_text}")
        # print(f"  Search 문서 수: {len(search_titles)}")
        # print(f"  MIRACL 문서 수: {len(miracl_titles)}")
        # print(f"  겹치는 문서 수: {overlap_count}")
        # if overlap_count > 0:
        #     print(f"  겹치는 제목들: {', '.join(overlapping_titles)}")
        # print()
    
    # 전체 통계 출력
    print("=" * 50)
    print("=== 전체 통계 ===")
    print(f"총 분석된 쿼리 수: {len(overlap_results)}")
    print(f"전혀 겹치지 않는 쿼리 수: {len(queries_with_no_overlap)}")
    print(f"하나 이상 겹치는 쿼리 수: {len(overlap_results) - len(queries_with_no_overlap)}")
    print(f"전체 겹치는 문서 수: {total_overlap_count}")
    print(f"평균 겹침 수: {total_overlap_count / len(overlap_results):.2f}")
    
    # 겹치지 않는 쿼리들 상세 정보
    if queries_with_no_overlap:
        print(f"\n=== 전혀 겹치지 않는 {len(queries_with_no_overlap)}개 쿼리 ===")
        for query_info in queries_with_no_overlap:
            print(f"Query ID {query_info['query_id']}: {query_info['query']}")
    
    # 겹침 수에 따른 분포
    overlap_distribution = {}
    for result in overlap_results:
        count = result["overlap_count"]
        overlap_distribution[count] = overlap_distribution.get(count, 0) + 1
    
    print(f"\n=== 겹침 수 분포 ===")
    for overlap_count in sorted(overlap_distribution.keys()):
        print(f"{overlap_count}개 겹침: {overlap_distribution[overlap_count]}개 쿼리")
    
    return {
        "overlap_results": overlap_results,
        "queries_with_no_overlap": queries_with_no_overlap,
        "total_queries": len(overlap_results),
        "no_overlap_count": len(queries_with_no_overlap),
        "total_overlap_count": total_overlap_count,
        "overlap_distribution": overlap_distribution
    }

# 분석 실행
if __name__ == "__main__":
    results = analyze_document_overlap()