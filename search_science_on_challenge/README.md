## 수집 문서 평가


### ScienceON
1. scion_search_evaluation..py
    두가지 결과 도출
    1. 수집 문서 전체를 기준으로, 포함되지 않은 정답 문서 도출
    2. 쿼리 별 수집 문서 기준으로, 정답 문서를 하나라도 포함하지 않는 경우 도출
    - 수집 결과 경로 지정 (search_results_file = "outputs/search_meta_results_*.json")
    - 정답 문서 경로 지정 (answer_docs_file = "outputs/scion_answer_docs.json")
    outputs/scion_document_coverage_analysis_*.json

    output 출력 예시
    ```
    전체 정답 문서 수: 45
    전체 검색 결과 커버리지: 91.11% (41/45)
    질문별 검색 결과 커버리지: 68.89% (31/45)
    ```
    output json 파일 예시
    ```
    {  "total_answer_documents": 45,
    "analysis_1": {
    "description": "전체 검색 결과에 포함되지 않은 정답 문서",
    "missing_count": 4,
    "coverage_rate": 91.11,
    "missing_docs": [
        ...
    "analysis_2": {
    "description": "각 질문별로 정답 문서가 누락된 경우",
    "missing_count": 14,
    "coverage_rate": 68.89,
    "missing_per_question": [
      {
        ...
    ```


### MIRACL
1. wiki_search_evaluation..py
    목적: 쿼리 별 수집 문서 기준으로, 정답 문서를 하나라도 포함하지 않는 경우 도출
    - 수집 결과 경로 지정 (search_results_path = "./outputs/search_meta_results_*.json")
    - 정답 문서 경로 지정 (answer_docs_file = "../data/miracl/questions/miracl_en_query_documents.json")
    - 질문 언어(한국어 위키, 영어 위키) 에 따라 정답 경로 수정 필요(miracl_en_query_documents.json, 
    miracl_ko_query_documents.json)
    
    결과 출력 예시
    ```
    === 겹침 수 분포 ===
    1개 겹침: 13개 쿼리
    2개 겹침: 7개 쿼리
    3개 겹침: 5개 쿼리
    ```
