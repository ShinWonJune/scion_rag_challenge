수집 문서와 정답 문서 비교 방법


### ScienceON
1. 검색 수집한 document의 title 과 query만 매핑한 json 생성
    - scion_extract_questions_titles_from_metadata.py
    - search_meta_results_*.json 기준으로 쿼리id, document.title 를 추출한다.
    output: outputs/extracted_questions_titles_*.json

```json
    "question_id": 13,
    "question": "How does the content describe the combination of phenomenological and mechanistic approaches through Big Data analytics to support personalized healthcare?",
    "document_titles": [
      "Big Data, Big Knowledge: Big Data for Personalized Healthcare",
      "Big data analytics for personalized medicine",
      "Revolutionizing Utility of Big Data Analytics in Personalized Cardiovascular Healthcare",
      "The use of Big Data Analytics in healthcare",
```
2. Human Judge가 생성한 scienceON 정답 문서 json 파일
    output/scion_answer_docs.json

3. 두 문서를 비교
    - scion_analyze_document_coverage.py
    - 두가지 결과를 도출
    1. 수집 문서 전체를 기준으로, 포함되지 않는 정답 문서 도출
    2. 쿼리 별 수집 문서 기준으로, 정답 문서를 포함하지 않는 경우 도출
    outputs/document_coverage_analysis_*.json

    ```
    전체 정답 문서 수: 45
    전체 검색 결과 커버리지: 91.11% (41/45)
    질문별 검색 결과 커버리지: 68.89% (31/45)
    ```