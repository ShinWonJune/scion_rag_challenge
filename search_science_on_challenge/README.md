수집 문서와 정답 문서 비교 방법


### ScienceON
1. scion_extract_questions_titles_from_metadata.py
    search_meta_results_*.json 기준으로 쿼리id, document 를 추출한다.
    output file: extracted_questions_titles_*.json

```json
    "question_id": 13,
    "question": "How does the content describe the combination of phenomenological and mechanistic approaches through Big Data analytics to support personalized healthcare?",
    "document_titles": [
      "Big Data, Big Knowledge: Big Data for Personalized Healthcare",
      "Big data analytics for personalized medicine",
      "Revolutionizing Utility of Big Data Analytics in Personalized Cardiovascular Healthcare",
      "The use of Big Data Analytics in healthcare",
```
2. scion_analyze_document_coverage.py
    search