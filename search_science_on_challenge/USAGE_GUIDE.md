# 🔍 Search Meta System 사용 가이드

## 📋 목차

1. [시스템 개요](#시스템-개요)
2. [설치 및 설정](#설치-및-설정)
3. [기본 사용법](#기본-사용법)
4. [고급 사용법](#고급-사용법)
5. [결과 파일 형식](#결과-파일-형식)
6. [문제 해결](#문제-해결)

## 🎯 시스템 개요

이 시스템은 **ScienceON API**를 사용하여 학술 논문을 검색하고, **Google Gemini API**로 키워드를 추출하는 모듈화된 검색 시스템입니다.

### 주요 기능

- ✅ 질문에서 키워드 자동 추출 (한국어/영어)
- ✅ 15개 검색어 생성 및 우선순위 설정
- ✅ 각 질문당 50개 고품질 논문 검색
- ✅ 자동 품질 필터링 및 중복 제거
- ✅ JSON, CSV, JSONL 형식으로 결과 출력

## 🛠 설치 및 설정

### 1. 필수 파일 준비

```bash
# API 자격 증명 파일
configs/gemini_api_credentials.json
configs/scienceon_api_credentials.json

# 검색할 질문 파일
test.csv
```

### 2. API 자격 증명 설정

```json
# configs/gemini_api_credentials.json
{
    "api_key": "YOUR_GEMINI_API_KEY"
}

# configs/scienceon_api_credentials.json
{
    "api_key": "YOUR_SCIENCEON_API_KEY",
    "base_url": "https://api.scienceon.kr"
}
```

### 3. CSV 파일 형식

```csv
질문
How can the rationale and structure of the free electronic textbook...
How do artificial neural networks employ weight matrices...
What approach and results characterize the system...
```

## 🚀 기본 사용법

### 단일 질문 처리

```bash
python main.py single "인공지능 수학 교육에 대한 연구는 어떻게 진행되고 있나요?"
```

### 배치 처리 (전체 질문)

```bash
python main.py batch test.csv
```

### 배치 처리 (제한된 질문 수)

```bash
python main.py batch test.csv 5  # 처음 5개 질문만 처리
```

### 목표 문서 수 설정

```bash
TARGET_DOCUMENTS=50 python main.py batch test.csv  # 각 질문당 100개 문서
```

## ⚙️ 고급 사용법

### 환경 변수 설정

```bash
# 목표 문서 수
export TARGET_DOCUMENTS=50

# 로그 레벨
export LOG_LEVEL=WARNING

# 출력 디렉토리
export OUTPUT_DIRECTORY=./custom_outputs
```

### 설정 파일 수정

`configs/settings.py`에서 다음 설정을 조정할 수 있습니다:

```python
# 검색 설정
"target_documents_per_query": 50,    # 목표 문서 수
"max_search_pages": 5,               # 최대 검색 페이지
"page_size": 20,                     # 페이지당 문서 수
"max_search_terms": 15,              # 최대 검색어 수
"max_keywords": 5,                   # 최대 키워드 수

# 품질 필터링 설정
"min_title_length": 5,               # 최소 제목 길이
"min_abstract_length": 10,           # 최소 초록 길이
"require_abstract": False,           # 초록 필수 여부
"require_source": True,              # 출처 필수 여부
```

## 📊 결과 파일 형식

### 1. JSON 결과 파일

```json
{
  "batch_statistics": {
    "total_queries": 3,
    "successful_queries": 3,
    "failed_queries": 0,
    "success_rate": 100.0,
    "total_documents_found": 150,
    "avg_documents_per_query": 50.0
  },
  "results": [
    {
      "question": "질문 내용",
      "status": "success",
      "keywords": {
        "korean": ["키워드1", "키워드2"],
        "english": ["keyword1", "keyword2"]
      },
      "search_queries": ["검색어1", "검색어2"],
      "documents": [
        {
          "title": "논문 제목",
          "abstract": "논문 초록",
          "CN": "논문 번호",
          "source": "논문 링크"
        }
      ],
      "total_documents_found": 50
    }
  ]
}
```

### 2. CSV 결과 파일

```csv
Question,Prediction_retrieved_article_name_1,Prediction_retrieved_article_name_2,...
질문,Title: 논문제목1 Abstract: 초록1 Source: 링크1,Title: 논문제목2 Abstract: 초록2 Source: 링크2,...
```

### 3. JSONL 결과 파일

```jsonl
{"title": "논문 제목", "abstract": "논문 초록", "CN": "논문 번호", "source": "논문 링크"}
{"title": "논문 제목", "abstract": "논문 초록", "CN": "논문 번호", "source": "논문 링크"}
```

## 🔧 문제 해결

### 자주 발생하는 오류

#### 1. API 키 오류

```
❌ API 키가 필요합니다.
```

**해결방법**: `configs/gemini_api_credentials.json` 파일에 올바른 API 키 설정

#### 2. CSV 파일 없음

```
❌ CSV 파일을 찾을 수 없습니다: test.csv
```

**해결방법**: CSV 파일이 올바른 경로에 있는지 확인

#### 3. 문서 수 부족

```
문서: 30개  # 목표 50개에 못 미침
```

**해결방법**:

- 검색어 수 증가: `max_search_terms` 설정 조정
- 페이지 수 증가: `max_search_pages` 설정 조정
- 품질 필터링 완화: `min_title_length` 등 설정 조정

### 성능 최적화

#### 1. 검색 속도 향상

```python
# configs/settings.py
"max_search_pages": 3,      # 페이지 수 줄이기
"page_size": 30,            # 페이지당 문서 수 늘리기
```

#### 2. 문서 품질 향상

```python
# configs/settings.py
"min_title_length": 10,     # 제목 길이 기준 강화
"require_abstract": True,   # 초록 필수로 설정
```

## 📁 디렉토리 구조

```
meta/refactored/
├── main.py                    # 메인 실행 파일
├── search_meta_system.py      # 시스템 오케스트레이터
├── configs/
│   ├── settings.py           # 설정 관리
│   ├── gemini_api_credentials.json
│   └── scienceon_api_credentials.json
├── core/
│   ├── keyword_extractor.py  # 키워드 추출기
│   └── document_searcher.py  # 문서 검색기
├── processors/
│   ├── single_query_processor.py  # 단일 질문 처리
│   └── batch_query_processor.py   # 배치 처리
├── utils/
│   ├── file_manager.py       # 파일 관리
│   └── result_converter.py   # 결과 변환
└── outputs/                  # 결과 파일 저장
    ├── search_meta_results_*.json
    ├── search_results_*.csv
    └── search_documents_*.jsonl
```

## 🎯 사용 예시

### 예시 1: 기본 배치 처리

```bash
# 3개 질문 처리
python main.py batch test.csv 3

# 결과:
# ✅ 배치 처리 완료!
#    총 질문: 3개
#    성공: 3개
#    실패: 0개
#    성공률: 100.0%
#    총 문서: 150개
#    평균 문서/질문: 50.0개
```

### 예시 2: 대량 문서 검색

```bash
# 각 질문당 100개 문서 검색
TARGET_DOCUMENTS=100 python main.py batch test.csv 5
```

### 예시 3: 단일 질문 테스트

```bash
python main.py single "머신러닝 알고리즘의 성능 평가 방법은?"
```

## 📞 지원

문제가 발생하면 다음을 확인하세요:

1. API 자격 증명 파일이 올바른지
2. CSV 파일 형식이 맞는지
3. 인터넷 연결 상태
4. API 사용량 한도

---

**🎉 이제 리팩토링된 Search Meta System을 사용하여 효율적으로 학술 논문을 검색하세요!**
