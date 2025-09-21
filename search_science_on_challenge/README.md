# 🔍 검색 메타데이터 시스템 v2.0

ScienceON API를 활용한 학술 논문 검색 및 메타데이터 생성 시스템의 리팩토링된 버전입니다.

## 📋 주요 특징

- **모듈화된 구조**: 각 기능별로 세분화된 파일 구조
- **50개 문서 보장**: 각 질문당 최소 50개 논문을 찾을 때까지 반복 검색
- **사용자 친화적**: 간단한 명령어로 실행 가능
- **다양한 출력 형식**: JSON, CSV, JSONL 형식 지원
- **자동 변환**: 결과 자동 변환 및 중복 제거
- **설정 관리**: 환경 변수 및 설정 파일 지원

## 🏗️ 디렉토리 구조

```
meta/refactored/
├── __init__.py
├── main.py                          # 메인 실행 파일
├── search_meta_system.py            # 통합 시스템 클래스
├── README.md                        # 사용자 가이드
├── core/                            # 핵심 엔진
│   ├── __init__.py
│   ├── keyword_extractor.py         # 키워드 추출기
│   └── document_searcher.py         # 문서 검색기
├── processors/                      # 쿼리 처리기
│   ├── __init__.py
│   ├── single_query_processor.py    # 단일 쿼리 처리기
│   └── batch_query_processor.py     # 배치 쿼리 처리기
├── utils/                           # 유틸리티
│   ├── __init__.py
│   ├── file_manager.py              # 파일 관리자
│   └── result_converter.py          # 결과 변환기
└── configs/                         # 설정 관리
    ├── __init__.py
    └── settings.py                  # 설정 관리자
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 환경 변수 설정
export GEMINI_API_KEY="your_gemini_api_key_here"
export TARGET_DOCUMENTS=50
export OUTPUT_DIRECTORY="./outputs"
```

### 2. 단일 질문 처리

```bash
cd meta/refactored
python main.py single "인공지능의 미래는 어떻게 될까요?"
```

### 3. 배치 처리 (CSV 파일)

```bash
python main.py batch ../test.csv
```

### 4. 결과 변환

```bash
python main.py convert
```

## 📖 상세 사용법

### 📚 가이드 문서
- **🚀 빠른 시작**: [`QUICK_START.md`](QUICK_START.md) - 기본 명령어와 빠른 시작
- **📖 상세 가이드**: [`USAGE_GUIDE.md`](USAGE_GUIDE.md) - 완전한 사용법과 문제 해결

### 명령어 옵션

| 명령어 | 설명 | 예시 |
|--------|------|------|
| `single [질문]` | 단일 질문 처리 | `python main.py single "질문내용"` |
| `batch [CSV파일]` | CSV 파일에서 배치 처리 | `python main.py batch test.csv` |
| `convert` | 최신 JSON 결과를 CSV/JSONL로 변환 | `python main.py convert` |
| `info` | 시스템 정보 출력 | `python main.py info` |
| `help` | 도움말 출력 | `python main.py help` |

### 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `GEMINI_API_KEY` | Gemini API 키 (필수) | - |
| `TARGET_DOCUMENTS` | 목표 문서 수 | 50 |
| `OUTPUT_DIRECTORY` | 출력 디렉토리 | ./outputs |
| `LOG_LEVEL` | 로그 레벨 | INFO |

### 출력 파일

- `search_meta_results_YYYYMMDD_HHMMSS.json` - 원본 JSON 결과
- `search_results_YYYYMMDD_HHMMSS.csv` - CSV 형식 (스프레드시트용)
- `search_documents_YYYYMMDD_HHMMSS.jsonl` - JSONL 형식 (중복 제거됨)
- `elapsed_times.json` - 실행 시간 통계

## 🔧 고급 설정

### 설정 파일 수정

`configs/settings.py`에서 기본값을 수정할 수 있습니다:

```python
# 검색 설정
"target_documents_per_query": 50,    # 목표 문서 수
"max_search_pages": 5,               # 최대 검색 페이지
"page_size": 20,                     # 페이지당 문서 수

# 품질 필터링
"min_title_length": 5,               # 최소 제목 길이
"require_abstract": False,           # 초록 필수 여부
"require_source": True,              # 출처 필수 여부
```

### 프로그래밍 방식 사용

```python
from search_meta_system import SearchMetaSystem

# 시스템 초기화
system = SearchMetaSystem("your_gemini_api_key")

# 단일 질문 처리
result = system.process_single_query("질문내용", target_documents=50)

# 배치 처리
results = system.process_batch_from_csv("test.csv", max_queries=10)

# 설정 변경
system.update_settings(
    target_documents_per_query=100,
    max_search_pages=10
)
```

## 📊 처리 과정

### 1. 키워드 추출
- Gemini API를 사용하여 질문에서 핵심 키워드 추출
- 한국어/영어 키워드 분리
- 학술 검색에 적합한 전문 용어 추출

### 2. 검색어 생성
- 추출된 키워드로부터 검색어 조합 생성
- 언어별 우선순위 설정
- 최대 8개 검색어 생성

### 3. 문서 검색
- ScienceON API를 통한 논문 검색
- **50개 문서 확보까지 반복 검색**
- 품질 필터링 적용
- 중복 제거

### 4. 결과 정리
- JSON 형식으로 구조화
- 자동 CSV/JSONL 변환
- 통계 정보 생성

## 🎯 50개 문서 보장 로직

시스템은 각 질문당 최소 50개의 고품질 논문을 찾을 때까지 다음 과정을 반복합니다:

1. **다중 검색어 사용**: 8개의 서로 다른 검색어로 검색
2. **페이지별 검색**: 최대 5페이지까지 검색
3. **품질 필터링**: 제목, 초록, 출처가 있는 논문만 선별
4. **중복 제거**: 동일한 제목의 논문 제거
5. **목표 달성 확인**: 50개 미만이면 추가 검색어로 재검색

## 🐛 문제 해결

### 자주 발생하는 오류

1. **API 키 오류**
   ```
   ❌ GEMINI_API_KEY가 설정되지 않았습니다.
   ```
   → 환경 변수 설정 또는 설정 파일 확인

2. **ScienceON API 오류**
   ```
   ❌ ScienceON API 인증 실패
   ```
   → `./configs/scienceon_api_credentials.json` 파일 확인

3. **CSV 파일 오류**
   ```
   ❌ CSV 파일을 찾을 수 없습니다
   ```
   → 파일 경로 및 존재 여부 확인

### 로그 확인

```bash
# 로그 레벨 변경
export LOG_LEVEL=DEBUG
python main.py batch test.csv
```

## 📈 성능 최적화

### 검색 속도 향상

1. **검색어 수 조정**: `max_search_terms` 설정
2. **페이지 수 조정**: `max_search_pages` 설정
3. **품질 필터링 완화**: `require_abstract=False`

### 메모리 사용량 최적화

1. **배치 크기 제한**: `max_queries` 파라미터 사용
2. **중간 결과 저장**: 큰 배치의 경우 중간 저장

## 🤝 기여하기

1. 이슈 리포트: 버그나 개선사항 제안
2. 기능 요청: 새로운 기능 제안
3. 코드 기여: Pull Request 제출

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해주세요.
