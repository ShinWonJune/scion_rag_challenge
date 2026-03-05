# AGENTS.md

ScienceON 챌린지용 과학 문헌 RAG 파이프라인이다.
질문 → 문서 검색(ScienceON/PubMed/Wikipedia) → 임베딩/벡터DB → 밀집 검색 → LLM 답변 생성 순서로 동작한다.
이 파일은 리팩토링을 수행하는 AI 에이전트를 위한 지침이다.

---

## 절대 수정 금지

- `pipeline/scienceon_api_example.py` — 대회 측 제공 파일. 내부 로직, 클래스명, 메서드 시그니처 변경 금지.  
  (`ScienceONAPIClient` 래핑 시 `search_articles(query, cur_page, row_count, fields)` 시그니처 기준으로 어댑터 작성)
- `data/`, `results/`, `outputs/` 아래 데이터 파일 내용 수정 금지 (이동/추적 해제는 허용)
- `configs/query_encoder/*.json`의 `model_name`, `embedding_dim` 등 핵심 설정값 임의 변경 금지
- `src/` 아래 재사용 라이브러리를 `pipeline/`으로 이동 금지 (단계 실행 스크립트만 이동 대상)
- `retrival` 키 이름 수정 시, 하류 파이프라인 호환성 먼저 확인 (`rg "retrival" src/` 로 사용처 조회)

---

## 설계 규칙

**의존 방향**: `pipeline/` → `src/` (단방향). `src/`가 `pipeline/`을 import하는 역방향 의존 금지.

**`pipeline/` vs `src/` 경계**: `pipeline/step*.py`는 CLI 인수 파싱 + 오케스트레이션만, 실제 로직은 `src/`에.

**컴포지션 루트 패턴**: 무거운 객체(임베딩 모델, LLM 클라이언트)는 `run_pipeline.py`에서 한 번만 생성하고 각 단계 함수에 인수로 주입. 싱글톤(`_instance` 클래스 변수) 사용 금지.

**코드 컨벤션**:
- 타입 힌트: 새로 작성하는 함수에 필수
- 로깅: `print()` 대신 `logging.info/warning/error()` (사용자 대면 진행상황 출력은 `print()` 허용)
- 경로: 하드코딩된 절대 경로 금지, `pathlib.Path` + `os.environ` 또는 CLI 인수 사용
- 에러 처리: 외부 API 호출은 `try/except` + `logging.error()` 필수
- 검색 클라이언트 반환값: T7에서 정의한 공통 스키마 준수

---

## 환경

- **실행 환경**: Docker + devcontainer (CUDA 12.1, Python 3.11)
- **의존성 설치**: `pip install -r requirements.txt`
- **API 키 설정**: `.env` 파일 또는 환경변수 (`GEMINI_API_KEY`, `OPENAI_API_KEY` 등)
- **GPU**: vLLM 사용 시 NVIDIA GPU 필수

---

## 목표 디렉토리 구조

```
scion_rag_challenge/
├── pipeline/
│   ├── run_pipeline.py        # 전체 파이프라인 단일 진입점
│   ├── step1_search.py        # 키워드 추출 + API 문서 검색
│   ├── step2_decompose.py     # 멀티홉 → 싱글홉 질문 분해 (선택 실행)
│   ├── step3_build_vectordb.py
│   ├── step4_retrieve.py
│   ├── step5_generate.py
│   ├── scienceon_api_example.py   # 수정 불가
│   ├── evaluate/
│   │   ├── eval_search.py     # Step 1 통합 평가 CLI
│   │   └── eval_answers.py    # Step 5 통합 평가 CLI
│   ├── core/                  # 키워드 추출기 구현체
│   ├── processors/
│   ├── utils/
│   └── __init__.py
│
├── src/
│   ├── retrieval_system/
│   ├── llm_client/
│   ├── search/                # 검색 클라이언트 공통 베이스/어댑터 (신규)
│   ├── evaluate/              # 평가 로직 + metrics (신규)
│   ├── features/
│   ├── data_handler/
│   ├── prompts/
│   ├── rerank/
│   └── utils/
│
├── configs/
│   ├── credentials/           # gitignored
│   ├── query_encoder/
│   └── csv_schema/
│
├── data/                      # gitignored
└── outputs/                   # gitignored
    ├── search/
    ├── vectordb/
    ├── retrieval/
    └── final/
```

---

---

## Refactoring Tasks (우선순위 순)

### 🔴 T1. 오타 수정 — `chellenge` → `challenge`

- `src/build_vectordb_search.py` 기본값 경로 문자열
- `configs/query_encoder/*.json` — `jsonl_path`, `output_file` 값

✅ `rg "chellenge" .` 결과 없음

---

### 🔴 T2. `.gitignore` 정리

아래 항목이 없으면 추가한다:
```
configs/credentials/*.json
.env
.DS_Store
outputs/
results/
data/
```
이미 Git 추적 중인 파일은 `git rm --cached <파일>` 로 추적 해제한다.

✅ `git status`에서 자격증명 파일이 나타나지 않음

---

### 🟠 T3. 디렉토리 구조 재편

1. `search_science_on_challenge/` → `pipeline/` 이름 변경
   ```bash
   git mv search_science_on_challenge pipeline
   ```

2. 파이프라인 래퍼 스크립트 생성 (`pipeline/step*.py` → `src/` 로직 호출). `sys.path.append` 해킹 제거.

3. `pipeline/run_pipeline.py` 신규 생성 — 5단계 순서대로 실행하는 단일 진입점:
   ```bash
   python pipeline/run_pipeline.py \
     --questions data/questions.jsonl \
     --encoder configs/query_encoder/config_bge_m3.json \
     --llm gemini \
     --output outputs/run_$(date +%Y%m%d)
   ```
   `step2_decompose`(멀티홉 분해)는 `--decompose` 플래그로 선택 실행 가능하도록 구현.

4. 파일 삭제 및 이동:
   - 삭제: `Dockerfile_old`, `docker-compose_old.yml`, `pipeline/core/keyword_extractor copy.py`, 루트의 `search_results_all_fields.json`
   - `pipeline/search_results_all_fields.json` → `outputs/search/`
   - `pipeline/scion_answer_docs.txt` → `data/`
   - `pipeline/test.csv` → `data/`
   - `src/documents_output.csv` → `outputs/`
   - `jobs.json` — 용도 확인 후 주석 추가 또는 삭제
   - `pipeline/README.md`, `pipeline/QUICK_START.md`, `pipeline/USAGE_GUIDE.md` → 루트 `README.md`에 통합 후 삭제

5. 파일명 오타 수정:
   ```bash
   git mv pipeline/Querry_Success_Rate.py pipeline/query_success_rate.py
   ```

6. 디렉토리 오타 수정:
   ```bash
   git mv configs/credientials configs/credentials
   ```

✅ `ls pipeline/` 에서 `step1~5.py`, `run_pipeline.py` 확인 / `search_science_on_challenge/` 디렉토리 없음

---

### 🟠 T4. 컨테이너 경로 통일 + 날짜 하드코딩 제거

- 루트 `Dockerfile`의 `WORKDIR /app`과 `docker-compose.yml`의 `.:/workspace` 혼재 → 둘 중 하나로 통일 (`/app` 권장)
- 날짜 포함 파일명 기본값 제거:
  ```bash
  rg "search_documents_2025" src/ configs/ -g "*.py" -g "*.json"
  ```
  Python 파일: `argparse` 필수 인수화 또는 `glob`으로 최신 파일 자동 탐색  
  JSON config: `jsonl_path`, `output_file` 필드에서 날짜 포함 경로 제거

✅ `rg "search_documents_2025" src/ configs/` 결과 없음

---

### 🟠 T5. PubMed 데드 코드 제거

`pipeline/pubmed_api_client.py` 에서:
- `# === 기존 History Server 방식 (사용 안 함) ===` 주석 이후 블록 삭제
- `_execute_searches()`, `_fetch_results()` — 호출처 없으면 삭제

```bash
rg "_execute_searches|_fetch_results" pipeline/   # 결과 없어야 삭제 진행
```

✅ `python -c "from pipeline.pubmed_api_client import PubMedAPIClient"` 에러 없음

---

### 🟡 T6. 평가 스크립트 통합

**목표 구조**:
```
pipeline/evaluate/
    eval_search.py    # Step 1 통합: --source scienceon|miracl_ko|miracl_en
    eval_answers.py   # Step 5 통합: --dataset pubmedqa|scifact|scienceon
src/evaluate/
    metrics.py        # normalize_title, calc_qsr, calc_bleu, calc_meteor
    eval_search.py    # 평가 로직
    eval_answers.py   # 평가 로직
```

**수행 작업**:
1. `src/evaluate/metrics.py` 신규 생성 — 공유 지표 함수 통합
2. `pipeline/evaluate/eval_search.py` 생성 — 기존 3개 스크립트 통합, 하드코딩 경로를 CLI 인수로 전환:
   ```bash
   python pipeline/evaluate/eval_search.py \
     --results outputs/search/search_results.json \
     --ground_truth data/miracl/questions/miracl_en_query_documents.json \
     --source miracl_en
   ```
3. `pipeline/evaluate/eval_answers.py` 생성 — 기존 2개 스크립트 통합:
   ```bash
   python pipeline/evaluate/eval_answers.py \
     --predictions results/final_answers/predictions.csv \
     --ground_truth data/questions/ground_truth.csv \
     --dataset pubmedqa
   ```
4. 기존 스크립트 삭제: `src/evaluate_pubmedqa.py`, `src/scifact_evaluation.py`, `pipeline/scion_search_evaluation.py`, `pipeline/wiki_search_evaluation_QSR.py`, `pipeline/query_success_rate.py`

✅ `python pipeline/evaluate/eval_search.py --help` 및 `eval_answers.py --help` 에러 없음

---

### 🟡 T7. 검색 클라이언트 공통 베이스 클래스 도입

`src/search/base_client.py` 신규 생성:
```python
from abc import ABC, abstractmethod

class BaseSearchClient(ABC):
    @abstractmethod
    def search(self, search_terms: list[str], max_results: int) -> list[dict]:
        """공통 반환 스키마:
        { "doc_id", "title", "abstract", "authors", "year", "url", "source" }
        source: "ScienceON" | "PubMed" | "Wikipedia"
        """
        ...
```

- `PubMedAPIClient`, `WikipediaAPIClient` → `BaseSearchClient` 상속으로 수정
- `ScienceONAPIClient`는 수정 불가 → `ScienceONAdapter(BaseSearchClient)` 래퍼 작성

✅ `python -c "from pipeline.search_meta_system import SearchMetaSystem"` 에러 없음

---

### 🟡 T8. `Integration` 클래스 순환 의존 제거

`PubMedIntegration`, `WikipediaIntegration`이 `self.system.keyword_extractor`를 역참조하는 구조를 제거한다.  
키워드 추출을 `SearchMetaSystem` 단에서 수행하고, 결과(`search_terms`)를 클라이언트에 전달:
```python
keywords = self.keyword_extractor.extract_keywords(query)
search_terms = self.keyword_extractor.generate_search_terms(keywords)
docs = self.pubmed_client.search(search_terms, max_results=50)
```

✅ `PubMedIntegration`, `WikipediaIntegration` 내부에 `self.system` 참조 없음

---

### 🟡 T9. 중복 제거 로직 통일

아래 4곳의 중복 제거 함수를 `src/utils/dedup.py`로 통합:
- `pipeline/core/document_searcher.py` — `_remove_duplicates`
- `pipeline/pubmed_api_client.py` — `_remove_duplicates_by_id`, `_remove_duplicates` (2개)
- `pipeline/wikipedia_api_client.py` — `_remove_duplicates_by_title`

```python
# src/utils/dedup.py
def remove_duplicates(docs: list[dict], key: str = "title") -> list[dict]: ...
```

✅ 위 4개 파일에서 기존 중복 제거 함수 삭제 후 `from src.utils.dedup import remove_duplicates` 호출

---

### 🟡 T10. Wikipedia 언어 하드코딩 제거

`wikipedia_api_client.py`의 `WikipediaAPIClient.__init__`에서:
```python
WIKI_BASE_URLS = {
    "ko": "https://ko.wikipedia.org/w/api.php",
    "en": "https://en.wikipedia.org/w/api.php",
}
def __init__(self, lang: str = "ko"):
    self.base_url = WIKI_BASE_URLS.get(lang, WIKI_BASE_URLS["en"])
```
`search_meta_system.py`에서 `WikipediaAPIClient` 생성 시 `--keyword-lang` 값을 전달한다.

✅ `wikipedia_api_client.py`에 `ko.wikipedia.org` 하드코딩 없음

---

## Factory 패턴 가이드

태스크 수행 시 아래 3개 팩토리를 함께 만든다:

```python
# pipeline/core/extractor_factory.py
def create_keyword_extractor(backend: str, config: dict) -> BaseExtractor: ...

# pipeline/core/search_client_factory.py  (T7과 함께)
def create_search_client(source: str, config: dict) -> BaseSearchClient: ...

# src/llm_client/llm_factory.py
def create_llm_client(backend: str, config: dict) -> BaseLLMClient: ...
```

`run_pipeline.py`에서 CLI 인수 `--llm`, `--sources`, `--extractor`를 받아 팩토리를 호출한다.

---

✅ **전체 통합 검증**: `python pipeline/run_pipeline.py --help` 에러 없음
