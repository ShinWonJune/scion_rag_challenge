# Repository 분석 보고서

> AGENTS.md 작성 전 기반 분석 자료

---

## 1. Repository 목적 및 기능 요약

### 핵심 목적

ScienceON 챌린지를 위한 **과학 문헌 RAG(Retrieval-Augmented Generation) 파이프라인** 구현.  
사용자의 (멀티홉) 질문을 받아 관련 과학 문헌을 검색·임베딩·재랭킹 후 LLM으로 최종 답변을 생성한다.

### 전체 파이프라인 흐름 (5단계)

```
[질문 입력]
    │
    ▼
[Step 1] 키워드 추출 + API 문서 검색        ← search_science_on_challenge/
    │  ScienceON / PubMed / Wikipedia API
    ▼
[Step 2] 멀티홉 → 싱글홉 질문 분해          ← src/multi_hop_to_single_hop.py
    │  Gemini / vLLM
    ▼
[Step 3] 벡터 DB 구축 (임베딩)              ← src/build_vectordb_search.py
    │  BGE-M3 / GTE-multilingual / Snowflake 등
    ▼
[Step 4] 밀집 검색 (Dense Retrieval)        ← src/retrieval_system/
    │  FAISS / NumPy 기반 ANN 검색
    ▼
[Step 5] 답변 생성 + 결과 집계              ← src/preprocess_and_generate_answer.py
    │  Gemini / vLLM / ChatGPT
    ▼
[CSV 제출 + BLEU/METEOR 평가]               ← src/final_result*.py, src/evaluate_pubmedqa.py
```

### 지원하는 데이터셋 / 태스크

| 태스크        | 관련 파일                                          |
|---------------|----------------------------------------------------|
| ScienceON     | `search_science_on_challenge/`, `src/final_result.py` |
| PubMedQA      | `src/final_result_pubmed.py`, `src/evaluate_pubmedqa.py` |
| SciFact       | `src/final_result_scifact.py`, `src/scifact_evaluation.py` |
| MIRACL (wiki) | `data/miracl/`, `search_science_on_challenge/miracl_wiki_compare/` |

### 지원하는 LLM 백엔드

- **Google Gemini** (기본) — `GEMINI_API_KEY` 환경변수
- **vLLM** (로컬 서버) — `--use-vllm` 플래그
- **ChatGPT / OpenAI** — `--use-chatgpt` 플래그

### 지원하는 임베딩 모델

`configs/query_encoder/` 아래 JSON 파일로 교체 가능:
- `BAAI/bge-m3` (기본)
- `Alibaba-NLP/gte-multilingual-base`
- `Snowflake/snowflake-arctic-embed-l`
- 그 외 HuggingFace 모델

---

## 2. 구조 및 코드 개선 제안

### 2-1. 보안 — 🔴 즉시 수정 필요

| 문제 | 현황 | 제안 |
|------|------|------|
| **실제 API 자격증명이 Git에 포함** | `configs/credientials/scienceon_api_credentials.json` 및 `search_science_on_challenge/configs/scienceon_api_credentials.json`가 커밋됨. 후자는 `.gitignore`에 있어도 이미 추적 중 | 하지만 토큰 키가 포함되어 있지만 중요한 키와 토큰이 아니기 때문에 gitignore 수준에서 정리하는걸로. |
| **Gemini/ChatGPT 키 관리 방식 혼재** | `GEMINI_API_KEY`/`OPENAI_API_KEY` 환경변수와 JSON 파일(`configs/*_api_credentials.json`) 혼재 | `.env` 단일 방식으로 통일하고 JSON은 `.example`만 유지 |
| **PubMed 자격증명 파일명 불일치** | 코드: `./configs/pubmed_credentials.json` / `.gitignore`: `pubmed_api_credentials.json` | 파일명/경로 통일 (문서·코드·gitignore 동일 명칭) |

```bash
# 추가 권장 .gitignore 항목
configs/credientials/*.json
search_science_on_challenge/configs/*.json
.env
*.csv          # 출력 데이터
outputs/
results/
data/
.DS_Store
```

---

### 2-2. 디렉토리 구조 — 🟠 정리 필요

#### 현재 구조의 문제점

```
scion_rag_challenge/
├── search_science_on_challenge/   # (1) 독립 실행형 패키지처럼 구성
│   ├── main.py
│   ├── outputs/                   # (2) 흩어진 outputs
│   └── configs/                   # (3) 루트 configs와 중복
└── src/                           # (4) 또 다른 독립 패키지
    ├── outputs/                   # (2) 흩어진 outputs
    └── ...
```

- **`search_science_on_challenge/`와 `src/`가 각각 독립 패키지처럼 동작** — 공유 유틸은 없고, `sys.path.append()` 해킹으로 연결
- **`outputs/` 디렉토리가 루트·src·search_science_on_challenge 세 곳에 분산**
- **`configs/`가 루트와 `search_science_on_challenge/configs/` 두 곳에 존재**
- **`results/`에 날짜별 결과 디렉토리가 Git에 추적됨**

#### 제안 구조

```
scion_rag_challenge/
├── .env.example                   # 환경변수 예시 (필수)
├── .gitignore                     # 데이터/출력/자격증명 제외
├── README.md                      # 개선된 README, 나중에 따로 작성할 예정
├── AGENTS.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
│
├── configs/                       # 단일 configs 디렉토리
│   ├── credentials/               # (오타 수정: credientials → credentials)
│   │   └── scienceon_api_credentials.json.example
│   ├── query_encoder/
│   └── csv_schema/
│
├── data/                          # 입력 데이터 (gitignored)
│   ├── questions/
│   └── ...
│
├── outputs/                       # 모든 출력 (gitignored)
│   ├── search/
│   ├── vectordb/
│   ├── retrieval/
│   └── final/
│
├── pipeline/                      # ← search_science_on_challenge/ 이름 변경 + 통합
│   ├── step1_search.py            # 기존 main.py
│   ├── step2_decompose.py         # 기존 multi_hop_to_single_hop.py. Optional 하도록 반영
│   ├── step3_build_vectordb.py    # 기존 build_vectordb_search.py
│   ├── step4_retrieve.py          # 기존 retrieval_system/main.py
│   ├── step5_generate.py          # 기존 preprocess_and_generate_answer.py
│   └── run_pipeline.py            # ← 신규: 전체 파이프라인 실행 진입점
│
└── src/                           # 공유 라이브러리
    ├── retrieval_system/
    ├── llm_client/
    ├── features/
    ├── data_handler/
    ├── prompts/
    ├── utils/
    └── ...
```

---

### 2-3. 하드코딩된 경로 — 🟠 수정 필요

#### 문제

여러 파일에서 절대 경로가 하드코딩되어 있어 다른 머신에서 실행 불가  
(예: `src/*.py`, `configs/query_encoder/*.json`, `README.md`, `main.ipynb` 등 전반):

```python
# src/build_vectordb_search.py
docs_jsonl_path="/app/search_science_on_chellenge/outputs/search_documents_20250912_013206.jsonl"
#                        ^^^^^^^^^^^^^^^ 오타: chellenge

# configs/query_encoder/config_bge_m3.json
"jsonl_path": "/app/search_science_on_chellenge/outputs/search_documents_20250912_013206.jsonl"
# 특정 날짜의 파일 이름이 하드코딩됨
```

#### 제안

- 모든 경로를 CLI 인수 또는 `.env` 파일에서 주입
- config JSON에서 날짜가 포함된 특정 파일명 제거 (빌드 시 자동 탐색)
- `WORKSPACE_ROOT` 환경변수 하나로 기준 경로 통일
- **경로 해석 유틸**(예: `src/utils/paths.py`)을 두고 `/app`, `/workspace` 문자열을 제거

---

### 2-3-1. Docker 환경 구성 분석 및 통합/분리 판단

#### 두 Docker 설정의 용도 비교

현재 레포에 Docker 관련 파일이 **두 곳**에 존재한다:

| 항목 | 루트 (`Dockerfile` + `docker-compose.yml`) | `.devcontainer/` (`Dockerfile` + `docker-compose.yml` + `devcontainer.json`) |
|---|---|---|
| **베이스 이미지** | `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` (3GB+) | `python:3.11-slim` (경량) |
| **GPU** | ✅ NVIDIA GPU 필수, shm_size 64g | ❌ GPU 없음 |
| **vLLM** | ✅ `vllm==0.10.2`, `torch==2.8.0` 설치 | ❌ 미설치 |
| **WORKDIR** | `/app` | `/workspace` |
| **소스코드** | 이미지 안에 `COPY . .` → 이미지에 포함 | 호스트에서 volume mount (`.:/workspace`) |
| **사용자** | root | `vscode` (비루트) |
| **용도** | vLLM 추론 + RAG 파이프라인 **실행용** | VS Code devcontainer **개발 환경** |
| **포트** | `8004:8000` (vLLM 서버) | `6001` (Jupyter) |
| **env 파일** | ❌ 없음 | ✅ `../.env` 자동 주입 |

**결론: 두 설정은 목적이 완전히 달라 분리가 맞다.** 통합하면 개발 환경에 불필요하게 vLLM과 CUDA가 포함된 이미지를 쓰게 되어 빌드 시간과 용량이 크게 증가한다.

---

#### T4 절대 경로 문제의 재해석

 `/app`, `/workspace`는 **컨테이너 실행 환경을 전제한 의도적 경로**다.  
하지만 현재 코드에는 두 가지 다른 종류의 경로 문제가 섞여 있다:

**문제 A — 컨테이너 내 경로가 두 가지로 혼재 (진짜 버그)**

루트 `Dockerfile`과 `docker-compose.yml`을 비교하면:
```dockerfile
# 루트 Dockerfile
WORKDIR /app           # ← 이미지의 작업 디렉토리는 /app

# 루트 docker-compose.yml
volumes:
  - .:/workspace       # ← 그런데 소스는 /workspace에 마운트
```

즉 루트 컨테이너에서는 `/app`(이미지 내 복사본)과 `/workspace`(마운트된 호스트 소스)가 동시에 존재하며 **서로 다른 내용**일 수 있다.  
한편 devcontainer의 WORKDIR은 `/workspace`다.

결과적으로 코드에서 `/app`을 참조하는 곳은 루트 컨테이너 전용이고, `/workspace`를 참조하는 곳은 devcontainer 전용인데, 두 경로가 같은 파일 안에 뒤섞여 있다.

**문제 B — 특정 날짜 파일명 하드코딩 (진짜 문제)**

```python
docs_jsonl_path="/app/search_science_on_chellenge/outputs/search_documents_20250912_013206.jsonl"
#                                                                           ^^^^^^^^^^^^^^^^^^^^^^^^
#                                                          이 파일은 2025-09-12에 생성된 특정 출력 파일
```

이건 컨테이너 여부와 무관하게, **다음 실행에서는 파일명이 바뀌므로 항상 실패**한다.

#### 제안

| 문제 | 수정 방향 |
|---|---|
| `/app` vs `/workspace` 혼재 | 루트 `docker-compose.yml`의 volume mount를 `/app`으로 통일하거나, 루트 `Dockerfile`의 WORKDIR을 `/workspace`로 통일 |
| 특정 날짜 파일명 하드코딩 | CLI 인수(`--docs_jsonl_path`)로 주입, 기본값은 최신 파일 자동 탐색(`glob` 최신순) |
| devcontainer `.env` 미적용 | 루트 `docker-compose.yml`에도 `env_file: .env` 추가 |

---

### 2-4. 오타 수정 — 🟡 소소하지만 중요

| 현재 | 수정 후 | 위치 |
|------|---------|------|
| `search_science_on_chellenge` | `search_science_on_challenge` | `src/build_vectordb_search.py`, `configs/*.json` 여러 곳 |
| `configs/credientials/` | `configs/credentials/` | 디렉토리명 |
| `retrival` (키 이름) | `retrieval` | `src/final_result.py` L33 |
| `pumbedqa_final_answers` | `pubmedqa_final_answers` | README 문구 (코드/폴더는 정상) |

---

### 2-5. 파이프라인 진입점 부재 — 🟠 신규 파일 필요

#### 문제

현재 파이프라인을 실행하려면 사용자가 README를 읽고 11개 단계를 직접 순서대로 실행해야 한다.  
각 단계 사이에 경로를 손으로 복사해야 하는 곳이 2~3군데 있다.

#### 제안: `run_pipeline.py` 생성

```python
# 예시 인터페이스
python run_pipeline.py \
  --questions data/questions.jsonl \
  --dataset scienceon \          # scienceon | pubmedqa | scifact
  --encoder configs/query_encoder/config_bge_m3.json \
  --llm gemini \                  # gemini | vllm | chatgpt
  --output outputs/run_$(date)
```

---

### 2-6. 불필요한 파일 정리 — 🟡

| 파일 | 상태 | 처리 제안 |
|------|------|-----------|
| `Dockerfile_old` | 구버전 백업 | 삭제 (git history에 남아 있음) |
| `docker-compose_old.yml` | 구버전 백업 | 삭제 |
| `search_science_on_challenge/core/keyword_extractor copy.py` | 이름에 공백 포함, 복사본 | 삭제 |
| `src/documents_output.csv` | 출력 데이터가 src/에 존재 | `outputs/`로 이동 후 gitignore |
| `search_results_all_fields.json` | 루트와 `search_science_on_challenge/` 양쪽에 중복 존재 | 하나 삭제 |
| `jobs.json` | 용도 불명확, 문서화 없음 | 용도 명시 또는 삭제 |
| `test.zip` | 루트에 있는 테스트 압축 파일 | `data/`로 이동 또는 gitignore |
| `.DS_Store` | macOS 메타 파일이 추적됨 | 삭제 + `.gitignore` 추가 |

---

### 2-7. 환경 설정 문서화 — 🟡

#### 문제

- README의 환경 설정 섹션이 `"this use docker compose and devcontainer for ~~~"` 처럼 미완성
- API 키를 어디서 발급하고 어디에 넣는지 명확하지 않음
- GPU 요구사항 명시 없음

#### 제안: `.env.example` 추가

```bash
# .env.example

# === LLM API Keys ===
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here   # ChatGPT 사용 시

# === ScienceON API ===
SCIENCEON_AUTH_KEY=your_auth_key
SCIENCEON_CLIENT_ID=your_client_id

# === 경로 설정 ===
WORKSPACE_ROOT=/workspace
OUTPUT_DIRECTORY=./outputs
TARGET_DOCUMENTS=50

# === vLLM 설정 (선택) ===
VLLM_SERVER_URL=http://localhost:8000/v1
VLLM_MODEL_NAME=openai/gpt-oss-120B
```

---

### 2-8. 테스트 부재 — 🟢 중장기 개선

- `src/tests/`에 `test_history.py` 하나만 존재
- 각 모듈(retrieval, embedding, LLM 호출)에 대한 단위 테스트 없음
- 검색 품질 평가(`QSR`, `BLEU`, `METEOR`)는 있으나, CI에 연결되지 않음

---

### 2-9. 레포 위생 및 데이터 추적 — 🟠 단기

- `results/`, `outputs/`, `data/`가 실제로 Git에 대량 추적됨  
  → 용량/재현성 모두 악화. **gitignore 추가 + tracked 파일 제거** 필요.
- 결과물 및 임베딩 산출물이 `configs/query_encoder/*.json`에 기록(`output_file`, `last_run`)  
  → 설정과 실행 아티팩트가 섞여 있음. **런타임 기록은 별도 로그/메타 파일로 분리** 권장.

---

### 2-10. 검색 클라이언트 통일성 부재 — 🟠 리팩토링 필요

> 대상: `search_science_on_challenge/` 내 `pubmed_api_client.py`, `wikipedia_api_client.py`, `scienceon_api_example.py`, `search_meta_system.py`  
> `scienceon_api_example.py`는 대회 제공 파일로 수정 불가 — 리팩토링 범위 제외.

#### (a) 공통 인터페이스 없음

세 클라이언트가 각자 다른 메서드 시그니처를 가지며 서로 교체 불가:

| 클라이언트 | 검색 메서드 | 입력 |
|---|---|---|
| `ScienceONAPIClient` | `search_articles(query, cur_page, row_count, fields)` | 단일 쿼리 문자열 |
| `PubMedAPIClient` | `search_multiple_terms(search_terms, max_terms)` | 검색어 리스트 |
| `WikipediaAPIClient` | `search_multiple_terms(search_terms, max_results_per_term)` | 검색어 리스트 |

PubMed와 Wikipedia는 동일한 메서드명(`search_multiple_terms`)이지만 파라미터명이 다르다 (`max_terms` vs `max_results_per_term`). 공통 추상 베이스 클래스가 없어 `DocumentSearcher`가 ScienceON 전용으로 고정되어 있고, PubMed/Wikipedia는 별도 `Integration` 클래스를 통해 우회한다.

**제안**: 공통 베이스 클래스 정의

```python
# src 또는 search_science_on_challenge/core/base_client.py
from abc import ABC, abstractmethod

class BaseSearchClient(ABC):
    @abstractmethod
    def search(self, search_terms: list[str], max_results: int) -> list[dict]:
        """검색어 리스트를 받아 정규화된 문서 리스트 반환"""
        ...
```

---

#### (b) 출력 스키마 불일치

세 클라이언트가 반환하는 딕셔너리 키가 제각각이며, 하류(downstream) 코드가 출처별로 분기 처리해야 하는 상황:

| 필드 의미 | ScienceON | PubMed | Wikipedia |
|---|---|---|---|
| 문서 ID | `CN` | `id` | _(없음)_ |
| 저자 | `author` | `authors` | `authors` |
| 발행연도 | `year` | `publication_year` | `publication_year` |
| 본문/초록 | `abstract` | `abstract`, `content` | `abstract` (가짜값) |
| URL | `link` | `url` | `url` |
| 출처 표시 | _(없음)_ | `source: 'PubMed'` | `source: 'Wikipedia'` |

특히 Wikipedia의 `abstract`는 실제 내용이 아닌 `"Wikipedia article about {title}"`로 하드코딩된 가짜값이다.

**제안**: 정규화된 공통 스키마 정의

```python
{
    "doc_id": str,          # 고유 ID
    "title": str,
    "abstract": str,        # 실제 초록/본문 요약
    "authors": str,         # 저자 (단수/복수 통일)
    "year": str,            # 발행연도 (필드명 통일)
    "url": str,
    "source": str,          # "ScienceON" | "PubMed" | "Wikipedia"
    "raw": dict             # 원본 응답 (디버깅용)
}
```

---

#### (c) `Integration` 클래스의 순환 의존성

`PubMedIntegration`과 `WikipediaIntegration` 모두 생성자에서 `search_meta_system` 인스턴스를 받아 내부에서 `self.system.keyword_extractor`를 직접 호출한다.  
`SearchMetaSystem`이 `Integration` 객체를 생성하고, `Integration`이 다시 `SearchMetaSystem`을 참조하는 **순환 의존 구조**다.

```python
# search_meta_system.py __init__
self.pubmed_integration = PubMedIntegration(self, ...)      # self를 넘김
self.wikipedia_integration = WikipediaIntegration(self)     # self를 넘김

# pubmed_api_client.py PubMedIntegration.search_with_pubmed
keywords = self.system.keyword_extractor.extract_keywords(query)  # 다시 system 접근
```

**제안**: `Integration` 클래스에서 `system` 참조를 제거하고, 키워드 추출 결과를 외부에서 주입(DI):

```python
# 변경 후 호출부 (search_meta_system.py)
keywords = self.keyword_extractor.extract_keywords(query)
search_terms = self.keyword_extractor.generate_search_terms(keywords)
docs = self.pubmed_client.search(search_terms, max_results=50)
```

---

#### (d) 키워드 추출 + skip 로직 중복

아래 블록이 `PubMedIntegration.search_with_pubmed`와 `WikipediaIntegration.search_with_wikipedia`에 **그대로 복붙**되어 있다:

```python
if hasattr(self.system, 'skip_keyword_extraction') and self.system.skip_keyword_extraction:
    keywords = {"english": [query], "korean": []}
    search_terms = [query]
else:
    keywords = self.system.keyword_extractor.extract_keywords(query)
    search_terms = self.system.keyword_extractor.generate_search_terms(keywords)
```

(c)의 제안대로 키워드 추출을 `SearchMetaSystem` 단에서 한 번만 수행하면 자연스럽게 해결된다.

---

#### (e) 중복 제거 로직이 4곳에 분산

`_remove_duplicates` 또는 유사 함수가 아래 파일에 각각 존재한다:

| 파일 | 함수명 | 중복 기준 |
|---|---|---|
| `core/document_searcher.py` | `_remove_duplicates` | 제목 |
| `pubmed_api_client.py` `PubMedAPIClient` | `_remove_duplicates_by_id` | ID |
| `pubmed_api_client.py` `PubMedIntegration` | `_remove_duplicates` | 제목 |
| `wikipedia_api_client.py` | `_remove_duplicates_by_title` | 제목 |

**제안**: `utils/dedup.py` 하나에 통일된 함수로 이동.

---

#### (f) PubMed 미사용 코드 (데드 코드)

`pubmed_api_client.py`에 `# === 기존 History Server 방식 (사용 안 함) ===` 주석 이후, 실행 불가 상태의 코드 블록이 약 80줄 남아 있다 (문서 문자열처럼 처리되어 에러 없이 무시되고 있음). `_execute_searches`, `_fetch_results` 메서드도 현재 어디서도 호출되지 않는 데드 코드다.

**제안**: 해당 블록 전체 삭제. History 방식이 필요하면 git 히스토리에서 복구 가능.

---

#### (g) Wikipedia URL이 언어와 무관하게 한국어로 고정

`WikipediaAPIClient`는 `base_url = "https://ko.wikipedia.org/w/api.php"`로 하드코딩되어 있어, 영어 키워드로 검색해도 한국어 위키 API에 요청이 간다. `--keyword-lang english` 옵션을 쓰는 MIRACL 영어 태스크에서 잘못된 결과를 반환할 가능성이 있다.  
(README에도 이 문제를 인지하고 `--keyword-lang` 옵션으로 우회하도록 설명되어 있으나, 클라이언트 내부에서 해결이 필요하다.)

**제안**: 생성자에서 `lang: str = "ko"` 파라미터를 받아 URL을 동적으로 결정.

```python
WIKI_BASE_URLS = {"ko": "https://ko.wikipedia.org/w/api.php", "en": "https://en.wikipedia.org/w/api.php"}
self.base_url = WIKI_BASE_URLS.get(lang, WIKI_BASE_URLS["en"])
```

---

## 3. 우선순위 요약

| 우선순위 | 항목 | 난이도 |
|----------|------|--------|
| 🔴 즉시 | 자격증명 파일 `.gitignore`로 정리 | 중간 |
| 🔴 즉시 | `search_science_on_chellenge` 오타 일괄 수정 | 낮음 |
| 🟠 단기 | 불필요한 백업 파일 (`_old`, `copy`) 삭제 | 낮음 |
| 🟠 단기 | 하드코딩된 절대 경로를 환경변수/CLI 인수로 전환 | 중간 |
| 🟠 단기 | `run_pipeline.py` 전체 파이프라인 진입점 생성 | 중간 |
| 🟡 중기 | `search_science_on_challenge/`와 `src/`의 역할 경계 명확화 | 높음 |
| 🟡 중기 | `outputs/` 디렉토리 단일화 | 중간 |
| 🟡 중기 | `results/`·`outputs/`·`data/` Git 추적 제거 + .gitignore 정리 | 중간 |
| 🟡 중기 | PubMed 데드 코드 제거 (미사용 History Server 방식, ~80줄) | 낮음 |
| 🟡 중기 | 검색 클라이언트 공통 스키마 + 베이스 클래스 도입 | 높음 |
| 🟡 중기 | `Integration` 클래스 순환 의존 제거 + 중복 키워드 추출 로직 통합 | 중간 |
| 🟡 중기 | Wikipedia 언어 하드코딩 제거 (`lang` 파라미터화) | 낮음 |
| ��� 장기 | 단위 테스트 추가 + CI 설정 | 높음 |
