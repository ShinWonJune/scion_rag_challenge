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
| **실제 API 자격증명이 Git에 포함** | `configs/credientials/scienceon_api_credentials.json`에 `access_token`, `client_id`, `auth_key` 등 민감 정보가 평문으로 저장되어 있으며 `.gitignore`에 없음 | 즉시 `git rm --cached` 후 `.gitignore`에 추가. `.env` 파일로 관리하고 `.env.example`을 제공 |
| **Gemini API 키 관리 방식 불명확** | 코드 내 `GEMINI_API_KEY` 환경변수 참조와 JSON 파일 방식이 혼재 | `.env` 단일 방식으로 통일 |

```bash
# 추가 권장 .gitignore 항목
configs/credientials/*.json
.env
*.csv          # 출력 데이터
outputs/
results/
data/
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
├── README.md                      # 개선된 README
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
│   ├── step2_decompose.py         # 기존 multi_hop_to_single_hop.py
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

여러 파일에서 절대 경로가 하드코딩되어 있어 다른 머신에서 실행 불가:

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

---

### 2-4. 오타 수정 — 🟡 소소하지만 중요

| 현재 | 수정 후 | 위치 |
|------|---------|------|
| `search_science_on_chellenge` | `search_science_on_challenge` | `src/build_vectordb_search.py`, `configs/*.json` 여러 곳 |
| `configs/credientials/` | `configs/credentials/` | 디렉토리명 |
| `retrival` (키 이름) | `retrieval` | `src/final_result.py` L33 |
| `pumbedqa_final_answers` | `pubmedqa_final_answers` | `results/` 디렉토리명 |

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

## 3. 우선순위 요약

| 우선순위 | 항목 | 난이도 |
|----------|------|--------|
| 🔴 즉시 | 자격증명 파일 gitignore 처리 + `.env` 방식으로 전환 | 낮음 |
| 🔴 즉시 | `search_science_on_chellenge` 오타 일괄 수정 | 낮음 |
| 🟠 단기 | 불필요한 백업 파일 (`_old`, `copy`) 삭제 | 낮음 |
| 🟠 단기 | 하드코딩된 절대 경로를 환경변수/CLI 인수로 전환 | 중간 |
| 🟠 단기 | `run_pipeline.py` 전체 파이프라인 진입점 생성 | 중간 |
| 🟡 중기 | `search_science_on_challenge/`와 `src/`의 역할 경계 명확화 | 높음 |
| 🟡 중기 | `outputs/` 디렉토리 단일화 | 중간 |
| 🟢 장기 | 단위 테스트 추가 + CI 설정 | 높음 |
