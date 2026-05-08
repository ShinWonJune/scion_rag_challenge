# SHRAG: Search like Human with RAG

> 🏆 **RDGENAI2025(CIKM Workshop) ScienceON Challenge 1등 프레임워크**

> **자연어 질의를 학술 검색 플랫폼이 이해할 수 있는 Boolean keyword query로 자동 변환하여 검색하는 RAG 프레임워크**

> 발표 논문: `shrag.pdf` — "SHRAG: A Framework for Combining Human-Inspired Search with RAG" (Hyunseok Ryu, Wonjune Shin, Hyun Park / GIST AI Convergence)

---

## 1. 핵심 아이디어 — LLM-as-Boolean-Query-Generator

학술 검색 플랫폼(ScienceON, PubMed 등)은 전통적으로 **Boolean operator (AND / OR / NOT)** 기반 keyword 검색을 지원한다. 하지만 일반 사용자에게 자연어 질의를 Boolean query로 변환하는 것은 생소하며, 자연어 그대로 검색 엔진에 넣으면 recall이 낮다.

SHRAG는 자연어 질의를 기반으로 학술 검색 문헌을 근거로 하는 답변을 생성하는 RAG 프레임워크이다. **LLM을 Query Strategist로 사용**해 자연어 질의를 structured search query로 자동 변환하고, 이를 기존 학술 검색 엔진에서 검색하여 문서를 확보한다. 이후 multilingual dense retrieval과 re-ranking(optional)을 거쳐 LLM이 답변을 생성한다.

```
자연어 질문 Q
   │
   ▼
[LLM Query Strategist]              ← 본 프로젝트 핵심
   ├─ multilingual keyword extraction (KO + EN)
   ├─ compound keyword splitting
   ├─ importance-based ranking → top-10
   └─ Strategic OR-query generation (n=10 → 1)
   │
   ▼
[학술 검색 엔진 — Boolean retrieval]
   ScienceON / PubMed / Wikipedia (plug-and-play)
   │
   ▼
[Multilingual dense retrieval — mGTE]
   query·doc 임베딩 cosine similarity → top-K
   │
   ▼ (선택)
[Cross-encoder re-ranking — dragonkue/bge-reranker-v2-m3-ko]
   (query, doc) per-pair score → top-K'
   │
   ▼
[Generative LLM]
   structured answer (Title / Introduction / Main Body)
```

### 1.1 왜 OR-only인가?

실험: AND 0개 ~ 9개를 변화시키며 ScienceON 50 queries × 10회 측정.

| Operator | 수집 문서 수 | gold 포함률 |
|---|---:|---:|
| OR-only (AND=0) | ~1,500 | **best** |
| OR + AND≥1 | ~2,000+ | 점점 낮아짐 |

→ AND를 추가하면 더 많은 docs를 collect하지만 **specific keywords에 over-emphasize**해서 핵심 주제에서 벗어남. RAG는 "정확한 몇 개"보다 **"gold가 corpus 안에 들어 있을 확률"** 이 중요하므로 OR로 넓게 잡고 dense retrieval로 좁히는 전략 채택.

### 1.2 왜 Strategic (progressive narrowing)인가?

키워드 10개를 한 번에 OR로 묶은 query 하나만 보내는 게 아니라, **n=10부터 n=1까지 점진적으로 좁혀가는 search query set**을 생성:

```
SQ_10 = k1 ∨ k2 ∨ ... ∨ k10    ← 가장 넓음
SQ_9  = k1 ∨ k2 ∨ ... ∨ k9
...
SQ_1  = k1                      ← 가장 핵심 키워드만
```

키워드는 importance로 정렬되어 있으므로 뒤 키워드를 하나씩 떼어내면 **diversity는 유지하되 thematic consistency 보존**. 각 SQ_n에서 top-10 docs 수집 후 union + dedup. 검색자(human)가 처음엔 broad query → 결과 부족하면 narrow query로 좁혀가는 행동을 모사.

### 1.3 왜 multilingual?

ScienceON / Wikipedia 같은 비영어권 학술 사이트는 한 문서 안에서도 한국어와 영어가 섞임 (특히 고유명사, 학술 용어). SHRAG는:
1. **언어별 keyword extraction prompt** 분리 (KO와 EN 따로 LLM 호출)
2. **complementary keyword set** 생성 (KO 키워드로 한글 docs, EN 키워드로 영어 docs 모두 cover)
3. **multilingual embedding** (`Alibaba-NLP/gte-multilingual-base`)로 cross-lingual cosine similarity

→ KO 질문 + EN gold 또는 EN 질문 + KO gold 같은 cross-lingual 케이스도 retrieval 가능.

---

## 2. 5-step Pipeline


| Step | 입력 | 출력 | 핵심 |
|---|---|---|---|
| **1. Multilingual Keyword Extraction** | 자연어 질문 Q | top-10 keywords {k1...k10} (importance 정렬) | LLM에 KO / EN 각각 prompt → compound 단어 split → importance rank |
| **2. Strategic Query Generation** | keywords | search query set {SQ_1, ..., SQ_10} | 진행적 OR-query, n 줄여가며 narrowing |
| **3. Document Search & Collection** | search query set | corpus D (dedup) | 각 SQ에서 top-10 → union → dedup |
| **4. Multilingual Embedding & Re-ranking** | Q, D | top-5 docs | mGTE 임베딩 → cosine similarity → top-K |
| **5. Structured Answer Generation** | Q + top-5 docs | Title / Intro / Body | LLM (vLLM gpt-oss-20b 또는 Gemini) |

### 코드 매핑

```
shrag/pipeline/steps/step1_search.py     ← Step 1+2+3 (extractor → search_terms → ScienceON 호출 → dedup)
shrag/pipeline/steps/step2_decompose.py  ← (옵션) multi-hop 분해
shrag/pipeline/steps/step3_build_vectordb.py  ← Step 4 전반부 (doc embedding)
shrag/pipeline/steps/step4_retrieve.py   ← Step 4 후반부 (query embed + cosine top-K)
shrag/pipeline/steps/step5_generate.py   ← Step 5
```

---

## 3. Architecture

```
shrag/
├── pipeline/                # 5-step 오케스트레이션
│   ├── run.py               # 통합 entrypoint (python -m shrag.pipeline.run)
│   ├── steps/               # step1 ~ step5
│   └── _impl/               # build_vectordb 등 구현
├── search/
│   ├── extractors/          # keyword extraction + Strategic OR-query 생성
│   ├── factories/           # source 별 client factory
│   ├── clients/             # ScienceON / PubMed / Wikipedia adapter
│   ├── cache.py             # SHA1 키 HTTP 응답 캐시
│   └── throttle.py          # rate limiting + 429 retry
├── retrieval/               # FAISS / cosine retrieval
├── features/                # 임베딩 인코더 (gte / bge-m3 / e5 / jina 등)
├── llm/                     # OpenAI / Gemini / vLLM 클라이언트
├── data_handler/, csv_headers/, evaluation/, utils/
└── prompts/                 # extractor / generator 프롬프트

experiments/
├── shared/                  # 공통 retrieval/rerank/eval 모듈
├── exp{1,3,4,7,8,...,18}*  # 실험별 산출물
├── BEST_CONFIG.md           # ⭐ 권장 setting 1-page 요약
└── outputs/
    ├── FOLLOWUP_EXPERIMENT_SUMMARY.md   # ⭐ 전체 실험 종합 보고서
    └── ...

configs/
├── query_encoder/           # encoder별 JSON config (cache_db_path 포함)
├── csv_schema/              # vector DB 스키마
└── credentials/             # API 자격증명 (gitignored)
```

---

## 4. Quick Start

### 4.1 환경 설정

```bash
pip install -r requirements.txt
cp .env.example .env  # API 키 채우기
```

`.env` 필수:
- `OPENAI_API_KEY` 또는 `GOOGLE_API_KEY` (extractor용)
- `GPT_OSS_VLLM_URL`, `GPT_OSS_VLLM_MODEL` (vLLM 사용 시)
- `SCIENCEON_CREDENTIALS_PATH` (default `configs/credentials/scienceon_api_credentials.json`)

### 4.2 vLLM 서버 (옵션)

```bash
docker compose up -d
curl http://localhost:8004/v1/models
```

### 4.3 E2E 실행 (권장 setting)

```bash
python -m shrag.pipeline.run \
  --questions data/test.csv \
  --encoder configs/query_encoder/config_gte-multilingual-base.json \
  --extractor vllm \
  --llm vllm \
  --sources scienceon \
  --vllm-url http://localhost:8004/v1 \
  --vllm-model openai/gpt-oss-20b \
  --target-documents 50 \
  --top-k 50 \
  --max-rank 5
```

질문 분해 사용 시 `--decompose`.

---

## 5. 핵심 기능

### 5.1 LLM 기반 Multilingual Keyword Extraction

`shrag/search/extractors/` 의 keyword extractor가 query Q에 대해:

1. **언어별 분리 prompt** — KO와 EN을 각각 호출 (prompt 언어와 output 언어 일치 시 성능·일관성이 더 좋음)
2. **Top-K keyword 생성** — 각 언어에서 importance 순으로 K개
3. **Compound split** — "free textbook" 같은 두 단어 이상은 분리 (검색 성능 저하 회피)
4. **Importance ranking + top-10 cap** — 분리 후 합친 keyword 중 상위 10개

지원 backend: `vllm` (gpt-oss-20b 등), `chatgpt`/`openai`, `gemini`. 결정론(`temperature=0`) 권장.

### 5.2 Strategic Search Query Generation (Boolean OR-query)

`shrag/search/extractors/search_terms.py`에서 keyword set으로 search query set 생성:

```
search_terms = [
  "k1|k2|k3|k4|k5|k6|k7|k8|k9|k10",   # n=10 (broad)
  "k1|k2|k3|k4|k5|k6|k7|k8|k9",        # n=9
  ...
  "k1|k2",                              # n=2
  "k1",                                 # n=1 (narrow, 핵심 키워드만)
]
```

ScienceON `searchQuery`는 `BI:"k1|k2|..."` 형태의 OR-string. 한국어와 영어를 각각 별도 query set으로 생성.

### 5.3 Frozen queries — 재현성 + 가속

같은 질문 set에 대해 키워드 / search_terms를 한 번 생성하고 JSONL로 동결하면, 이후 step1은 vLLM 호출 없이 ScienceON 호출만 수행 → 실험 N회 반복 시 결과 결정성 + 시간 절약.

```bash
# 1) 키워드만 생성
python -m shrag.pipeline.steps.step1_search \
  --questions data/test.csv \
  --extractor vllm --vllm-url http://localhost:8004/v1 --vllm-model openai/gpt-oss-20b \
  --extractor-temperature 0 \
  --emit-frozen-queries outputs/frozen/queries.jsonl

# 2) frozen queries로 검색 (vLLM 호출 없음)
python -m shrag.pipeline.steps.step1_search \
  --questions data/test.csv \
  --frozen-queries outputs/frozen/queries.jsonl \
  --target-documents 50 \
  --output-dir outputs/search_run1
```

### 5.4 Document Search & Dedup

`shrag/search/clients/scienceon_api_example.py` (ScienceON adapter):
- 각 search query에서 max_pages × row_count = 최대 docs 수집 (default 5 × 10 = 50)
- 응답 caching: SHA1 키 (source, term, page, row_count) → `outputs/_shared_cache/scienceon/`
- Throttle: `min_interval_sec`, `max_concurrency`, 429 retry with backoff
- target_documents (m) 도달 시 stop

수집 후 `shrag/utils/dedup.py`로 doc_id 기반 dedup.

**(k, n, m) 변수 채택 근거** (Exp1 sweep — k = SCIENCEON_ROW_COUNT × max_pages, n = 검색어 수, m = dedup 후 cap):

| 변수 | 채택값 | 근거 |
|---|---:|---|
| k | **30** | k=30이 k=50 대비 동등 Hit@5 + 노이즈 적음. k=10은 gold_found_rate 손실 |
| n | **10** | Strategic narrowing(n=10→1)이 어떤 단일 n보다 broad query coverage 우수 |
| m | **50** | m=30부터 Hit@5 plateau, m=50은 안전 마진. m=70 추가 이득 0. m=10은 search recall 급락 (Hit@5 0.683) |

상세 sweep 결과: [`experiments/outputs/exp1_kmn/PROGRESS.md`](experiments/outputs/exp1_kmn/PROGRESS.md), [`experiments/outputs/FOLLOWUP_EXPERIMENT_SUMMARY.md`](experiments/outputs/FOLLOWUP_EXPERIMENT_SUMMARY.md).

### 5.5 Multilingual Embedding + dense retrieval

`Alibaba-NLP/gte-multilingual-base` (768d, mGTE) 채택 근거:
- MMTEB benchmark 기준 KO+EN 우수
- 8,000 토큰 long-context 지원
- Snowflake / Jina / BGE 후보 중 GTE가 query–gold cosine 가장 높음 (BGE와 동률이나 자원 효율에서 우위)

ScienceON 41Q에서 **gte vs bge-m3** 비교 (Exp3/Exp16/Exp18, 동일 corpus 3,557 docs, fp16 b32 len=512):

| Encoder | Hit@1 | Hit@5 | MRR@10 | mean_gold_rank | cold encoding |
|---|---:|---:|---:|---:|---:|
| **gte (768d)** | 0.610 | **0.902** | 0.723 | 2.24 | **13s** |
| bge-m3 (1024d, len=512) | 0.732 | 0.878 | 0.798 | 1.95 | 23s |
| bge-m3 (1024d, len=1024) | 0.683 | **0.902** | 0.769 | 2.00 | 31s |

→ Hit@5 동일 (0.902 천장)이지만 **gte가 2.4× 빠름**. Hit@1·MRR는 bge-m3가 우세하지만 추가 비용 정당화 어려움 → gte 채택 + 필요 시 reranker로 top-rank 보강.

**확장 비교** (Exp2b — 9 후보 model, 동일 config):

| Model | dim | Hit@1 | Hit@5 | MRR@10 | enc_corpus |
|---|---:|---:|---:|---:|---:|
| `Alibaba-NLP/gte-multilingual-base` (현재) | 768 | 0.610 | **0.902** | 0.723 | **13.0s** |
| `BAAI/bge-m3` (len=1024) | 1024 | 0.683 | **0.902** | 0.769 | 31.4s |
| **`OrdalieTech/Solon-embeddings-large-0.1`** | 1024 | **0.756** | 0.854 | **0.801** | 21.4s |
| **`dragonkue/BGE-m3-ko`** | 1024 | 0.732 | 0.878 | 0.799 | 20.9s |
| `Snowflake/snowflake-arctic-embed-l-v2.0` | 1024 | 0.707 | 0.878 | 0.790 | 20.9s |
| `SamilPwC/PwC-Embedding_expr` | 1024 | 0.707 | 0.878 | 0.793 | 21.0s |
| `intfloat/multilingual-e5-large-instruct` | 1024 | 0.659 | 0.829 | 0.736 | 21.4s |
| `intfloat/multilingual-e5-base` | 768 | 0.634 | 0.829 | 0.715 | 7.4s |
| `intfloat/multilingual-e5-small` | 384 | 0.561 | 0.780 | 0.654 | **3.4s** |
| `codefuse-ai/F2LLM-v2-0.6B` | 1024 | 0.683 | 0.805 | 0.736 | 51.4s |
| `codefuse-ai/F2LLM-v2-330M` | 896 | 0.659 | 0.780 | 0.710 | 31.0s |

→ **Hit@5 ceiling=0.902는 gte/bge-m3만 도달** — search recall 보존 우선이면 baseline 유지. **Hit@1/MRR 우선이면 Solon-large 또는 dragonkue/BGE-m3-ko가 alternative** (Hit@1 +12pp). 자세한 분석: [`experiments/outputs/exp2b_embedding_models/report.md`](experiments/outputs/exp2b_embedding_models/report.md).

**Runtime config**: `fp16, max_seq_length=512, batch_size=32, attn_implementation=sdpa` (Exp18 13-cell sweep으로 도출):
- fp32 default(len=8192) b32 → fp16 b32 len=512: **116s → 13s (5.78× 가속)**, Hit@5 변화 없음
- batch saturation: GTE len=512에서 b32/b64/b96 encode 시간 ≈ 13s 동등 → **b32면 충분** (b96은 VRAM만 +50%)
- max_seq_length=512는 corpus token p90=538이라 90% docs 무손실 truncation

### 5.6 Embedding cache (SQLite)

`shrag/utils/embedding_cache.py` — `(encoder_id, embedding_mode, doc_id, text_sha1)` 4중 키. 같은 doc은 한 번만 인코딩 후 SQLite에 저장 → 후속 cell은 즉시 hit. `cache_db_path` field 추가하면 활성화.

### 5.7 Cross-encoder Reranker (선택)

dense retrieval 후 cross-encoder로 candidates → top-K 재정렬. **rerank output을 top-5로 fix**해서 LLM context로 전달.

**핵심 효과 (gte + dragonkue-rerank cand=5)**:
- **Hit@1**: 0.610 → **0.780** (+17pp, +7 queries)
- **Hit@3**: 0.829 → **0.902** (+7.3pp, ceiling 도달 — 41Q 중 37개의 gold가 top-3 안에 포함)
- **Hit@5**: 0.902 (search recall ceiling 유지)
- **MRR@10**: 0.723 → **0.837**
- **mean_gold_rank**: 2.24 → 1.16

→ rerank output이 top-5로 잘려서 **Hit@3 = Hit@5 = Hit@10이 모두 동일** (37/41). 즉 LLM에 top-3만 넘겨도 동일 coverage 확보 → prompt token 절감 가능.

**Candidates 수 sweep** (Exp8b, dragonkue/bge-reranker-v2-m3-ko 기준):

| candidates | Hit@1 | Hit@3 | Hit@5 | MRR@10 | mean_gold_rank | step4 sec/q |
|---:|---:|---:|---:|---:|---:|---:|
| (no rerank) | 0.610 | 0.829 | 0.902 | 0.723 | 2.24 | 0.27 |
| **5** ⭐ | **0.780** | **0.902** | 0.902 | **0.833** | **1.19** | 0.542 |
| 10 | 0.780 | 0.902 | 0.902 | 0.833 | 1.19 | 0.648 |

→ **candidates=5가 best** — cand=10 추가 비용 정당화 안 됨 (모든 metric 동일). 같은 BGE-v2-m3 base architecture 모델이라 더 wider pool에서도 같은 saturation 패턴 예상.

**Reranker model 비교** (Exp7b 품질 + Exp7c 속도, cand=5, dense top-5 → rerank → top-5):

| Reranker | params | Hit@1 | Hit@3 | MRR@10 | mgr | warm s/q |
|---|---:|---:|---:|---:|---:|---:|
| (no rerank) | — | 0.610 | 0.829 | 0.723 | 2.24 | 0 |
| BAAI/bge-reranker-v2-m3 | 568M | 0.732 | **0.902** | 0.813 | 1.22 | 0.174 |
| Alibaba-NLP/gte-multilingual-rerank | 305M | 0.585 | 0.780 | 0.704 | 1.70 | **0.074** |
| **dragonkue/bge-reranker-v2-m3-ko** ⭐ | 568M | **0.780** | **0.902** | **0.833** | **1.19** | 0.177 |

→ **dragonkue-KO 채택**: 품질 최우수 (Hit@1 +4.8pp vs BGE-v2-m3, MRR +0.020). BGE-v2-m3와 같은 base architecture라 속도도 동등 (0.177 vs 0.174 s/q, 차이 noise 수준). GTE-rerank는 빠르지만 품질 ↓로 기각. 자세한 분석: `experiments/outputs/FOLLOWUP_EXPERIMENT_SUMMARY.md` §4.

**Retriever × Reranker stack** (Exp7d): retriever를 dragonkue/BGE-m3-ko로 바꿔도 추가 이득 없음. cand=5에서 Hit@5 -1 query, cand=10 회복하지만 속도 2× 손실. **gte retriever + dragonkue rerank cand=5가 winner** (warm 0.195 s/q, cold 31.4s, Hit@1=0.780/Hit@5=0.902/MRR=0.837).

harness: `experiments/shared/rerank/cross_encoder.py`

### 5.8 Structured Answer Generation

Step 5에서 LLM 프롬프트는 답변을 다음 구조로 강제:
- **Title** — 핵심 답변 한 줄
- **Introduction** — 컨텍스트 요약
- **Main Body** — top-5 docs 인용 기반 상세

LLM 후보: vLLM `openai/gpt-oss-20b` (default), `Qwen3-8B`, OpenAI `gpt-5.4`, Gemini `2.5 Flash`.

---

## 6. 단계별 entrypoint

| 단계 | 모듈 | 출력 |
|---|---|---|
| step1 | `shrag.pipeline.steps.step1_search` | `search/<TS>/search_documents.jsonl`, `search_meta_results.json` |
| step2 (옵션) | `shrag.pipeline.steps.step2_decompose` | `decompose/<TS>/singlehop_decompose.jsonl` |
| step3 | `shrag.pipeline.steps.step3_build_vectordb` | `<output_dir>/<TS>/vector_db_*.csv` |
| step4 | `shrag.pipeline.steps.step4_retrieve` | `retrieval/<TS>/{qid}_*.json` |
| step5 | `shrag.pipeline.steps.step5_generate` | `final/<TS>/answers.jsonl` |

`pyproject.toml`의 `[project.scripts]` 설치 시 `shrag-search`, `shrag-retrieve` 등 단축 명령 사용 가능.

---

## 7. 출력 디렉토리

`run.py`:
- `--output <path>` 지정 시 그 폴더가 실행 루트
- 미지정 시 `outputs/run_<timestamp>/`
- 내부: `search/`, `decompose/` (옵션), `retrieval/`, `final/`, `run_manifest.json`

각 step 단독 실행 시 출력은 `outputs/<step-name>/<timestamp>/`.

---

## 8. 평가

### Step1 / Step4 (검색 + retrieval)
gold 포함 여부, Hit@k, MRR:

```bash
python -m experiments.shared.evaluate.retrieval_eval \
  --retrieval <retrieval.jsonl> \
  --gold data/gold/scienceon_gold.json \
  --ks 1 3 5 10 20 \
  --output metrics.json
```

### Step5 (답변 품질) — LLM-as-judge
```bash
python -m experiments.eval_gold_judge.run_gold_judge \
  --final-dir <run>/final \
  --judge-backend openai --judge-model gpt-5.4 \
  --output-dir experiments/outputs/eval/<run_tag>
```

---

## 9. ScienceON 41Q 최적 setting (실험 도출)

```
검색: k=30, n=10, m=50
인코더: Alibaba-NLP/gte-multilingual-base (fp16, max_seq_length=512, batch_size=32, sdpa)
임베딩 모드: 3*title+abstract
검색 단계: dense top-5 → dragonkue/bge-reranker-v2-m3-ko candidates 5 → top-3
LLM: gpt-oss-20b (vLLM)
```

**ScienceON gold 41Q metric** (전체 50Q 중 정답 gold 문서가 존재하는 41Q만 평가):
- Hit@1 = **0.780** (32/41)
- Hit@3 = **0.902** (37/41)
- Hit@5 = 0.902 (search recall ceiling, 나머지 4 queries는 어떤 setting에서도 gold 미수집)
- MRR@10 = **0.833**
- mean_gold_rank = **1.19**
- Cold embedding 13s (3,557 docs), hot e2e ~3.5s/q, cold e2e ~40s/q (acquisition dominant)

근거: [`experiments/BEST_CONFIG.md`](experiments/BEST_CONFIG.md), [`experiments/outputs/FOLLOWUP_EXPERIMENT_SUMMARY.md`](experiments/outputs/FOLLOWUP_EXPERIMENT_SUMMARY.md)

---

## 10. 데이터셋

- **questions**: `data/test.csv`, `experiments/outputs/exp1_kmn/questions_gold41.jsonl` (41Q)
- **gold**: `data/gold/scienceon_gold.json` ({qid: [doc_id]})
- **gold artifacts (LLM-judge 검증)**: `experiments/gold_scienceon/artifacts/scienceon_gold_full_only.jsonl`

---

## 11. 운영 도구

| 스크립트 | 용도 |
|---|---|
| `scripts/run_experiment_setup.sh` | `.env` 로드 + conda `shrag` 환경 활성 |
| `experiments/exp1_kmn/run_cell.sh` | (k, n, m) 변수 단일 cell 실행 |
| `experiments/exp18_isolated_speed/run_sweep.py` | subprocess 격리 sweep + memory abort |
| `experiments/shared/evaluate/embed_benchmark.py` | encoder 비교 harness (YAML cases) |
| `experiments/shared/evaluate/retrieval_eval.py` | Hit@k / MRR CLI |

---

## 12. 추가 문서

| 문서 | 내용 |
|---|---|
| [`shrag.pdf`](shrag.pdf) | 발표 논문 (5-step 파이프라인 정의, OR vs AND 실험, MIRACL 일반화 검증) |
| [`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md) | 본 repo의 후속 실험 설계서 |
| [`experiments/BEST_CONFIG.md`](experiments/BEST_CONFIG.md) | 1-page ScienceON 최적 setting 요약 |
| [`experiments/outputs/FOLLOWUP_EXPERIMENT_SUMMARY.md`](experiments/outputs/FOLLOWUP_EXPERIMENT_SUMMARY.md) | 전체 실험 종합 보고서 |
| [`problem_solve.md`](problem_solve.md) | 운영 중 만난 문제 + 해결 |

---

## 13. Notes

- step1은 single-source 정책. multi-source 동시는 별도 호출.
- 평가 CLI는 `run.py` 자동 호출 안 함. 별도 실행.
- 본 repo는 ScienceON 위주로 검증. PubMed / Wikipedia adapter는 동작하지만 동일 강도 평가는 미수행.
- WSL2 + 단일 GPU (RTX 4070 12GB) 환경에서 검증. 다른 VRAM에서는 batch / max_len 재조정.

---

## 14. Citation

```bibtex
@inproceedings{shrag2025,
  title={SHRAG: A Framework for Combining Human-Inspired Search with RAG},
  author={Ryu, Hyunseok and Shin, Wonjune and Park, Hyun},
  year={2025},
  organization={Gwangju Institute of Science and Technology (GIST)}
}
```

---

## 15. License

Apache License 2.0 (`pyproject.toml` 참고).
