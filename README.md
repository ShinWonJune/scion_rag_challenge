# SHRAG: Search like Human with RAG

학술 검색 플랫폼(ScienceON, PubMed, Wikipedia)과 RAG를 결합한 5-step 파이프라인. 질문이 들어오면 LLM이 검색어를 만들어 외부 검색 플랫폼에서 문서를 수집하고, dense retrieval로 후보를 선별한 뒤 LLM이 답변을 생성한다.

```
질문 → 검색어 생성 → 플랫폼 검색 → 임베딩/벡터DB → 밀집 검색 → (선택) reranker → LLM 답변
```

> **권장 setting과 측정 결과**: [`experiments/BEST_CONFIG.md`](experiments/BEST_CONFIG.md)
> **전체 실험 분석**: [`experiments/outputs/FOLLOWUP_EXPERIMENT_SUMMARY.md`](experiments/outputs/FOLLOWUP_EXPERIMENT_SUMMARY.md)


---

## 1. Architecture

```
shrag/
├── pipeline/                # 5-step 오케스트레이션
│   ├── run.py               # 통합 entrypoint (python -m shrag.pipeline.run)
│   ├── steps/               # 단계별 스크립트
│   │   ├── step1_search.py        # 검색어 생성 + 플랫폼 검색
│   │   ├── step2_decompose.py     # (옵션) 질문 분해
│   │   ├── step3_build_vectordb.py# 문서 임베딩 + vector DB
│   │   ├── step4_retrieve.py      # dense retrieve
│   │   └── step5_generate.py      # LLM 답변 생성
│   └── _impl/               # build_vectordb 등 구현 모듈
├── search/                  # ScienceON / PubMed / Wikipedia 어댑터, 캐시, throttle
├── retrieval/               # FAISS / cosine retrieval
├── features/                # 임베딩 인코더 (gte / bge-m3 / e5 / jina 등)
├── llm/                     # OpenAI / Gemini / vLLM 클라이언트
├── data_handler/, csv_headers/, evaluation/, utils/
└── prompts/

experiments/
├── shared/                  # 공통 retrieval/rerank/eval 모듈
├── exp{1,3,4,7,8,...,18}*  # 실험별 산출물
├── BEST_CONFIG.md           # ⭐ 권장 setting 1-page 요약
└── outputs/                 # 실험 결과 디렉토리
    ├── FOLLOWUP_EXPERIMENT_SUMMARY.md   # ⭐ 전체 실험 종합 보고서
    └── ...

configs/
├── query_encoder/           # encoder별 JSON config (cache_db_path 포함)
├── csv_schema/              # vector DB 스키마
└── credentials/             # API 자격증명 파일 위치 (gitignored)
```

---

## 2. Quick Start

### 2.1 환경 설정

```bash
pip install -r requirements.txt
cp .env.example .env  # API 키 등 채우기
```

`.env`에 필요한 항목:
- `OPENAI_API_KEY` — OpenAI 사용 시
- `GOOGLE_API_KEY` — Gemini 사용 시
- `GPT_OSS_VLLM_URL`, `GPT_OSS_VLLM_MODEL` — vLLM 사용 시
- `SCIENCEON_CREDENTIALS_PATH` — ScienceON API JSON 위치 (default `configs/credentials/scienceon_api_credentials.json`)

### 2.2 vLLM 서버 (옵션)

local vLLM 사용 시 `docker-compose.yml` 기준으로 서버 먼저 띄우기:

```bash
# 1) docker-compose.yml에서 model path / GPU id / port 수정
# 2) 시작
docker compose up -d

# 3) 상태 확인
curl http://localhost:8004/v1/models
```

### 2.3 E2E 실행 (best-config baseline)

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

질문 분해 옵션 추가 시 `--decompose`.

---

## 3. 단계별 entrypoint

각 단계는 단독 실행 가능. 모두 `python -m shrag.pipeline.steps.<step>` 패턴.

| 단계 | 모듈 | 입력 | 출력 |
|---|---|---|---|
| step1 | `shrag.pipeline.steps.step1_search` | questions, extractor settings | `search/<TS>/search_documents.jsonl` + `search_meta_results.json` |
| step2 (옵션) | `shrag.pipeline.steps.step2_decompose` | questions | `decompose/<TS>/singlehop_decompose.jsonl` |
| step3 | `shrag.pipeline.steps.step3_build_vectordb` | search docs + encoder config | `<output_dir>/<TS>/vector_db_*.csv` |
| step4 | `shrag.pipeline.steps.step4_retrieve` | encoder config + vectordb + questions | `retrieval/<TS>/{qid}_*.json` |
| step5 | `shrag.pipeline.steps.step5_generate` | retrieval results | `final/<TS>/answers.jsonl` |

`pyproject.toml`의 `[project.scripts]`로 설치 시 `shrag-search`, `shrag-retrieve` 등 단축 명령도 사용 가능.

---

## 4. 출력 디렉토리 규약

`run.py` 실행:
- `--output <path>` 지정 시: 그 폴더가 실행 루트
- 미지정 시: `outputs/run_<timestamp>/`
- 내부: `search/`, `decompose/` (옵션), `retrieval/`, `final/`, `run_manifest.json`

각 step 단독 실행 시 출력은 `outputs/<step-name>/<timestamp>/...`. 자세한 cli 옵션은 `python -m shrag.pipeline.steps.step1_search --help` 참고.

---

## 5. 핵심 기능

### 5.1 Frozen queries (재현성 + cache 가속)

같은 질문 set에 대해 키워드를 미리 생성·저장해두면 이후 step1은 vLLM 호출 없이 ScienceON 호출만 수행.

```bash
# 키워드만 생성하고 종료
python -m shrag.pipeline.steps.step1_search \
  --questions data/test.csv \
  --extractor vllm --vllm-url http://localhost:8004/v1 --vllm-model openai/gpt-oss-20b \
  --emit-frozen-queries outputs/frozen/queries.jsonl

# 이후 frozen queries로 검색만 실행
python -m shrag.pipeline.steps.step1_search \
  --questions data/test.csv \
  --frozen-queries outputs/frozen/queries.jsonl \
  --target-documents 50 \
  --output-dir outputs/search_run1
```

### 5.2 Embedding cache (SQLite, 결정론적 재사용)

같은 (encoder, embedding_mode, doc_id, text_sha1) 조합은 한 번만 인코딩 → SQLite에 저장 → 이후 cell은 즉시 hit.

encoder config에 다음 줄 추가하면 활성:
```json
{
  ...
  "cache_db_path": "outputs/_embedding_cache.db"
}
```

상세: `shrag/utils/embedding_cache.py`. 위험 / invalidation 정책은 `EXPERIMENT_DESIGN.md §13a`.

### 5.3 ScienceON 요청 캐시

ScienceON HTTP 응답은 자동으로 SHA1 키로 디스크에 캐시 (`outputs/_shared_cache/scienceon/`). 동일 (term, page, row_count)은 캐시 hit.

비활성: `--no-cache`

### 5.4 Reranker (선택)

dense retrieval 후 cross-encoder reranker 적용 가능. 권장:
- `BAAI/bge-reranker-v2-m3` (multilingual)
- candidates 5 → top 3 (Hit@3 0.829 → 0.902 검증, MRR 0.723 → 0.825)

CLI: `experiments/shared/rerank/cross_encoder_rerank.py`

---

## 6. 평가

### Step1 (검색) 평가
gold 문서가 step1 corpus에 포함됐는지 (gold_found_rate):

```bash
python -m experiments.shared.evaluate.retrieval_eval \
  --retrieval <retrieval.jsonl> \
  --gold data/gold/scienceon_gold.json \
  --ks 1 3 5 10 20 \
  --output metrics.json
```

### Step5 (답변 생성) 평가 — LLM-as-judge

```bash
python -m experiments.eval_gold_judge.run_gold_judge \
  --final-dir <run>/final \
  --judge-backend openai \
  --judge-model gpt-5.4 \
  --output-dir experiments/outputs/eval/<run_tag>
```

---

## 7. 권장 setting (요약)

```
검색: k=30, n=10, m=50 (검색어당 docs / 질문당 검색어 수 / dedup 후 cap)
인코더: Alibaba-NLP/gte-multilingual-base
       fp16, max_seq_length=512, batch_size=32, attn_implementation=sdpa
임베딩 모드: 3*title+abstract
검색: dense top-5 → rerank c=5 → context top-3
LLM: gpt-oss-20b (vLLM)
```

검증된 metric (ScienceON gold 41Q):
- Hit@5 = 0.902 (search recall ceiling)
- MRR@10 = 0.825
- mean_gold_rank = 1.30
- Cold embedding 13s (3,557 docs)
- Hot-path e2e ~3.5s/query, cold e2e ~40s/query (acquisition dominant)

근거: [`experiments/BEST_CONFIG.md`](experiments/BEST_CONFIG.md), [`experiments/outputs/FOLLOWUP_EXPERIMENT_SUMMARY.md`](experiments/outputs/FOLLOWUP_EXPERIMENT_SUMMARY.md)

---

## 8. 데이터셋

- **questions**: `data/test.csv` 또는 `experiments/outputs/exp1_kmn/questions_gold41.jsonl`
- **gold (41Q)**: `data/gold/scienceon_gold.json` ({qid: [doc_id]})
- **gold artifacts (LLM-judge 검증)**: `experiments/gold_scienceon/artifacts/scienceon_gold_full_only.jsonl`

---

## 9. 운영 도구

| 스크립트 | 용도 |
|---|---|
| `scripts/run_experiment_setup.sh` | `.env` 로드 + conda `shrag` 환경 활성 |
| `scripts/run_exp4_cell.sh` | 단일 e2e cell 실행 (Exp4 전용) |
| `experiments/exp1_kmn/run_cell.sh` | (k, n, m) 변수 단일 cell 실행 |
| `experiments/exp18_isolated_speed/run_sweep.py` | subprocess 격리 sweep + memory abort |
| `experiments/shared/evaluate/retrieval_eval.py` | Hit@k / MRR CLI |
| `experiments/shared/evaluate/embed_benchmark.py` | encoder 비교 harness (YAML cases) |

---

## 10. Notes

- step1은 single-source 정책. 여러 source 동시 사용은 별도 호출.
- 평가 CLI는 `run.py`가 자동 호출 안 함. 별도 실행.
- 본 repo는 ScienceON 위주로 검증되었음. PubMed / Wikipedia 어댑터는 동작하지만 동일 강도의 평가 미수행.
- Embedding 모델은 단일 GPU 환경에서 검증. 다중 GPU / 다른 VRAM에서는 batch / max_len을 재조정 필요.

---

## 11. License

Apache License 2.0 (`pyproject.toml` 참고).
