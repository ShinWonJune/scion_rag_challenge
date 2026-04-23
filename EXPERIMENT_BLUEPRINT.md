# SHRAG 실험 구현 설계도 (Blueprint)

작성일: 2026-04-22
기반: `PROJECT_IMPROVEMENT_PLAN.md`, `AGENTS.md`, 현재 `pipeline/` + `src/` 코드 상태

본 문서는 coding agent가 추가 질의 없이 **구현 → 검증 → 실험 실행**까지 진행할 수 있도록 모든 파일 경로, 인터페이스, 데이터 스키마, 완료 기준(DoD)을 명시한다.

---

## 0. 공통 규약

### 0.1 AGENTS.md 불변 제약 (모든 WP에 적용)

- `src/search/clients/scienceon_api_example.py` — **수정 금지.** 래핑은 `ScienceONAdapter` 레이어에서만.
- 의존 방향 `experiments/ → pipeline/ → src/` 일방통행. `src/`·`pipeline/` 내부에서 `experiments.*` import 금지. `src/` 내부에서 `pipeline.*` import 금지.
- `src/utils/` 의 공용 유틸은 `pipeline/`으로 이동 금지.
- `"retrival"` 키(오타) 문자열은 `rg "retrival" src/ pipeline/` 결과가 유지되는 범위에서만 수정.
- `configs/query_encoder/*.json` 의 `model_name` / `embedding_dim` 수정 시 반드시 WP0의 assertion을 통과해야 함.

### 0.2 구현 공통 규약

- Python 3.11. 타입 힌트 필수.
- 로깅: `logging.info/warning/error`. 사용자 대면 진행출력에 한해 `print` 허용.
- 경로: `pathlib.Path` + `os.environ`. 하드코딩된 절대경로 금지.
- 외부 API 호출: `try/except` + `logging.error`. 예외를 삼키지 않으려면 원인 type을 로그에 남길 것.
- 새 CLI entrypoint는 `argparse` 기반이며 `--help`가 에러 없이 출력되어야 함 (`scripts/smoke_help.ps1`로 회귀 검증).
- 파이프라인 코어 테스트는 `src/tests/` 에 `test_*.py` (`python -m pytest src/tests/`). 실험 코드 테스트는 `experiments/tests/` 에 `test_*.py` (`python -m pytest experiments/tests/`).

### 0.3 디렉토리 규약 (프로젝트 코드 ↔ 실험 코드 분리)

프로젝트 코드(`src/`, `pipeline/`, `configs/`)와 실험 전용 코드(`experiments/`)를 물리적으로 분리한다.

**분리 기준**

- `src/` / `pipeline/` 유지: 일반 run과 실험 run이 공유하는 파이프라인 인프라. 예) embedding_dim assertion(WP0), 요청 캐시(WP1), throttle(WP2), dedup(WP3), run_manifest·used_context(WP7), frozen-queries 모드(WP8).
- `experiments/` 이관: 실험에서만 쓰이는 평가 도구·runner·case 정의·대안 retriever/rerank/query-transform. 예) concurrency sweep(WP4), BM25(WP9), reranker(WP10), embed_benchmark·metrics(WP11), embedding 비교 case(WP12), HyDE(WP13), judge(WP14), significance(WP15), 실험 runbook(WP16). 실험에서 채택이 확정되면 `src/`로 승격.

**전체 디렉토리 구조**

```
scion_rag_challenge/
├── src/                                # 파이프라인 핵심 (기존)
├── pipeline/                           # 파이프라인 entrypoint (기존)
├── configs/                            # 파이프라인 설정 (기존)
├── data/                               # 입력 데이터 (기존)
├── scripts/                            # 기존 보조 스크립트 (smoke_help 등; 실험 스크립트는 experiments/로 이관)
├── outputs/                            # 일반 파이프라인 run 산출물
│   ├── _shared_cache/                  # WP1 요청 캐시 (run 간 공유, 파이프라인 레벨)
│   │   └── scienceon/<sha1>.json
│   └── run_<timestamp>/                # `pipeline/run_pipeline.py` 산출
│       ├── run_manifest.json           # WP7
│       ├── search/
│       ├── retrieval/
│       └── final/
└── experiments/                        # 실험 전용 (신규)
    ├── README.md                       # 실험 허브 개요
    ├── shared/                         # 실험 간 공유 코드
    │   ├── __init__.py
    │   ├── evaluate/
    │   │   ├── __init__.py
    │   │   ├── embed_benchmark.py      # WP11
    │   │   ├── metrics_ir.py           # WP11
    │   │   ├── judge.py                # WP14
    │   │   └── significance.py         # WP15
    │   ├── rerank/
    │   │   ├── __init__.py
    │   │   └── cross_encoder.py        # WP10
    │   ├── retrievers/
    │   │   ├── __init__.py
    │   │   └── bm25.py                 # WP9
    │   ├── query_transform/
    │   │   ├── __init__.py
    │   │   ├── factory.py              # WP13
    │   │   └── hyde.py                 # WP13
    │   ├── prompts/
    │   │   └── judge_pointwise.py      # WP14
    │   └── cases/                      # 실험 케이스 YAML
    │       ├── exp2_embed.yaml         # WP12
    │       └── exp3_query.yaml         # WP13/16
    ├── exp1_acquisition/
    │   ├── README.md
    │   └── run.sh                      # WP16
    ├── exp2_embedding/
    │   ├── README.md
    │   └── run.sh                      # WP12·16
    ├── exp3_query_ablation/
    │   ├── README.md
    │   └── run.sh                      # WP16
    ├── exp4_e2e/
    │   ├── README.md
    │   └── run.sh                      # WP16
    ├── tests/                          # 실험 코드 전용 테스트
    └── outputs/                        # 실험 산출물
        ├── frozen/                     # WP8 frozen-queries 산출
        │   └── queries.jsonl
        ├── acquisition/<run_id>/       # Experiment 1
        ├── embedding_benchmark/<run_id>/
        ├── query_ablation/<run_id>/
        ├── e2e/<run_id>/
        └── eval/<run_id>/              # WP14 judge 산출
```

`<run_id>` = `YYMMDD_HHMMSS_<slug>` (예: `260422_153012_gte_vs_bge`).

**Import / 경로 규약**

- `experiments/ → pipeline/ → src/` 일방통행. `src/`·`pipeline/`는 `experiments.*`에 의존하지 않는다.
- 실험 간 공유 import는 `experiments.shared.*` 모듈을 통해서만 한다. 예) `from experiments.shared.evaluate.judge import JudgeClient`.
- `experiments/**` 각 디렉토리에는 필요한 `__init__.py`를 둔다. runbook(shell)은 repo root에서 실행하며 repo root가 `PYTHONPATH`에 포함된다고 가정한다.
- 실험 runbook은 파이프라인 CLI(`python -m pipeline.step1_search`, `python -m src.retrieval_system.main` 등)를 호출해도 되지만, 파이프라인 CLI가 `experiments.*`에 의존해선 안 된다.

---

## 1. Work Package 의존 그래프

```
WP0  ──────────────────────┐
WP3 ───────────┐            │
WP1 ── WP2 ── WP4           │
  │     │      │            │
  │     │      WP8 ──────── WP11 ── WP12
  │     │                    │
  WP7 ─────────────────────── │
                   WP9 ──────┤
                   WP10 ─────┤
                   WP13 ─────┤
                             ├── WP16 (runbooks)
                   WP14 ── WP15
```

| WP | 제목 | 대응 plan 섹션 | 의존 |
|---|---|---|---|
| WP0 | embedding_dim 정정 + assertion | 6.2 | - |
| WP1 | 요청 캐시 | 6.3-2 | - |
| WP2 | Global throttle (AIMD) | 6.3-1 / 1.3 Step 1 | WP1 |
| WP3 | Dedup 키 `doc_id` 전환 | 6.4 | - |
| WP4 | Max concurrency sweep 실험 | 1.3 Step 2 | WP1, WP2 |
| WP7 | run_manifest + used_context | 6.7 | - |
| WP8 | Frozen queries 모드 | 6.7 | - |
| WP9 | BM25 retriever | 6.5-1 | - |
| WP10 | Cross-encoder reranker | 6.5-2 | - |
| WP11 | 고정 corpus benchmark harness | 2.3 / 2.4 | WP0, WP8 |
| WP12 | Embedding 모델 비교 runner | 2.4 | WP11 |
| WP13 | HyDE query transform | 3.4 | WP11 |
| WP14 | LLM judge (pointwise) | 3.3 | WP7 |
| WP15 | Significance testing util | 6.6 | WP14 |
| WP16 | Experiment 1–4 runbooks | 3.5 | 위 전부 |

(구 WP5 zero-doc fallback · WP6 runtime provenance join 은 plan §1.3 개편으로 제거되었다. 설계 이력 보존은 git log 참조.)

---

## 2. WP 상세 명세

각 WP 블록은 다음 구조를 따른다: **목표 / 편집 파일 / 인터페이스 / 스키마 / 검증 / DoD**.

### WP0. embedding_dim 정정 + assertion

**목표**: `embedding_dim` config 값과 실제 embedding 차원의 silent mismatch 차단.

**편집 파일**:

- `configs/query_encoder/config_gte-multilingual-base.json` — `embedding_dim: 1024 → 768`.
- 다른 encoder config들도 점검 (`config_bge_m3.json` → 1024 확인, `config_PwC-Embedding_expr.json`, `config_snowflake.json` 각 모델 card 기준).
- `src/build_vectordb_search.py` — `generate_batch_embeddings` 반환 직후 검증.

**인터페이스 (추가 로직)**:

```python
# src/build_vectordb_search.py, build_vectordb_search() 내 embeddings 생성 직후
if embeddings is None:
    return
expected_dim = int(config["embedding_dim"])
actual_dim = int(embeddings.shape[1])
if actual_dim != expected_dim:
    raise ValueError(
        f"embedding_dim mismatch for {config['model_name']}: "
        f"config={expected_dim} actual={actual_dim}"
    )
```

**검증**:

```bash
python -m src.build_vectordb_search \
  --config_path configs/query_encoder/config_gte-multilingual-base.json \
  --data_schema configs/csv_schema/test_2.json \
  --docs_jsonl_path outputs/<기존 run>/search/search_documents.jsonl
# 에러 없이 vectordb CSV 생성되어야 함
```

**DoD**:
- [ ] 기존 저장된 `search_documents.jsonl` 로 vectordb 빌드가 완주.
- [ ] `embedding_dim`을 고의로 틀린 값(예: 999)으로 바꾸면 `ValueError` 발생.
- [ ] 모든 `configs/query_encoder/*.json`의 `embedding_dim`이 실제 모델 native dim과 일치하거나, 주석으로 Matryoshka truncation 의도를 명시.

---

### WP1. ScienceON 요청 캐시

**목표**: 동일 `(source, term, page, row_count, fields)` 요청을 disk cache로 재사용해 429 빈도↓, 재현성↑.

**편집 파일**:

- 신규 `src/search/cache.py`
- `src/search/scienceon_adapter.py` — adapter가 캐시를 경유하도록 수정
- `src/search/clients/pubmed_api_client.py`, `wikipedia_api_client.py` — 동일 패턴 적용 (선택, WP1.b로 분리 가능)

**인터페이스**:

```python
# src/search/cache.py
from pathlib import Path

class RequestCache:
    def __init__(self, root: Path, enabled: bool = True) -> None: ...
    def get(self, key: dict) -> dict | None: ...
    def put(self, key: dict, value: dict) -> None: ...
    @staticmethod
    def make_key(source: str, term: str, cur_page: int,
                 row_count: int, fields: list[str]) -> dict: ...
```

- 저장 경로: `root / source / <sha1(key)>.json`. `root` 기본값 `outputs/_shared_cache`.
- 직렬화: `{"key": {...}, "value": {...}, "created_at": "...", "hit_count": N}`.
- 쓰기는 atomic: `<sha1>.tmp` 로 쓰고 `os.replace` 로 rename.

**환경변수 / CLI**:

- `SHRAG_CACHE_ROOT` (env) → root 기본값 덮어쓰기.
- `pipeline/run_pipeline.py` 에 `--no-cache` 추가 (cache disable).

**검증**:

```bash
# 동일 질문으로 두 번 실행 → 두 번째 실행은 ScienceON HTTP 호출 0회
python pipeline/run_pipeline.py --questions data/test_5q.csv ... --output outputs/cache_test_1
python pipeline/run_pipeline.py --questions data/test_5q.csv ... --output outputs/cache_test_2
# 두 run의 search_documents.jsonl 이 bit-identical (dedup order 유의)
```

**DoD**:
- [ ] `src/tests/test_request_cache.py` 로 get/put/miss round-trip 검증.
- [ ] 두 번째 실행의 ScienceON 호출 수가 0 (또는 새 term만큼).
- [ ] cache disable 경로 동작 (`--no-cache` 시 파일 접근 없음).

---

### WP2. Global throttle (AIMD)

**목표**: 동시 요청 수와 최소 간격을 runtime 조정해 429 근본 원인 완화.

**편집 파일**:

- 신규 `src/search/throttle.py`
- `src/search/scienceon_adapter.py` — adapter 호출 전후에 throttle wrap

**인터페이스**:

```python
# src/search/throttle.py
class AimdThrottle:
    def __init__(
        self,
        max_concurrency: int = 2,
        min_interval_sec: float = 0.5,
        increase_step: float = 0.25,   # AIMD A (429 시 증가)
        decrease_factor: float = 0.9,  # AIMD D (연속 성공 시 감소)
        decrease_after_success: int = 20,
        interval_cap_sec: float = 10.0,
    ) -> None: ...
    def acquire(self) -> None: ...   # blocks until allowed
    def report_success(self) -> None: ...
    def report_429(self, retry_after_sec: float | None = None) -> None: ...
```

- `threading.Semaphore(max_concurrency)` + `threading.Lock` + `time.monotonic()`.
- `report_429` 는 `min_interval_sec *= (1 + increase_step)` 하고 `interval_cap_sec` clamp.
- `report_success` 를 `decrease_after_success` 번 받으면 `min_interval_sec *= decrease_factor`, 하한 0.1s.

**ScienceONAdapter 적용**:

```python
# ScienceONAdapter.search(...) 내부 각 HTTP 호출 직전
self.throttle.acquire()
try:
    resp = self.client.search_articles(...)
    self.throttle.report_success()
except HTTPError as e:
    if e.status == 429:
        self.throttle.report_429(retry_after_sec=parse_retry_after(e))
        raise
```

**검증**:

- 단위 테스트: `acquire → report_429 → acquire` 후 경과시간이 증가한 `min_interval_sec` 이상.
- 통합: 50질문 run에서 throttle counter(`rate_limit_count`, `request_count`)를 누적 (WP4 sweep 실험에서 재사용).

**DoD**:
- [ ] `src/tests/test_throttle.py` — 타이밍 테스트(허용 오차 ±50ms).
- [ ] 동일 코퍼스 재생성 조건에서 기존 대비 429 수 > 50% 감소.

---

### WP3. Dedup 키 `doc_id` 전환

**목표**: title 기반 false positive 제거. 서로 다른 논문이 동명으로 합쳐지는 케이스 차단.

**편집 파일**:

- `src/utils/dedup.py`
- `pipeline/step1_search.py` — 호출부 key 인자 변경
- `src/search/services/search_meta_system.py` (있다면 내부 호출부)

**인터페이스**:

```python
# src/utils/dedup.py
from typing import Iterable, Sequence

def remove_duplicates(
    docs: Iterable[dict],
    key: str = "doc_id",
    fallback_keys: Sequence[str] = ("title",),
) -> list[dict]:
    """
    우선 `key`가 존재하고 truthy한 문서끼리 dedup.
    `key`가 없는 문서는 `fallback_keys` 순서대로 시도.
    어느 것도 없으면 drop 하지 않고 그대로 보존.
    """
```

**동작 규약**:
- 첫 번째로 등장한 doc 유지, 이후 동일 키는 drop.
- 키 정규화: `str(value).strip()`. 공백만인 값은 fallback으로.

**검증**:

```python
# src/tests/test_dedup.py
def test_prefers_doc_id_over_title():
    docs = [
        {"doc_id": "CN1", "title": "같은 제목"},
        {"doc_id": "CN2", "title": "같은 제목"},  # 서로 다른 논문
    ]
    assert len(remove_duplicates(docs)) == 2

def test_fallback_to_title_when_no_doc_id():
    docs = [
        {"title": "A"},
        {"title": "A"},
    ]
    assert len(remove_duplicates(docs)) == 1
```

**DoD**:
- [ ] 단위 테스트 통과.
- [ ] 기존 pipeline 회귀: 동일 corpus에서 dedup 후 문서 수가 오히려 증가(=false positive 제거 효과) 해야 함.

---

### WP4. ScienceON max_concurrency sweep 실험

**목표**: plan §1.3 Step 2 의 운영 지점 탐색. WP1(cache)·WP2(AIMD throttle) 을 고정한 상태에서 `max_concurrency` 만 sweep 하여 *429 rate 를 허용치 이하로 유지하면서 throughput 을 최대화하는 동시성* 을 찾고, 그 값을 WP2 의 operating default / AIMD 상한으로 확정한다.

**편집 파일**:

- `src/search/throttle.py` — `get_counters() -> dict` 추가 (누적 `rate_limit_count`, `request_count`). AIMD 끄고 고정 concurrency 로 동작시키는 `fixed_concurrency` 플래그 지원.
- `pipeline/step1_search.py` — `--max-concurrency N` / `--fixed-concurrency` CLI pass-through.
- 신규 `experiments/shared/evaluate/concurrency_report.py` — 각 sweep run의 counter 집계 + `report.md` 생성.
- 신규 `experiments/exp0_concurrency_sweep/run.sh`
- 신규 `experiments/exp0_concurrency_sweep/README.md`

**실행 구조**:

```bash
# experiments/exp0_concurrency_sweep/run.sh
set -euo pipefail
RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)}"
OUT_ROOT="experiments/outputs/concurrency_sweep/${RUN_ID}"
mkdir -p "${OUT_ROOT}"

# cache cold start: throughput 측정 왜곡 방지
rm -rf outputs/_shared_cache/scienceon

for MAX_C in 1 2 3 5 8; do
  python -m pipeline.step1_search \
    --questions data/test.csv \
    --sources scienceon \
    --extractor vllm \
    --max-concurrency "${MAX_C}" \
    --fixed-concurrency \
    --output-dir "${OUT_ROOT}/c${MAX_C}"
done

python -m experiments.shared.evaluate.concurrency_report \
  --input "${OUT_ROOT}" \
  --epsilon 0.01 \
  --output "${OUT_ROOT}/report.md"
```

**측정 지표** (per `max_concurrency`, run 종료 시 throttle counter 를 JSON 으로 덤프):

- `rate_limit_count` — 누적 429 응답 수
- `request_count` — 캐시 miss 로 실제 전송된 요청 수
- `rate_limit_rate` = `rate_limit_count / max(1, request_count)`
- `wall_clock_sec` — `step1_search` 전체 실행 시간
- `effective_throughput` = `request_count / wall_clock_sec` (req/s)
- `zero_doc_question_count`

**선정 규칙**:

- `rate_limit_rate ≤ ε` (기본 ε = 1%, CLI로 조정) 를 만족하는 `max_concurrency` 중 `effective_throughput` 최대값을 채택.
- 후보가 없으면 `min_interval_sec` 을 상향한 뒤 재실행. 채택값은 WP2 AIMD 의 `max_concurrency` 상한으로 clamp.

**DoD**:
- [ ] `bash experiments/exp0_concurrency_sweep/run.sh` 가 단일 명령으로 완주.
- [ ] 각 `${OUT_ROOT}/c${MAX_C}/` 에 throttle counter JSON 포함.
- [ ] `${OUT_ROOT}/report.md` 에 per-concurrency 표 + ε 기준 선정값 기록.
- [ ] 선정값을 ScienceON 기본 config(또는 `src/search/throttle.py` default)에 반영하고, 동일 설정으로 재실행 시 `rate_limit_rate` 재현 가능 (±1%p).

---

### WP7. run_manifest + step5 used_context

**목표**: 재현성/관측가능성의 최소 세트.

**편집 파일**:

- `pipeline/run_pipeline.py`
- `src/preprocess_and_generate_answer.py`

**스키마 S3 (run_manifest.json)**:

```json
{
  "run_id": "260422_153012",
  "git_sha": "3256ce0abc...",
  "requirements_sha1": "...",
  "started_at": "2026-04-22T15:30:12+09:00",
  "finished_at": "2026-04-22T16:11:03+09:00",
  "cli_args": { "questions": "...", "encoder": "...", "...": "..." },
  "vllm": { "url": "...", "model": "openai/gpt-oss-20b", "version": null },
  "credentials_fingerprint": {
    "scienceon": "sha1:abcd...",
    "pubmed":    "sha1:efgh..."
  },
  "cache": {
    "enabled": true,
    "root": "outputs/_shared_cache",
    "hit_count": 128,
    "miss_count": 54
  },
  "steps_completed": ["step1", "step3", "step4", "step5"]
}
```

**스키마 S5 (step5 최종 JSON/CSV — 행 단위)**:

기존 필드에 추가:

```json
{
  "question_id": "...",
  "answer": "...",
  "used_context": [
    {"doc_id": "CN123", "rank": 1, "from_step": "step4"}
  ],
  "llm_backend": "vllm",
  "llm_model": "openai/gpt-oss-20b"
}
```

**credentials_fingerprint** 생성:

```python
import hashlib, json
from pathlib import Path
def fingerprint_json_file(p: Path) -> str:
    if not p.exists():
        return "missing"
    return "sha1:" + hashlib.sha1(p.read_bytes()).hexdigest()[:12]
```

**검증**:

```bash
jq '.run_id, .git_sha, .cache' outputs/<run>/run_manifest.json
jq '.[0].used_context' outputs/<run>/final/<subdir>/predictions.json
```

**DoD**:
- [ ] `run_pipeline.py` 완료 시 `run_manifest.json` 존재.
- [ ] step5 결과의 모든 레코드가 비어있지 않은 `used_context` 배열을 가짐.
- [ ] `git_sha`, `requirements_sha1`, `credentials_fingerprint` 가 채워짐(값은 해시만, 원문 노출 금지).

---

### WP8. Frozen queries 모드

**목표**: keyword extractor 비결정성을 benchmark 실험에서 제거.

**편집 파일**:

- `pipeline/step1_search.py`

**인터페이스**:

```bash
python -m pipeline.step1_search \
  --questions data/test.csv \
  --sources scienceon \
  --frozen-queries experiments/outputs/<run>/queries_frozen.jsonl \
  --output-dir experiments/outputs/<run>/search
```

**frozen queries 파일 스키마 S7**:

```jsonl
{"question_id": "0", "query": "...", "keywords": {"korean": [...], "english": [...]}, "search_terms": ["...", "..."]}
{"question_id": "1", "query": "...", "keywords": {...}, "search_terms": [...]}
```

**동작 규약**:

- `--frozen-queries` 가 주어지면 `keyword_extractor.extract_keywords()` / `generate_search_terms()` 호출을 건너뛰고 파일에서 읽음.
- 파일에 없는 `question_id` 는 기존 extractor 경로로 fallback하되 경고 로그.
- 생성용 보조 CLI: `python -m pipeline.step1_search --emit-frozen-queries <path>` — 추출만 수행하고 종료.

**검증**:

```bash
# 1. frozen 파일 생성
python -m pipeline.step1_search --questions data/test.csv --extractor vllm \
  --emit-frozen-queries experiments/outputs/frozen/queries.jsonl

# 2. 같은 frozen 파일로 두 번 실행 → search_terms 비트 동일
for i in 1 2; do
  python -m pipeline.step1_search --questions data/test.csv --sources scienceon \
    --frozen-queries experiments/outputs/frozen/queries.jsonl \
    --output-dir experiments/outputs/frozen/run_$i
done
diff <(jq -S '.results[].search_terms' experiments/outputs/frozen/run_1/<ts>/search_meta_results.json) \
     <(jq -S '.results[].search_terms' experiments/outputs/frozen/run_2/<ts>/search_meta_results.json)
```

**DoD**:
- [ ] 동일 frozen 파일로 두 run의 `search_terms` 가 정확히 일치.
- [ ] frozen 파일이 없는 질문에 대해서도 extractor fallback 경로 정상.

---

### WP9. BM25 retriever

**목표**: sparse baseline 추가. dense 개선의 실제 이득 여부 검증.

**편집 파일**:

- 신규 `experiments/shared/retrievers/__init__.py`
- 신규 `experiments/shared/retrievers/bm25.py`
- `experiments/shared/evaluate/embed_benchmark.py` — `BenchmarkCase.retriever == "bm25"` 분기에서 위 모듈 사용
- `requirements.txt` — `bm25s==0.2.0` 추가 (런타임 의존성은 프로젝트 레벨 관리)

**비주입 원칙**: 파이프라인 경로(`src.retrieval_system.main`)에는 BM25 스위치를 추가하지 않는다. 실험에서 채택이 확정되면 그때 `src/`로 승격한다.

**인터페이스**:

```python
# experiments/shared/retrievers/bm25.py
from typing import Protocol

class Retriever(Protocol):
    def build(self, corpus_texts: list[str]) -> None: ...
    def search(self, queries: list[str], top_k: int): ...

class BM25Retriever:
    def __init__(self, tokenizer: str = "whitespace") -> None: ...
    def build(self, corpus_texts: list[str]) -> None: ...
    def search(self, queries: list[str], top_k: int):
        """Returns (scores[B, top_k], indices[B, top_k])."""
```

- `tokenizer` 선택지: `"whitespace"`, `"mecab"` (optional, mecab 설치 시).
- corpus 텍스트 구성은 dense와 동일하게 `embedding_text` 필드 재사용 (config의 `embedding_mode` 고려).

**실행 방법**: BM25 케이스는 embed_benchmark(WP11)를 통해서만 실행한다.

```bash
python -m experiments.shared.evaluate.embed_benchmark \
  --cases experiments/shared/cases/exp_bm25.yaml \
  --corpus <fixed_corpus.jsonl> \
  --queries <frozen_queries.jsonl> \
  --gold <gold.json> \
  --output experiments/outputs/embedding_benchmark/<run>/bm25
```

case YAML 예:

```yaml
cases:
  - case_id: E_BM25_whitespace
    retriever: bm25
    encoder_config: null
    bm25_tokenizer: whitespace
```

**검증**:

- MIRACL-ko 또는 내부 gold set으로 Recall@10 계산.
- dense Recall@10 대비 절대차이, paired bootstrap CI 확인 (WP15 연계).

**DoD**:
- [ ] 1070문서 corpus에서 쿼리당 ≤ 100ms.
- [ ] embed_benchmark의 BM25 케이스가 dense 케이스와 동일한 `metrics.json` schema 생성.

---

### WP10. Cross-encoder reranker

**목표**: Top-K → Top-k 재정렬로 retrieval 품질 upper bound 탐색.

**편집 파일**:

- 신규 `experiments/shared/rerank/cross_encoder.py`
- `experiments/shared/rerank/__init__.py`

**인터페이스**:

```python
# experiments/shared/rerank/cross_encoder.py
class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 16,
        device: str | None = None,
    ) -> None: ...

    def rerank(
        self,
        query: str,
        candidates: list[dict],   # each has 'title', 'abstract'
        top_k: int = 5,
    ) -> list[dict]:
        """Returns candidates sorted by rerank score desc, with added `rerank_score` field."""
```

**CLI**:

```bash
python -m experiments.shared.rerank.cross_encoder \
  --input outputs/<run>/retrieval/<subdir> \
  --output experiments/outputs/rerank/<run_id>/<subdir> \
  --model BAAI/bge-reranker-v2-m3 \
  --top_k 5
```

- 입력 디렉토리(`outputs/<run>/retrieval/...`)의 각 JSON 파일을 **read-only** 로 읽어 `hits` 를 rerank. 파이프라인 산출물은 직접 수정하지 않는다.
- 출력은 `experiments/outputs/rerank/<run_id>/<subdir>/` 에 별도 저장. 파일에 `rerank_score`, 재정렬된 `rank` 기록.

**검증**:

- 동일 corpus/질문 세트에서 rerank 전/후 Recall@5, MRR@10 비교.

**DoD**:
- [ ] GPU 사용 가능 환경에서 50질문 × Top-50 rerank가 2분 이내.
- [ ] 재정렬 이후 hit의 rank 필드가 1..k 로 정규화됨.

---

### WP11. 고정 corpus benchmark harness

**목표**: 한 번 수집한 corpus를 고정하고 embedding/retriever만 바꿔가며 비교.

**편집 파일**:

- 신규 `experiments/shared/evaluate/embed_benchmark.py`
- 신규 `experiments/shared/evaluate/metrics_ir.py` — Recall@k, MRR@k, nDCG@k, paired bootstrap hook

**인터페이스**:

```python
# experiments/shared/evaluate/embed_benchmark.py
@dataclass
class BenchmarkCase:
    case_id: str
    encoder_config: Path
    retriever: str = "dense"     # "dense" | "bm25"
    query_transform: str = "raw"  # "raw" | "keywords" | "hyde"
    rerank_model: str | None = None
    top_k: int = 50

def run_benchmark(
    cases: list[BenchmarkCase],
    corpus_jsonl: Path,
    queries_jsonl: Path,        # frozen queries (WP8)
    gold_path: Path,            # qid → list[doc_id]
    output_root: Path,
) -> None: ...
```

**산출 구조**:

```
<output_root>/
├── corpus.jsonl                      # 입력 corpus 사본 (symlink 또는 hash 파일)
├── queries.jsonl                     # frozen queries
├── gold.json                         # 정답 레이블
├── cases/<case_id>/
│   ├── vectordb.csv                  # dense인 경우
│   ├── retrieval.jsonl
│   └── metrics.json
└── report.md                         # 모든 case 비교표
```

**metrics.json 스키마**:

```json
{
  "case_id": "E1_bge_m3",
  "n_queries": 50,
  "recall_at_5": 0.64,
  "recall_at_10": 0.78,
  "mrr_at_10": 0.51,
  "ndcg_at_10": 0.59,
  "per_query_recall_at_10": [1, 0, 1, ...]
}
```

**검증**:

```bash
python -m experiments.shared.evaluate.embed_benchmark \
  --cases experiments/shared/cases/exp2_embed.yaml \
  --corpus experiments/outputs/embedding_benchmark/<run>/corpus.jsonl \
  --queries experiments/outputs/embedding_benchmark/<run>/queries.jsonl \
  --gold data/gold/scienceon_gold.json \
  --output experiments/outputs/embedding_benchmark/<run>
```

**DoD**:
- [ ] 동일 corpus/gold에서 2회 실행 시 metrics.json이 bit-identical (embedding은 deterministic 전제).
- [ ] `report.md` 가 케이스별 테이블로 생성.

---

### WP12. Embedding 모델 비교 runner

**목표**: plan 2.4의 E0~E4 + BM25 를 한 번에 실행.

**편집 파일**:

- 신규 `experiments/shared/cases/exp2_embed.yaml` — case 정의
- 신규 `experiments/exp2_embedding/run.sh`

**case 정의 예**:

```yaml
# experiments/shared/cases/exp2_embed.yaml
cases:
  - case_id: E_BM25
    retriever: bm25
    encoder_config: null
    bm25_tokenizer: whitespace
  - case_id: E0_gte
    encoder_config: configs/query_encoder/config_gte-multilingual-base.json
    retriever: dense
  - case_id: E1_bge_m3
    encoder_config: configs/query_encoder/config_bge_m3.json
    retriever: dense
  - case_id: E2_jina_v3
    encoder_config: configs/query_encoder/config_jina_v3.json
    retriever: dense
    query_transform: raw
  - case_id: E3_me5_large
    encoder_config: configs/query_encoder/config_multilingual_e5_large.json
    retriever: dense
    query_instruction: "query: "
  - case_id: E4_me5_instruct
    encoder_config: configs/query_encoder/config_multilingual_e5_large_instruct.json
    retriever: dense
```

**준비할 신규 config**:
- `configs/query_encoder/config_jina_v3.json` (`model_name: jinaai/jina-embeddings-v3`, `embedding_dim: 1024`)
- `configs/query_encoder/config_multilingual_e5_large.json`
- `configs/query_encoder/config_multilingual_e5_large_instruct.json`

**DoD**:
- [ ] `bash experiments/exp2_embedding/run.sh` 단일 명령으로 전체 case 실행.
- [ ] 각 case에 대해 `metrics.json` 생성, `report.md` 생성.
- [ ] 실행 실패 case가 있어도 다른 case는 계속 진행, 실패는 `report.md`에 명시.

---

### WP13. HyDE query transform

**목표**: plan 3.4의 retrieval query representation ablation.

**편집 파일**:

- 신규 `experiments/shared/query_transform/__init__.py`
- 신규 `experiments/shared/query_transform/hyde.py`
- 신규 `experiments/shared/query_transform/factory.py`
- `experiments/shared/evaluate/embed_benchmark.py` — `BenchmarkCase.query_transform` 분기에서 factory 사용
- 신규 `experiments/shared/cases/exp3_query.yaml` — query_transform 비교 케이스

**비주입 원칙**: 파이프라인 경로(`src.retrieval_system.main`)에는 `--query_transform`을 추가하지 않는다. HyDE/Multi-HyDE는 embed_benchmark(WP11) BenchmarkCase 를 통해서만 사용한다. 채택이 확정되면 `src/`로 승격.

**인터페이스**:

```python
# experiments/shared/query_transform/hyde.py
class HyDETransform:
    def __init__(self, llm_client: BaseLLMClient, n_samples: int = 1,
                 language: str = "auto") -> None: ...

    def transform(self, question: str) -> list[str]:
        """Returns list of hypothetical abstract texts (len == n_samples)."""
```

- n_samples == 1 → single HyDE, n_samples > 1 → Multi-HyDE.
- Multi-HyDE 의 embedding은 평균(default) 또는 union retrieval (factory option).

**실행 방법**: query_transform 비교는 embed_benchmark cases YAML에서 지정한다.

```yaml
# experiments/shared/cases/exp3_query.yaml
cases:
  - { case_id: Q_raw,          query_transform: raw }
  - { case_id: Q_keywords,     query_transform: keywords }
  - { case_id: Q_hyde_mean_1,  query_transform: hyde_mean_1 }
  - { case_id: Q_hyde_mean_3,  query_transform: hyde_mean_3 }
  - { case_id: Q_hyde_union_3, query_transform: hyde_union_3 }
```

```bash
python -m experiments.shared.evaluate.embed_benchmark \
  --cases experiments/shared/cases/exp3_query.yaml \
  --corpus <fixed_corpus.jsonl> \
  --queries <frozen_queries.jsonl> \
  --gold <gold.json> \
  --output experiments/outputs/query_ablation/<run>
```

**검증**:

- LLM 호출 캐시: 같은 question → 같은 hypothetical 문서. temperature=0, seed 고정.
- 산출 retrieval JSON 에 `query_transform: "hyde_mean_3"` 메타.

**DoD**:
- [ ] `query_transform: raw | keywords | hyde_mean_1 | hyde_mean_3 | hyde_union_3` 각 모드가 embed_benchmark 케이스로 실행 가능.
- [ ] 동일 조건 2회 실행 시 결과 bit-identical.

---

### WP14. LLM judge (pointwise)

**목표**: plan 3.3 의 context_relevance / answer_faithfulness / answer_completeness 자동 채점.

**편집 파일**:

- 신규 `experiments/shared/evaluate/judge.py`
- 신규 `experiments/shared/prompts/judge_pointwise.py`

**스키마 S6 (judge input)**:

```json
{
  "question_id": "4",
  "question": "...",
  "retrieved_contexts": [
    {"title": "...", "abstract": "...", "rank": 1, "doc_id": "..."}
  ],
  "answer": "...",
  "gold_answer": null,
  "gold_doc_titles": null
}
```

**스키마 S6 (judge output)**:

```json
{
  "question_id": "4",
  "judge_model": "gemini-2.5-flash",
  "context_relevance": 4,
  "answer_faithfulness": 3,
  "answer_completeness": 4,
  "unsupported_claims": ["..."],
  "missing_key_points": ["..."],
  "verdict": "partial",
  "raw_response": "..."
}
```

**CLI**:

```bash
python -m experiments.shared.evaluate.judge \
  --predictions outputs/<run>/final/<ts>/predictions.json \
  --retrieval_dir outputs/<run>/retrieval/<ts> \
  --judge_backend gemini \
  --judge_model gemini-2.5-flash \
  --output experiments/outputs/eval/<run_id>/judge_scores.jsonl
```

**판정 규약**:

- Judge prompt는 외부 지식 금지, context 근거만 사용, 길이 선호 금지 명시 (plan 3.3 원칙).
- `verdict` 결정 로직: `pass` if all three scores >= 4, `fail` if any < 3, else `partial`.

**검증**:

- 10개 질문에 대해 human annotation 샘플링 후 judge score와 Spearman 상관 > 0.5.

**DoD**:
- [ ] 동일 입력에 대해 temperature=0 고정 시 score 재현성 (±0 또는 ±1 이내 허용).
- [ ] `raw_response` 가 파싱 실패 시에도 `verdict: "fail_parse"` 로 graceful degrade.

---

### WP15. Significance testing util

**목표**: 6.6의 paired bootstrap 판정 규칙 자동화.

**편집 파일**:

- 신규 `experiments/shared/evaluate/significance.py`

**인터페이스**:

```python
# experiments/shared/evaluate/significance.py
from dataclasses import dataclass

@dataclass
class BootstrapResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    verdict: str   # "A_wins" | "B_wins" | "tie"

def paired_bootstrap(
    scores_a: list[float],
    scores_b: list[float],
    n_resample: int = 1000,
    confidence: float = 0.95,
    min_abs_diff: float = 0.03,
    seed: int = 42,
) -> BootstrapResult:
    """
    sign rule (6.6 규약):
      verdict = A_wins if (mean_diff >= min_abs_diff) AND (ci_low > 0)
      verdict = B_wins if (mean_diff <= -min_abs_diff) AND (ci_high < 0)
      else tie
    """
```

**보고서 연계**:

- `embed_benchmark` / `query_ablation` report.md 생성 시 pairwise 비교 표에 `verdict` 열 자동 추가.

**DoD**:
- [ ] 합성 데이터(A-B=0.05 deterministic)에서 `A_wins`.
- [ ] 합성 데이터(A=B random)에서 `tie`.

---

### WP16. Experiment 1–4 runbooks

**목표**: 개별 experiment를 단일 스크립트로 재현 가능하게.

**신규 파일**:

- `experiments/exp1_acquisition/run.sh`
- `experiments/exp2_embedding/run.sh`        # WP12 와 겸용
- `experiments/exp3_query_ablation/run.sh`
- `experiments/exp4_e2e/run.sh`

각 스크립트는 아래 공통 구조를 따른다:

```bash
#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)}"
OUT_ROOT="experiments/outputs/<expname>/${RUN_ID}"
mkdir -p "${OUT_ROOT}"

# 0. frozen queries 확보 (WP8)
# 1. 고정 corpus 준비
# 2. cases 실행
# 3. report 생성

echo "Done: ${OUT_ROOT}"
```

**Experiment 1 (Platform Acquisition) 실행 형태**:

```bash
# experiments/exp1_acquisition/run.sh
# 조건: extractor ∈ {gemini, vllm, raw}, retry ∈ {off, on}
for EXTRACTOR in gemini vllm; do
  for RETRY in off on; do
    python -m pipeline.step1_search \
      --questions data/test.csv \
      --sources scienceon \
      --extractor "${EXTRACTOR}" \
      $([ "${RETRY}" = "on" ] && echo "--retry") \
      --output-dir "${OUT_ROOT}/${EXTRACTOR}_retry_${RETRY}"
  done
done
python -m src.evaluate.eval_search \
  --results "${OUT_ROOT}" --source scienceon --output "${OUT_ROOT}/summary.json"
```

**Experiment 3 (Query Representation)**:

```bash
# 동일 corpus + 동일 embedding + query_transform만 변경.
# case YAML에서 query_transform 별 BenchmarkCase를 정의하고
# embed_benchmark가 각 case를 순회하며 metrics.json + report.md 생성.
python -m experiments.shared.evaluate.embed_benchmark \
  --cases experiments/shared/cases/exp3_query.yaml \
  --corpus "${OUT_ROOT}/corpus.jsonl" \
  --queries "${OUT_ROOT}/queries.jsonl" \
  --gold data/gold/scienceon_gold.json \
  --output "${OUT_ROOT}"
```

**Experiment 4 (E2E)**:

- best acquisition (Exp1) + best embedding (Exp2) + best query transform (Exp3) 조합으로 `run_pipeline.py` 실행.
- baseline: 현재 default pipeline.
- judge (WP14) + significance (WP15) 비교.

**DoD**:
- [ ] 각 스크립트가 `bash experiments/expN_*/run.sh` 단일 호출로 완주.
- [ ] 산출 디렉토리 구조가 0.3 규약과 일치.
- [ ] 실패한 case가 있어도 스크립트가 non-zero exit 대신 `report.md`에 fail로 기록 (단, 인프라 오류는 즉시 중단).

---

## 3. 데이터 스키마 레지스트리 (요약)

| ID | 이름 | 경로 | 정의 WP |
|---|---|---|---|
| S1 | search_meta_results.json | outputs/<run>/search/<ts>/ | 기존 pipeline (확장 없음) |
| S2 | BaseSearchClient doc record | n/a (Python return) | 기존 AGENTS.md |
| S3 | run_manifest.json | outputs/<run>/ | WP7 |
| S4 | retrieval hits (확장) | outputs/<run>/retrieval/<ts>/ | WP13 |
| S5 | final predictions | outputs/<run>/final/<ts>/ | WP7 |
| S6 | judge input/output | experiments/outputs/eval/<run_id>/ | WP14 |
| S7 | frozen queries | experiments/outputs/frozen/queries.jsonl | WP8 |

**S2 (고정, 기존)**:

```json
{"doc_id": "...", "title": "...", "abstract": "...",
 "authors": [...], "year": "...", "url": "...", "source": "ScienceON"}
```

**S4 (hits 레코드, 확장판)**:

```json
{
  "rank": 1, "score": 0.82, "doc_id": "CN123",
  "title": "...", "abstract": "...", "source": "ScienceON",
  "rerank_score": 0.91               // WP10 (optional)
}
```

---

## 4. 전체 진행 순서 권장

다음 순서로 진행하면 의존성이 깨지지 않고 각 단계에서 즉시 회귀 테스트가 가능하다.

1. **Day 1 — 안정화 기반**: WP0 → WP3 → WP7
2. **Day 2 — 수집 안정성**: WP1 → WP2 → WP4 (concurrency sweep → operating point 확정)
3. **Day 3 — 실험 인프라**: WP8 → WP9 → WP10 → WP11
4. **Day 4 — 평가 인프라**: WP14 → WP15 → WP13
5. **Day 5 — 실험 실행**: WP12 → WP16 (Exp1 → Exp2 → Exp3 → Exp4 순)

중간 회귀 체크(매 WP 완료 시):

```bash
./scripts/smoke_help.ps1                         # 모든 CLI --help 통과
python -m pytest src/tests/                      # 파이프라인 코어 테스트
python -m pytest experiments/tests/              # 실험 코드 테스트
rg "from pipeline" src/                          # 역방향 import 0건 확인
rg "from experiments" src/ pipeline/             # 실험 → 코어 역참조 0건 확인
git diff --name-only src/search/clients/scienceon_api_example.py   # 변경 0건
```

---

## 5. 열린 결정 사항 (구현 전 합의 필요)

다음 항목은 본 설계도가 strict하게 강제하지 않았으며, 첫 PR 전에 결정이 필요하다.

- **LLM judge backend**: gemini-2.5-flash 고정 vs 상황별 선택. 본 설계도는 default gemini-2.5-flash 를 가정.
- **BM25 tokenizer**: ko/en 혼합 corpus 에 대해 `whitespace` 단일 vs `mecab + 영어 공백` 하이브리드. 본 설계도는 WP9에서 `whitespace` default, mecab optional.
- **HyDE 생성 모델**: vLLM local(gpt-oss-20b) vs Gemini. vLLM이 재현성/비용 면에서 우세하므로 default는 vLLM.
- **Layer C gold set 크기**: 본 설계도는 파일럿 10~15질문 이후 확장 여부를 Exp2 결과 본 뒤 결정(plan 6.5-3 합의).
- **reranker 모델**: `BAAI/bge-reranker-v2-m3` default. `jina-reranker-v2-base-multilingual` 는 옵션.

---

## 6. 비완주 조건 (실험 중단 판정)

다음 중 하나라도 발생하면 실험을 중단하고 원인 분석부터:

- `run_manifest.json` 의 `git_sha` 가 dirty (untracked/uncommitted) 상태에서 Exp4가 실행됨.
- Exp2 의 동일 case 2회 실행 결과가 bit-identical 이 아님 (deterministic 전제 붕괴).
- WP4 sweep 에서 채택된 `max_concurrency` 를 적용했음에도 후속 run 에서 `rate_limit_rate > ε` 이 재현됨 (throttle/interval 설계 자체 문제 시사).
- judge output 파싱 실패율 > 10%.
