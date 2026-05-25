# 중요도 순 정렬 제거 실험 (No-Importance-Order Ablation)

## 목적
키워드 추출 프롬프트의 "중요도가 높은 순서대로 나열" 규칙(KO 6, EN 5/6)이 문서 수집 성능에 미치는 영향을 (k=30, n=10, m=50) 조건에서 정량 평가.

## 방법
- **변수 분리**: 원본 frozen queries(`queries_v3_chatgpt_gold41.jsonl`)의 키워드 *집합*은 보존하고, 고정 시드(seed=42)로 *순서*만 셔플 → `_frozen/queries_v3_shuffled_seed42.jsonl`.
- **이유**: 프롬프트만 바꿔 LLM을 재호출하면 키워드 선택 자체가 달라져 "순서" 변수만 격리되지 않음.
- **재구성**: 셔플된 키워드로 `build_search_terms`를 다시 실행해 rotation/truncation 검색식 생성.

## 실험 환경
- 라이브 ScienceON step1_search 실행 완료 (`live_shuffled_k30_n10_m50/`, runtime 764s, 1100 requests, 22 rate-limit retries, cache hit 42/miss 1078).
- 참고로 OR-commutative 캐시 replay 오프라인 시뮬레이터(`simulate_acquisition.py`)도 함께 검증용으로 유지 — 베이스라인 mismatch 0건으로 정합성 확인됨.

## 시뮬레이터 검증
- `search_meta_results.json`(실 베이스라인 run, k30_n10_m50)을 정답지로 사용해 시뮬레이터를 검증.
- 41개 질문 중 **mismatch 0건**, 시뮬레이터 acquisition gold rate = 실제 acquisition gold rate = **34/41 = 0.8293**.
- (참고: `metrics.json`의 `gold_found_rate=0.927`은 retrieval 단계 메트릭으로, acquisition 단계 0.829와 다름.)

## n=1 sanity check
n=1일 때 양쪽 모두 전체 키워드 set 1개 검색식만 사용 → OR commutativity로 동일 결과여야 함.
| 조건 | gold_found | rate | cache_hit | cache_miss |
|---|---|---|---|---|
| baseline n=1 | 28/41 | 0.6829 | 244 | 0 |
| shuffled n=1 | 28/41 | 0.6829 | 244 | 0 |

→ OR commutativity 가정 확인됨. 전체-키워드 단일 검색만으로는 양쪽 동일.

## 주 결과 (n=10, m=50, k=30) — 라이브 ScienceON 실행

| 조건 | gold_found | rate | mean docs/q | 비고 |
|---|---|---|---|---|
| **baseline** (중요도 순, 기존 run) | **34/41** | **0.8293** | 93.71 | — |
| **shuffled live** (seed=42) | **31/41** ↓ | **0.7561** ↓ | 94.80 | 22 rate-limit retries |
| Δ | **−3** | **−7.32%pt** | +1.09 | |

### 질문별 차이
- 양쪽 모두 회수: **31문**
- **baseline ONLY (중요도 순서일 때만 회수): Q2, Q5, Q45 — 3문**
- shuffled ONLY: **0문** (셔플이 도와준 사례 없음)
- 양쪽 모두 미회수: 7문 (Q1, Q4, Q22, Q40, Q41, Q44, Q47 — ScienceON 코퍼스 범위 한계)

## 해석
1. **n=1 (전체 키워드 set만)**: 양쪽 동일 28/41 (OR commutative). 순서는 truncation 단계에서만 작동.
2. **n=10 (truncation rotation 포함)**:
   - baseline: 28 + 6 = **34** (중요도 순 덕에 좁아진 검색식이 핵심 단어를 보존)
   - shuffled: 28 + 3 = **31** (랜덤 truncation은 절반만 살림)
3. **mean docs/q 차이 미미(94 vs 94)**: m=50 budget은 양쪽 다 채움. 즉 *얼마나* 모으는가가 아니라 *어떤 문서를* 모으는가의 문제.
4. **단방향 효과**: shuffled ONLY=0개 → 셔플이 새 gold를 가져오는 일은 없음. 중요도 순서가 strictly 도움.

## 라이브 재현 명령
```bash
source scripts/run_experiment_setup.sh   # shrag conda env + .env 로딩

SLICED="experiments/outputs/exp_no_importance_order/_frozen/queries_v3_shuffled_seed42_n10.jsonl"
python -m experiments.exp1_kmn.slice_frozen_queries \
  --input experiments/outputs/exp_no_importance_order/_frozen/queries_v3_shuffled_seed42.jsonl \
  --output "$SLICED" --n 10

OUT_DIR="experiments/outputs/exp_no_importance_order/live_shuffled_k30_n10_m50"
mkdir -p "$OUT_DIR/search"
python -m shrag.pipeline.steps.step1_search \
  --questions experiments/outputs/exp1_kmn/questions_gold41.jsonl \
  --sources scienceon --extractor vllm \
  --vllm-url "$GPT_OSS_VLLM_URL" --vllm-model "$GPT_OSS_VLLM_MODEL" \
  --extractor-temperature 0 --frozen-queries "$SLICED" \
  --target-documents 50 --scienceon-max-pages 3 \
  --scienceon-max-concurrency 2 --scienceon-min-interval-sec 0.5 \
  --output-dir "$OUT_DIR/search"
```

## 산출물
- `build_shuffled_queries.py` — 셔플 frozen queries 생성기
- `check_cache_coverage.py` — 캐시 커버리지 진단
- `simulate_acquisition.py` — OR-commutative 캐시 replay 시뮬레이터 (검증용)
- `_frozen/queries_v3_shuffled_seed42.jsonl` / `..._n10.jsonl` — 셔플된 frozen queries (41q)
- `live_shuffled_k30_n10_m50/search/...` — **라이브 ScienceON acquisition 결과** (search_meta_results.json, search_documents.jsonl)
- `baseline_n10_m50/`, `shuffled_seed42_n10_m50/` — 검증용 오프라인 시뮬레이션 산출물
