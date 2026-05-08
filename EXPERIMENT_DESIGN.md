# SHRAG 실험 설계서

> 작성: 2026-05-05
> Target: 사용자 역할 4축 (키워드 추출 / 임베딩 모델 / 임베딩 전략 / 문서 수집 평가) 의 논리적 완결성
> 기반: `CLAUDE.md` 예상 빈약 포인트 + 현재 구현 (`shrag/`) + 사용자 피드백
> Scope 원칙:
> - **PRIMARY**: retrieval 단계까지 (step1~step4). gold doc Hit@k 기반 평가.
> - **NOT IN SCOPE**: step5 답변 생성 및 답변 품질(LLM-as-a-judge) 평가는 본 plan에서 수행하지 않음.
> - **LLM 통일**: 키워드 추출 = vLLM gpt-oss-20b (T=0). 다른 LLM(gemini/codex/claude)는 본 plan에서 사용하지 않음.

---

## 0. 사용자 plan 사전 검토

### 0.1 타당성 검토

| 항목 | 문제점 | 사용자 추가 요청 | 결정 |
|---|---|---|---|
| "각 키워드별 수집 문서는 10개..?" | ScienceON `row_count`=10 × `max_pages`=5 = term당 최대 50건. 새 변수 아닌 듯 | row_count도 변수로 | **수용**: (k=term당 수집 문서, m=target_documents per-lang, n=검색어 수) **3변수 통합 설계** §0.1.1 |
| "target 50일 때 키워드 수 vs m일 때 키워드 수 차이?" | (k, n)이 m이 binding 상태로 confounded | factorial 불가피? | **부분 부정**: m을 cap-unbinding 값으로 두면 (k, n) 단독 sweep 가능. §0.1.1 3-phase 설계 |
| "768로 고정 후 비교 가능/의미?" | bge-m3 Matryoshka 지원 → 가능 | 좋음 | native 1024 vs 768 + bge-m3@768 truncated 두 비교 모두 보고 |

#### 0.1.1 (k, m, n) 통합 실험 설계 — 3-phase (채택)

**변수 정의**:

| 기호 | 의미 | 현재값 | sweep 범위 |
|---|---|---|---|
| **k** | 검색어 1개당 ScienceON 수집 docs (= row_count × max_pages) | 10 × 5 = 50 | {10, 20, 30, 40, 50} |
| **m** | per-language target_documents (dedup 후 cap) | 20 | Phase A: 70 고정 / Phase B: {10, 20, 30, 50, 70} |
| **n** | per-language 검색어 수 (rotation OR-string) | extractor가 만드는 모든 (~5-10) | {1, 3, 5, 8, 10} |
| s | top-5에 gold 포함 (o/x) | — | 종속변수 |
| t | 41Q step1 + step3 + step4 총 시간 (sec) | — | 종속변수 |

**Confoundation 분석**:

```
n ↑   →  ScienceON 호출 횟수 ↑    →  raw docs ↑
k ↑   →  호출 1회당 응답 docs ↑   →  raw docs ↑
m     →  unique docs cap          →  m이 binding이면 (k,n)의 효과 가림

         dedup
raw docs ────→ unique docs ──cap by m──→ embedding 대상
```

→ m이 binding되는 한 (k,n)의 main effect는 m 뒤에 가려짐. 해결: m을 cap-unbinding 값으로 두고 (k, n) 단독 sweep.

**3-phase 설계**:

| Phase | 변수 | 고정 | Cells | 목표 |
|---|---|---|---|---|
| **A: cap 풀기** | (k, n) 격자 | m=70 (§0.1.2 ceiling=30/41 per-q에 두 배 여유) | 5×5=25 | (k*, n*) 도출 |
| **B: cap 좁히기** | m | k=k*, n=n* | 5 | m* 도출 — 가성비 cap |
| **C: cross-check** | (k, n) ± 1 단계 | m=m* | 4-6 | m이 binding된 환경에서 (k*, n*) 재확인 |

**총 cell**: ~35 (factorial 5×5×5=125 대비 1/4 미만)

**시간 측정 + 보고 (t)**:

t = 41Q step1 + step3 + step4 총 시간 (step5 제외 — 본 plan scope 외).

| 임계 | 의미 |
|---|---|
| t ≤ 5분 | live demo 가능 (bonus) |
| t ≤ 10분 | replicate(N=3) 30분 안에 가능 (soft target) |
| t ≤ 20분 | one-pass sweep 가능 (informational) |
| t > 20분 | 기록 보존. 우선순위 ↓ but **결과 폐기 안 함** |

**평가 시각화**: 모든 cell에 대해 (s, t) 2D Pareto plot. cell 탈락 없음. 사용자가 trade-off를 직접 본다.

**현실 체크 (이전 41Q E2E 측정)**:
- gte run (cold cache, target=20): step1 16분 + step3 0.5분 + step4 0.2분 ≈ 16.7분
- bge_m3 run (cold cache, target=20): step1 42분 + step3 1분 + step4 0.2분 ≈ 43분
- 캐시 warm 시 step1 ~수십초. → Phase A 첫 cell만 cold, 나머지 24+ cell은 warm.

#### 0.1.2 Empirical Search Recall Ceiling (과거 17개 run 분석)

| 지표 | 값 |
|---|---|
| 단일 run 최대 corpus gold 매칭 | **31/41 = 75.6%** (모든 LLM family에서 동일 천장 → ScienceON 검색 자체의 한계) |
| 단일 run 최대 per-question gold 매칭 | 30/41 = 73.2% |
| Union of ALL 17 past corpora | 37/41 = 90.2% |
| 어떤 시도에서도 0 hit | 2/41 (qid=44 `ART002777203`, qid=41 `NPAP11397415`) |

**시사점**:
1. **search recall ceiling ~75%** — m을 무한히 키워도 천장 못 깸. m=70은 이론 천장의 두 배 여유로 cap-unbinding 보장.
2. **m=10, 5는 search recall 급락** (target_sweep: 8/41, 5/41) — m ≥ 20 권장.
3. **2 gold (qid 41, 44) 영구 미스**: 분석 시 "39 reachable / 41 total" split 보고.

### 0.2 보완 항목

| 누락 항목 | 결정 |
|---|---|
| **검색 변동성 제어** (재현성) | frozen cache + frozen-queries로 변동 제거 시 N=1 충분. 그렇지 않으면 N=3. (§0.4 Q2) |
| **Reranker** | Exp7로 추가. 사용자 역할 외부지만 retrieval pipeline 가치 측정에 핵심. (§0.4 Q3) |
| **Negative case 분류** | 모든 Exp 보고에 의무 — failure taxonomy F1/F2 (§0.2.1) |

#### 0.2.1 Failure Mode Taxonomy

각 cell에서 **Hit@5=0 케이스**를 다음 2 카테고리로 분류:

| 카테고리 | 정의 | 진단 |
|---|---|---|
| **F1. Search failure** | gold doc 자체가 step1 단계에서 수집 안됨 | 키워드 추출 또는 ScienceON 응답 부족 |
| **F2. Retrieval failure** | gold 수집됐으나 top-5 진입 못함 | 임베딩/retrieval 약점 |

각 실험 보고 표:

| Cell | n_total | F1 | F2 | OK |
|---|---|---|---|---|

> step5 답변 생성/평가는 본 plan scope 외이므로 F3 (Generation failure)는 분류하지 않음.

### 0.3 실험 순서 — 순차 + spot-check

```
Exp1 (k, m, n: 3-phase)         → m*, k*, n* 도출
        ↓
Exp2 (검색어 수 marginal 분석)   → n* 정밀화 (선택, Exp1 Phase A에 흡수 가능)
        ↓
Exp3 (embedding model)          → gte vs bge-m3
        ↓
Exp4 (embedding strategy)       → text composition
        ↓
Exp7 (reranker, 선택)           → retrieval pipeline 천장 측정
        ↓
(spot check) 다른 model에서도 (k*, n*, m*) 유효한지 sample
```

main effects 파악, 비용은 sequential 15-25 cell + Exp1 Phase A의 25 cell ≈ 40-50 cell.

### 0.4 추가 질문에 대한 답변 (Q&A)

#### Q1. 캐시 warm 기준이란?

ScienceON cache 키 = `(source, term, page, row_count, fields)` SHA1. **"warm"** = 현재 (k, n)이 부르는 모든 키 튜플이 디스크에 이미 있음 → cache miss = 0.

| 상태 | 정의 | step1 시간 |
|---|---|---|
| cold | `outputs/_shared_cache/scienceon/` 빈 상태 | 16-42분 (41Q) |
| partial warm | 일부 cache hit. 키워드 변동 또는 (k,n) 변경으로 새 term 발생 | 5-15분 |
| **warm** | 현재 (k,n)이 부르는 모든 키 튜플이 cache 됨. miss=0 | ~수십초 |
| frozen | warm + 키워드도 `--frozen-queries`로 고정 → 100% 재현성 | ~수십초 + 결정성 |

**warm 만들기 워크플로**:
1. Phase A 첫 cell `(k_max=50, n_max=10, m=70)` cold start로 한 번 실행 → 모든 (term, page) 응답 적재
2. `--emit-frozen-queries` 로 키워드 set을 JSONL에 고정 → 이후 cell들 키워드 동일
3. 같은 또는 더 작은 (k, n)은 모두 cache hit → step1 ~수초

`Phase A 첫 cell만 cold (16-42분), 나머지 cell은 warm (각 ~3-5분 = step3+step4 위주)` 가 본 plan의 시간 가정.

#### Q2. min_interval_sec=0.5로도 429 발생? N=3 replicate 필요?

**429 실측 (직전 bge_m3 41Q run)**:

| Run | min_interval_sec | max_concurrency | 429 횟수 | step1 |
|---|---|---|---|---|
| gte | 0.5 | 2 | 4건 | 16분 |
| bge_m3 | 0.5 | 2 | 2건 | 42분 |

**해석**:
- 429는 발생하지만 **매우 적음** (~0.2-0.4%)
- bge_m3 step1 42분 ≠ 429 retry 폭발. **다른 원인** (vLLM 키워드 변동으로 cache miss 多 + 네트워크 등)
- ScienceONAdapter retry 메커니즘이 적은 429를 흡수 — 사용자에게 영향 거의 없음

**더 보수적이면 거의 0**:
```
--scienceon-max-concurrency 1
--scienceon-min-interval-sec 1.0
```
시간 1.5-2배 늘지만 안정.

**N=3 vs N=1 결정**:

| 변동성 원인 | 해결책 |
|---|---|
| 429 timing → docs 풀 변동 | 보수 throttle + cache warm |
| vLLM 키워드 추출 nondeterminism | `--frozen-queries` (✓ 완전 제거) |
| 임베딩 모델 deterministic | (no-op) |

→ **frozen cache + frozen queries 시 N=1 충분**. 그렇지 않으면 N=3.

#### Q3. 리랭커 모델 선택 + Exp7 설계?

**모델 후보 (한·영 academic 적합)**:

| 모델 | params | 권장 |
|---|---|---|
| **BAAI/bge-reranker-v2-m3** | 568M | ⭐ 1순위 (multilingual bge-m3 family) |
| Alibaba-NLP/gte-multilingual-reranker-base | 305M | encoder=gte 페어링 시 |
| dragonkue/bge-reranker-v2-m3-ko | 568M | 한국어 비중 클 때 |

자세한 Exp7 설계는 §7 참조.

#### Q5. 성공기준 top-5 타당성?

**유지**. step5 max_rank=5 기준이 production 실제 슬롯과 일치. 단 reranker rank 재정렬 효과는 binary로 가려질 수 있어 보조 지표 추가:

| 지표 | 역할 |
|---|---|
| **Hit@5** | PRIMARY (production success) |
| Hit@10 | recall headroom — reranker 잠재력 |
| Hit@20 | search-stage 품질 |
| **MRR** | rank 평균 (재정렬 효과 핵심) |
| **mean_gold_rank** | gold 평균 rank (낮을수록 좋음) |

PRIMARY 3개(Hit@5 + MRR + mean_gold_rank)로 cell ranking. Hit@10/20은 진단용.

---

## 1. 측정 지표 표준화

모든 실험 공통 지표. **gold dataset = `experiments/gold_scienceon/artifacts/scienceon_gold_full_only.jsonl`** (41 질문 × 1 gold doc).

### 1.1 Retrieval 지표 (PRIMARY)

| 지표 | 정의 | 역할 |
|---|---|---|
| Hit@1 | rank 1에 gold | 보조 |
| Hit@3 | top-3에 gold | 보조 |
| **Hit@5** | top-5에 gold (production success, step5 max_rank=5와 일치) | **PRIMARY** |
| Hit@10 | top-10에 gold (recall headroom) | 보조 |
| Hit@20 | top-20에 gold (search-stage 품질) | 보조 |
| **MRR** | mean reciprocal rank | **PRIMARY** (rank-sensitive) |
| **mean_gold_rank** | gold rank 평균 (낮을수록 좋음) | **PRIMARY** (재정렬 효과) |
| Coverage | gold doc이 step1에서 수집됐는지 | 진단 (F1 분류용) |

출처: `eval_gold_judge.compute_retrieval_metrics` + 확장 (Hit@10/20, mean_gold_rank).

### 1.2 Step1 수집 지표

| 지표 | 정의 |
|---|---|
| total_unique_docs | 질문당 dedup 후 unique doc 수 |
| docs_by_lang | {korean, english} 각각 수집한 unique doc 수 |
| zero_doc_rate | 0건 수집된 질문 비율 |
| step1_runtime_sec | step1 총 시간 |

### 1.3 비용/속도

| 지표 | 단위 |
|---|---|
| step1_search_sec | s — ScienceON HTTP 호출 |
| step3_build_vectordb_sec | s — 임베딩 비용 |
| step4_retrieve_sec | s — query encoding + dense search |
| **t** = step1 + step3 + step4 | s — Pareto 평가 축 |

### 1.4 재현성

각 cell:
- frozen cache + frozen queries 시 **N=1**, 그렇지 않으면 **N=3** mean ± std
- vLLM T=0 + (가능 시) seed 고정

---

## 2. 실험 매트릭스

| Exp | 변수 | 고정 | Cells | 산출 |
|---|---|---|---|---|
| **Exp1** (3-phase) | (k, n) 격자 → m sweep → cross-check | encoder=gte, strategy=3T+A | ~35 | (k*, n*, m*) |
| **Exp2** (선택) | 검색어 수 marginal contribution 분석 | Exp1 결과 | ~6 | n* 정밀화 (Exp1 Phase A에 흡수 가능) |
| **Exp3** | encoder ∈ {gte-768, bge-m3-1024, bge-m3-768 truncated} | (k*, n*, m*), strategy=3T+A | 3 | best encoder |
| **Exp4** | strategy ∈ {T, A, T+A, 2T+A, 3T+A, 5T+A, T+A_trunc, sliding_window} × encoder ∈ {best, runner-up} | (k*, n*, m*) | 8×2=16 | (encoder, strategy) interaction |
| **Exp7** (선택) | reranker ∈ {none, bge-rerank-v2-m3 × top_n {20/50/100}, gte-rerank, dragonkue-ko} | best (encoder, strategy) | 6 | rerank 효과 |

**총 cell**: ~60 (Exp1 35 + Exp2 6 + Exp3 3 + Exp4 16 + Exp7 6).

---

## 3. Exp1 — (k, m, n) 통합 sweep

§0.1.1의 3-phase 설계 그대로 실행. 자세한 변수/cell 정의는 §0.1.1 참조.

**측정**:
- PRIMARY: Hit@5, MRR, mean_gold_rank
- 보조: Hit@1/3/10/20, Coverage, F1/F2 비율
- 비용: t (step1+step3+step4)

**분석**:
- Phase A heatmap: (k, n) → Hit@5 (cell color)
- Phase B 곡선: m → Hit@5, Coverage 두 줄
- Phase C: ± 1 단계에서 (k*, n*) plateau 검증
- Pareto plot: (s, t) — 모든 cell 시각화

**산출**: (k*, n*, m*) 결정. Exp3-7의 baseline.

---

## 4. Exp2 — 검색어 수 marginal contribution (선택)

Exp1 Phase A에서 (k, n) 격자가 이미 n sweep을 포함하므로 별도 실험 필요성 ↓. 단 **검색어별 marginal recall** (검색어 N → N+1 추가 시 새 unique doc 수)을 보고 싶을 때만.

**셀**: n ∈ {1, 2, 4, 6, 8, 10}, k=k*, m=m* 고정

**측정**: marginal_new_docs_per_term, Hit@5 변화

**플롯**: x=n, y=Hit@5 + marginal_new_docs

> Exp1 Phase A 결과로 충분하면 Exp2 생략 가능.

---

## 5. Exp3 — 임베딩 모델 비교

### 5.1 셀

| Encoder | dim | 비고 |
|---|---|---|
| gte-multilingual-base | 768 native | 현재 default |
| bge-m3 | 1024 native | 강력 multilingual |
| bge-m3 | 768 truncated (Matryoshka) | 같은 dim에서 비교 |

### 5.2 통제
- (k*, n*, m*) Exp1에서 도출
- strategy = 3T+A
- **frozen-queries로 step1 출력 fix** → step3부터만 모델 다르게 = 같은 docs 풀 위에서 임베딩만 비교

### 5.3 측정
- PRIMARY: Hit@1/5, MRR, mean_gold_rank
- 비용: step3 (per-doc 임베딩 시간) + step4 (retrieval)
- 메모리: vector_db CSV 크기

### 5.4 표 양식

| Encoder | Hit@5 | MRR | mean_gold_rank | step3 sec | step4 sec | DB size |
|---|---|---|---|---|---|---|

---

## 6. Exp4 — 임베딩 전략 비교

### 6.1 셀 (text composition)

| ID | 전략 | 비고 |
|---|---|---|
| T | title only | baseline |
| A | abstract only | baseline |
| T+A | title + abstract | 단순 |
| 2T+A | title×2 + abstract | weighting 약 |
| **3T+A** | title×3 + abstract | 현재 default |
| 5T+A | title×5 + abstract | over-weighted |
| T+A_trunc | title + abstract[:512] | length normalization |
| sliding_window | abstract을 256-token chunks로 분할, 별도 row | chunk-level |

### 6.2 통제
- (k*, n*, m*) Exp1
- encoder ∈ {best, runner-up} from Exp3 (8 strategy × 2 encoder = 16 cell)

### 6.3 cross-effect
- **encoder × strategy interaction** 측정 (ANOVA two-way → Hit@5)
- best (encoder, strategy) combination 도출

---

## 7. Exp7 — Cross-encoder reranker (선택)

§0.4 Q3 답변 상세 그대로.

### 7.1 셀

| reranker | top_n_to_rerank | 비고 |
|---|---|---|
| none (baseline) | — | dense top-5 |
| bge-reranker-v2-m3 | 20 | 가벼움 |
| bge-reranker-v2-m3 | 50 | 표준 |
| bge-reranker-v2-m3 | 100 | recall 최대 |
| gte-multilingual-reranker-base | 50 | gte family 페어 |
| dragonkue/bge-reranker-v2-m3-ko | 50 | 한국어 fine-tune |

→ 6 cell. 비용: top-50 × 41Q = 2050 inference, 모델당 ~5분.

### 7.2 통제
- (k*, n*, m*) Exp1
- encoder = best from Exp3
- strategy = best from Exp4

### 7.3 측정
- PRIMARY: Hit@5, Hit@10, MRR, mean_gold_rank
- **mean_gold_rank_delta** = before_rerank - after_rerank (음수일수록 좋음)
- step4 + rerank 추가 latency

### 7.4 가설
- reranker가 dense top-5 밖 (rank 5-10)에 들어온 gold를 top-3로 끌어올림
- Hit@5 +5~15%pt, MRR +0.10 이상 기대

### 7.5 실제 실행 (2026-05-06 ~ 05-08, 갱신)

원래 plan은 6 cell sweep (top_n {20/50/100} × 3 model)이었지만 실제 진행 분기:

| 분기 | 셀 | 결과 요약 |
|---|---|---|
| **Exp7** | bge-v2-m3 cand={20, 50} | Hit@5 0.902 ceiling, top20=top50 동일. cand sweep은 Exp8에서 더 정밀 |
| **Exp8** | bge-v2-m3 cand={5, 7, 10, 20, 50} | **cand=5가 best** (Hit@3=0.902 ceiling, MRR 최대, cost 최저). wider pool은 노이즈 introduce |
| **Exp7b** | 3 model × cand=5 fair sweep (bge-v2-m3, gte-rerank, dragonkue-ko) | **dragonkue-ko 채택**: Hit@1 0.780 (vs bge 0.732 +4.8pp), MRR 0.833. gte-rerank는 dense baseline보다도 Hit@1 ↓ → 기각 |
| **Exp7c** | subprocess 격리 + warmup pass + N=2 측정 | dragonkue 0.177 s/q ≈ bge-v2-m3 0.174 s/q (1.7% noise) — **같은 architecture, 속도 동등 검증**. gte-rerank 0.074 s/q 빠르지만 품질 손실로 채택 X |

**최종 채택**: `dragonkue/bge-reranker-v2-m3-ko`, candidates=5.
- Hit@5 ceiling 0.902 유지하며 Hit@1 0.610 → 0.780 (+0.170), MRR 0.723 → 0.833 (+0.110)
- top_n=100 cell은 미실행 (Exp8에서 cand=50조차 cand=5보다 못함이 도출되어 우선순위 ↓)

상세: `experiments/outputs/FOLLOWUP_EXPERIMENT_SUMMARY.md` §4

---

## 8. 통제 변수 매트릭스

| 변수 | Exp1 | Exp2 | Exp3 | Exp4 | Exp7 |
|---|---|---|---|---|---|
| (k, m, n) | **vary** (3-phase) | n vary, k=k*, m=m* | (k*, n*, m*) | (k*, n*, m*) | (k*, n*, m*) |
| encoder | gte | gte | **vary** | **vary** | best |
| strategy | 3T+A | 3T+A | 3T+A | **vary** | best |
| reranker | none | none | none | none | **vary** |
| extractor | vllm gpt-oss T=0 | 동 | 동 | 동 | 동 |
| ScienceON cache | warm (Phase A 1회 cold) | warm | **frozen** | frozen | frozen |
| keywords | live | live | **frozen** | frozen | frozen |
| replicates | N=1 (frozen 시) ~ N=3 | 동 | N=1 | N=1 | N=1 |

---

## 9. 보고 표준 형식

각 실험 산출물 4종:

1. **Headline plot**: main effect 시각화
   - Exp1: (k, n) heatmap → Hit@5 / Pareto plot (s, t)
   - Exp3: encoder bar chart Hit@5 + step3 sec
   - Exp4: encoder × strategy heatmap
   - Exp7: rank-shift histogram (gold rank before vs after rerank)
2. **Aggregate table**: cell별 PRIMARY 3 metric (mean ± std)
3. **Failure analysis**: Hit@5=0 케이스 분류 (F1 / F2 빈도) — 5건 케이스 정성 분석
4. **One-line takeaway**: e.g. "Hit@5 best at (k=20, n=5, m=30): 0.78 ± 0.03, t=8.2분"

---

## 10. 산출물 디렉토리 규약

```
experiments/outputs/
├── exp1_kmn/
│   ├── phaseA/k20_n5_m70/{search,retrieval}/
│   ├── phaseB/m30/...
│   ├── phaseC/k20_n5_m30/...
│   ├── _aggregate.csv
│   └── _report.md
├── exp3_encoder/
│   ├── gte_768/{search,retrieval}/
│   ├── bge_m3_1024/...
│   ├── bge_m3_768_trunc/...
│   ├── _aggregate.csv
│   └── _report.md
├── exp4_strategy/
│   ├── gte_3TA/...
│   ├── bge_TA_trunc/...
│   └── ...
└── exp7_reranker/
    ├── none/
    ├── bge_v2m3_top50/
    └── ...
```

`_aggregate.csv`: cell별 PRIMARY 3 metric + t + F1/F2 카운트.

---

## 11. 사용자 역할 4축 매핑

| 역할 | 답하는 실험 | 산출물 |
|---|---|---|
| **키워드 추출 로직** | Exp1 (n axis), Exp2 | "왜 N=n* 검색어가 적정인가" 정량 근거 + marginal recall 곡선 |
| **임베딩 모델 선택** | Exp3 | "왜 X encoder인가" — Hit@5 + 비용 비교 |
| **임베딩 전략 선택** | Exp4 | "왜 3T+A 또는 X 전략인가" — encoder×strategy interaction |
| **문서 수집 성능 평가** | Exp1 (k, m), Exp7 | "왜 (k*, m*) 인가" + "reranker가 retrieval 천장을 어디까지 끌어올리나" |

---

## 12. 권장 실행 순서 + 시간 예산

| 주차 | 실험 | 가정 |
|---|---|---|
| Week 1 | Exp1 Phase A (25 cell, 첫 cell만 cold ~17-43분, 나머지 warm ~3-5분) | ~3시간 |
| Week 1 | Exp1 Phase B + C (10 cell, frozen cache) | ~1시간 |
| Week 2 | Exp3 (3 cell, frozen) | ~1시간 |
| Week 2 | Exp4 (16 cell, frozen) | ~3시간 |
| Week 3 | Exp7 (6 cell) | ~1시간 |
| Week 3 | 분석 + 보고서 | — |

**vLLM 시간 합계**: 약 9시간 + 분석.

---

## 13. 실패 시 대안 (자원 부족)

| 우선순위 | 실험 | 이유 |
|---|---|---|
| **1** | Exp3 | 사용자 역할 핵심, 단일 변수 명확 |
| 2 | Exp1 (Phase A만) | (k, n, m) 천장 측정 |
| 3 | Exp4 | encoder 결정 후 전략 fine-tune |
| 4 | Exp7 | reranker 효과 dramatic 가능 |
| 5 | Exp2 | Exp1 Phase A로 흡수 가능 |

---

## 13a. Embedding cache (2026-05-06 추가)

### 동기

GPU(RTX 4070 12GB) step3 인코딩이 cell당 3-30분 → 풀 sweep 비현실적. 같은 doc은 같은 (encoder, embedding_mode) 조합에서 항상 같은 임베딩이므로 결정론적 캐시 가능.

### 설계

**Storage**: SQLite 단일 파일 `outputs/_embedding_cache.db` (WAL mode, atomic writes)

**Schema**:
```sql
CREATE TABLE embeddings (
  encoder_id  TEXT NOT NULL,    -- "{model_name}@{truncate_dim}", e.g. "Alibaba-NLP/gte-multilingual-base@768"
  mode        TEXT NOT NULL,    -- canonical embedding_mode (alias normalized: "3*title+abstract" → "3T+A")
  doc_id      TEXT NOT NULL,    -- ScienceON CN
  text_sha1   TEXT NOT NULL,    -- SHA1(embedding_text) — invalidates on doc content change
  dim         INTEGER NOT NULL,
  vector      BLOB NOT NULL,    -- numpy float32 bytes
  created_at  TEXT NOT NULL,
  PRIMARY KEY (encoder_id, mode, doc_id, text_sha1)
);
```

### 위험 / 제약 매핑

| 위험 | 대응 |
|---|---|
| R1. embedding_text 정의 코드 변경 | text_sha1 키에 포함 → 새 텍스트면 자동 cache miss |
| R2. ScienceON doc 내용 변경 | text_sha1 변동으로 stale 자동 무효화 |
| R3. 인코더 모델 변경 | encoder_id에 model_name + truncate_dim 포함 → 자동 분리 |
| R4. embedding_mode alias ("3T+A" vs "3*title+abstract") | normalize_embedding_mode()로 canonical form 키 생성 |
| R5. 동시 write | SQLite WAL + INSERT OR REPLACE, 30s busy_timeout |
| R6. Truncate dim 변경 | encoder_id에 dim 포함 → 768 vs 1024 자동 분리 |

### 예상 효과

| 시나리오 | Cold | Warm (full hit) | 절감 |
|---|---|---|---|
| Exp4 (8 strategy × same corpus) | 8 × 3분 = 24분 | 3분 + 7 × ~5초 = 4분 | 83% |
| Phase B m sweep (m∈{10..70}, 같은 k,n) | 5 × 3분 = 15분 | 3분 + 4 × ~5초 = 3.5분 | 77% |
| Phase A k 비교 (k=10 ⊂ k=50 corpus) | 2 × 3분 = 6분 | 3분 + ~5초 = 3분 | 50% |
| Encoder 변경 시 (Exp3) | 무관 (encoder_id 분리) | 무관 | 0% |
| Embedding mode 변경 시 (Exp4) | 무관 (mode 분리) | 무관 | 0% |

→ Exp4 / Phase B의 경우 cache 효과 결정적 (corpus 동일, 모드/cap만 변경).

### 통합

`shrag/utils/embedding_cache.py` (EmbeddingCache 클래스), `shrag/features/embedding_processor.py:generate_batch_embeddings_cached()` 추가. encoder config의 `cache_db_path` 필드가 활성화 트리거.

### 검증 (smoke test 결과)

| Test | 결과 |
|---|---|
| Round-trip put/get | PASS |
| Mode alias normalization (3*title+abstract → 3T+A) | PASS |
| Encoder isolation (gte vs bge-m3 분리) | PASS |
| Mode isolation (3T+A vs T 분리) | PASS |
| text_sha1 invalidation (doc 내용 변경 → miss) | PASS |
| 100% cache hit run: model 로드 자체 스킵 | PASS (10 docs RUN1 9.1s → RUN2 3.2s) |
| Cached vector vs fresh encode 동일성 | PASS (max abs diff = 0.0) |

### Cell 실행 순서 권장

GPU 캐시 priming 효과 극대화를 위해 **큰 cell부터** 실행:
- Phase A: k=50 cell 먼저 (full corpus) → k=10 cell (subset, 100% hit 기대)
- Phase B: m=70 먼저 → m≤70 모두 hit
- Exp3: encoder당 corpus 동일, 첫 encoder만 cold
- Exp4: 첫 strategy만 cold, 나머지 7개 strategy는 corpus 동일 mode만 다름

---

## 13b. 1Q 대표 timing 측정 (2026-05-06 추가)

### 동기

41Q aggregate 측정은 batch optimization 효과(GPU 병렬, vectordb 단일 빌드 비용 분산)로 single-query latency를 흐릴 수 있다. "live demo 가능?" (t≤5분) 같은 **production-realistic** 결정은 단일 query 기준이 적합.

### 방법

`experiments/exp1_kmn/timing_1q.py`:
1. 41Q frozen-queries에서 지정한 qid(s)만 필터링 + n-slice
2. 그 question만으로 step1 → step3 → step4 실행
3. step1/3/4 각 단계 wall-clock + total + docs_count 기록
4. 출력 JSON: `{label, qids, k, n, m, step1_sec, step3_sec, step4_sec, total_sec, docs_count}`

### 사용 패턴

대표 query 3개 평균 (qid 1=ANN, qid 5=의료영상, qid 20=NLP) 권장. 단일 qid 편향 방지.

```bash
for qid in 1 5 20; do
  python -m experiments.exp1_kmn.timing_1q \
    --qids $qid --k 10 --n 1 --m 30 \
    --encoder configs/query_encoder/config_gte-multilingual-base.json \
    --output experiments/outputs/_timing/cell_k10_n1_m30_q${qid}.json
done
```

### 측정 임계값 재조정 (1Q 기준)

41Q t = 41 × 1Q t (대략) 가정 시:

| 1Q 임계 | 41Q 환산 | 의미 |
|---|---|---|
| 1Q ≤ 7s | 41Q ≤ 5분 | live demo 가능 |
| 1Q ≤ 15s | 41Q ≤ 10분 | replicate(N=3) 30분 가능 |
| 1Q ≤ 30s | 41Q ≤ 20분 | one-pass sweep 가능 |

→ 단 step3는 corpus 단위 한 번만 실행되므로 1Q 곱셈 환산이 정확하지 않음. step1+step4만 per-query 비례. step3는 별도로 corpus 크기 함수로 보고.

---

## 13c. Phase A 결과 (2026-05-06 실측)

### 슬라이서 버그 → invalidate된 측정

초기 슬라이서 (`experiments/exp1_kmn/slice_frozen_queries.py` v1)가 `search_terms` 필드만 슬라이스하고 `search_terms_by_lang`는 풀 리스트 그대로 둠. 그러나 `step1_search.py:173`은 `search_terms_by_lang`를 우선 읽음 → n axis가 측정되지 않음 (모든 cell이 사실상 풀 search_terms 사용, m=70 cap binding으로 동일 결과). 5 cells × ~30분 GPU 시간 낭비.

**수정 (slice_frozen_queries v2)**: `search_terms_by_lang.korean/english` 둘 다 슬라이스 + meta 필드. 검증 단위 테스트 추가.

### 압축 후 valid 측정

| Cell | Hit@5 | mean_gold_rank | gold_found | total t |
|---|---|---|---|---|
| k=10 (m=70, full search_terms) | 0.878 | 2.00 | 0.902 | 127s |
| k=50 (m=70, full search_terms) | 0.854 | 2.45 | 0.927 | 1107s (cold) / 206s (k=50_n1 warm) |

**관찰**: k=10 > k=50 — fewer docs/term → less noise → better Hit@5. (n axis는 슬라이서 수정 후 재실험 필요.)

### Phase B (m sweep at k=10 + 풀 search_terms)

| m | Hit@5 | gold_found | t |
|---|---|---|---|
| 10 | 0.683 | 0.683 | 34s |
| 20 | 0.854 | 0.878 | 101s |
| **30** ⭐ | **0.878** | 0.902 | 114s |
| 50 | 0.878 | 0.902 | 127s |
| 70 | 0.878 | 0.902 | 131s |

**m\* = 30** (Hit@5 plateau onset)

### 잠정 best_config

```json
{"k": 10, "n": "TBD (slicer-bug invalidated)", "m": 30, "encoder": "gte", "embedding_mode": "3T+A", "hit_at_5": 0.878}
```

n 정확 측정은 슬라이서 수정 후 후속 실험에서.

---

## 14. 미해결 / 사용자 결정 필요

- **번역 augmentation**: 한국어 질문을 영어로 번역해서 영어 검색어 보강. 사용자 역할 scope 외이지만 retrieval pool 확장 잠재.
- **Sparse + Dense hybrid**: bge-m3는 sparse(BM25-like) 지원. 별도 실험 가치 있음.
- **다른 데이터셋 일반화**: 41 질문 단일 데이터셋 기반. MIRACL ko / KorQuAD-RAG 등으로 일반화는 후속 작업.
