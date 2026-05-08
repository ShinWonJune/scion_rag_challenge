# SHRAG 최적 설정 요약

> 작성: 2026-05-07
> 데이터셋: ScienceON gold 41 questions, 학술 논문 abstract corpus
> 상세 분석: `experiments/outputs/FOLLOWUP_EXPERIMENT_SUMMARY.md`

---

## 1. 권장 pipeline 

```
질문 → 검색어 생성 (vLLM) → ScienceON 수집 (k=30, n=10, m=50)
     → GTE fp16 b32 len=512 임베딩 (3T+A)
     → dense retrieve top-5
     → dragonkue/bge-reranker-v2-m3-ko rerank → top-3 (or top-5)
     → vLLM 답변 생성
```

### Config 한 장

```json
{
  "search":   {"k": 30, "n": 10, "m": 50},
  "encoder":  "Alibaba-NLP/gte-multilingual-base",
  "encoder_runtime": {
    "torch_dtype":      "float16",
    "max_seq_length":   512,
    "batch_size":       32,
    "attn_implementation": "sdpa"
  },
  "embedding_mode":     "3*title+abstract",
  "dense_retrieve_top_k": 5,
  "reranker":           "dragonkue/bge-reranker-v2-m3-ko",
  "rerank_candidates":  5,
  "llm_context_top_k":  3
}
```

---

## 2. 핵심 metric (41Q gold)

| metric | dense top5 only | + bge-v2-m3 (c5) | + dragonkue-ko (c5) ⭐ |
|---|---:|---:|---:|
| **Hit@1** | 0.610 | 0.732 | **0.780** |
| **Hit@3** | 0.829 | **0.902** | **0.902** |
| **Hit@5** | **0.902** | **0.902** | **0.902** |
| Hit@10 | 0.902 | 0.902 | 0.902 |
| **MRR@10** | 0.723 | 0.813 | **0.833** |
| mean_gold_rank | 2.24 | 1.22 | **1.19** |

> Hit@5 ceiling = 0.902 (37/41). 나머지 4 query는 step1 단계에서 gold 미수집 (search recall 한계).

---

## 3. 속도 (3,557 docs corpus, RTX 4070 12GB)

| 단계 | cold | warm/cache | 비고 |
|---|---:|---:|---|
| step1 검색어 생성 | ~8.3s/q | 캐시 시 ~0s | vLLM gpt-oss-20b |
| step1 ScienceON 수집 | ~16.1s/q | cache hit ~0.1s | request cache |
| step3 embedding (전체 corpus) | **13s** | 0.01s | fp16 b32 len=512 |
| step4 dense retrieve | 0.001s/q | — | FAISS |
| reranker (c5 → top5) | **0.177s/q** (warm) | model load 6s 1회 | dragonkue/bge-reranker-v2-m3-ko (568M). Exp7c subprocess 격리 측정 |
| step5 vLLM 답변 생성 | ~3.7s/q | — | gpt-oss-20b |

**Hot-path e2e (warm cache)**: ~3.5s/query
**Cold e2e**: ~40s/query (acquisition dominant)

---

## 4. 도출 근거 요약

### 검색 변수 `(k, n, m) = (30, 10, 50)`
- 검색어 1개당 ScienceON 수집 docs `k`, 질문당 검색어 수 `n`, dedup cap `m`
- 출처: 초기 Exp1 + 후속 검증

### Encoder = GTE
- Hit@5 동일 (0.902) — bge-m3와 무차별
- Cold embedding 비용: gte ~99-129s vs bge-m3 ~642s (6.5×)
- bge-m3는 Hit@1 +0.097, MRR +0.057 우위 — 그러나 cost 6.5× → ROI 낮음
- Reranker (dragonkue-ko)로 Hit@1 +0.170, MRR +0.110 더 큰 이득 가능
- → **gte + reranker** 조합이 운영 효율 최적
- 출처: Exp3, Exp12, Exp13, Exp14, Exp16, Exp18, Exp2b
- 추가 후보 (Exp2b 9-model 확장 비교): Solon-embeddings-large (Hit@1=0.756, MRR=0.801)와 dragonkue/BGE-m3-ko (Hit@1=0.732, MRR=0.799)가 dense Hit@1/MRR 우위지만 Hit@5 -1~-2 query 손실 → search recall 보존 우선이라 gte 유지

### Encoder runtime: fp16 + b32 + len=512 + sdpa
- fp32 default(8192) 대비 **5.78× 빠름** (116s → 13s for 3,557 docs)
- Hit@5 변화 없음 (truncation: 90% docs ≤ 538 tokens, 10% 잘림이지만 retrieval 영향 없음)
- batch saturation: b32/b64/b96 모두 같은 13s — b32면 충분, VRAM 절약
- 출처: Exp15, Exp16, Exp18 (subprocess 격리 verification)

### Embedding mode = `3*title+abstract`
- title only Hit@5 = 0.659, abstract only = 0.780, **3T+A = 0.902**
- title 가중치가 검색 신호 dominant
- 5T+A는 Hit@5 동일하지만 Hit@1/MRR 하락
- 출처: Exp4, Exp4 sampled20

### Dense retrieve top-5
- top5/7/10/20/50 모두 Hit@5 = 0.902 (top5 이상은 단순 추가 비용)
- 출처: Exp8

### Reranker: dragonkue/bge-reranker-v2-m3-ko, candidates=5
- **품질 비교** (Exp7b, fair sweep, cand=5, dense top-5 후 rerank → top-5):
  - **dragonkue-ko**: Hit@1 0.780, Hit@3 0.902, MRR 0.833, mgr 1.19 ⭐
  - BAAI/bge-reranker-v2-m3: Hit@1 0.732, MRR 0.813, mgr 1.22
  - Alibaba-NLP/gte-multilingual-rerank: Hit@1 0.585 (dense baseline보다 낮음, 채택 X)
- **속도 정확 측정** (Exp7c, subprocess 격리 + warmup pass + N=2 측정):
  - dragonkue-ko: **0.177 s/query** (warm), VRAM 6759 MiB
  - BAAI/bge-reranker-v2-m3: 0.174 s/query (warm) — dragonkue와 차이 1.7% (noise)
  - Alibaba-NLP/gte-multilingual-rerank: 0.074 s/query — 빠르지만 품질 ↓
  - cold model load: 모든 모델 ~6-7s (1회만 발생, warm reuse)
- candidates 효과 (Exp8b, dragonkue 기준): cand=5와 cand=10 모두 Hit@1=0.780, MRR=0.833, mgr=1.19로 완전히 동일 → cand=5 best (cand=10은 19.8% 추가 비용에 이득 없음)
- candidates 5 → rerank → top3 (LLM): Hit@3 0.829 → 0.902, MRR 0.723 → 0.833
- BGE-v2-m3와 dragonkue-ko는 같은 base architecture (568M) → 운영 warm 비용 동등 (Exp7c로 검증)
- **Retriever × Reranker stack** (Exp7d): retriever를 dragonkue/BGE-m3-ko로 바꿔도 추가 이득 없음. cand=5에서 Hit@5 -1 query, cand=10에서 회복하지만 warm 속도 0.195 → 0.403 s/q (2×). gte retriever + dragonkue reranker cand=5가 모든 metric + 속도에서 winner (warm 0.195 s/q, cold 31.4s 포함 첫 query).
- 출처: Exp7, Exp7b, Exp7c, Exp7d, Exp8, Exp9

### LLM context top-3
- top5 → top3 prompt token -29.5%, vLLM 응답 시간 -3.2 ~ -6.2%
- Hit@3 0.902 → top3에 gold 포함 41Q 중 37개
- generation 시간이 dominant이라 token 절감의 직접 이득은 제한적
- 출처: Exp9

---

## 5. 대안 (목표별)

| 목표 | setting | 절충 |
|---|---|---|
| **최저 e2e latency** | gte fp16 b32 len=512 + dense top5 (no reranker) | Hit@1 0.610, MRR 0.723. 단순/빠름 |
| **Hit@5 + BGE-M3 quality** | bge-m3 fp16 b16 len=1024 + dense top5 | Hit@5 0.902 유지, MRR 0.769. embedding 31s (gte 13s 대비 2.4×) |
| **최고 top-1, MRR** | bge-m3 fp16 b16 len=512 + dense top5 | Hit@1 0.756, MRR 0.810. 단 Hit@5 0.878 (-1 query) |
| **검증된 권장 (균형)** | **본 권장: gte fp16 b32 len=512 + rerank c5 → top3** | best balance |

---

## 6. 한계

- 단일 dataset (Q41 ScienceON gold). 다른 도메인 일반화 미검증.
- N=1 cold run: 작은 차이(≤5%)는 noise.
- Hit@5 ceiling 0.902 = step1 search recall 한계 (4 queries는 어떤 setting에서도 gold 미수집).
- vLLM 답변 품질(LLM-as-judge)은 본 평가 scope 외.
