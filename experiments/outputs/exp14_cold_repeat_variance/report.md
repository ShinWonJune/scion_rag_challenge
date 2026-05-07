# Exp14. Cold GTE Repeat Variance

작성일: 2026-05-07

## 목적

Exp12와 Exp13이 같은 Q41 corpus 3,557문서를 같은 GTE 설정으로 임베딩했는데도 약 30초 차이가 난 이유를 확인하기 위해, 동일 조건을 두 번 더 독립 프로세스로 재실행했다.

주의: OMX worker-1은 별도 worktree에서 CPU-only runtime으로 실행되어 첫 batch 314초 ETA가 나와 중단했다. 아래 comparable rerun A/B는 leader가 GPU-enabled runtime에서 직접 수행한 결과다.

측정 범위:

- 포함: JSONL 로드, 모델 로드, CUDA/PyTorch 초기화, document embedding, vector DB CSV 저장, temp config write
- 제외: 검색어 생성, ScienceON 수집, query encoding, retrieval, rerank, vLLM generation
- cache: embedding cache off (`cache_db_path` 제거)
- corpus: `experiments/outputs/exp3_encoder/corpus.jsonl`
- docs: `3557`
- model: `Alibaba-NLP/gte-multilingual-base`
- embedding mode: `3T+A`

## 결과

| Run | Source | Docs | Build sec | Docs/sec | First batch observed |
|---|---|---:|---:|---:|---:|
| Exp12 | `experiments/outputs/exp12_q41_embedding_cold/summary.json` | 3557 | 98.6966 | 36.0397 | - |
| Exp13 | `experiments/outputs/exp13_exp3_gte_cold_rerun/summary.json` | 3557 | 128.9104 | 27.5928 | 43.38s |
| Exp14 rerun A | `experiments/outputs/exp14_cold_repeat_variance/rerun_a/summary.json` | 3557 | 149.9598 | 23.7197 | 42.37s |
| Exp14 rerun B | `experiments/outputs/exp14_cold_repeat_variance/rerun_b/summary.json` | 3557 | 175.0776 | 20.3167 | 56.14s |
| Exp14 rerun C | `experiments/outputs/exp14_cold_repeat_variance/rerun_c/summary.json` | 3557 | 129.3573 | 27.4975 | 42.47s |

Exp14 세 번의 평균은 `151.4649s`이고, Exp14 repeat range는 `45.7203s`다. Exp12/Exp13/Exp14 다섯 관측값 전체 범위는 `98.6966s`부터 `175.0776s`까지이며, range는 `76.3810s`다.

## 해석

Exp12와 Exp13의 약 30초 차이는 문서나 설정 차이로 설명되지 않는다.

확인된 동일 조건:

- corpus 동일: `experiments/outputs/exp3_encoder/corpus.jsonl`
- 문서 수 동일: `3557`
- temp encoder config 차이는 output path와 run timestamp뿐
- vector DB CSV 크기/라인 수 동일 수준
- embedding cache off

이번 Exp14에서 같은 GPU 조건을 다시 실행했을 때 `149.9598s`, `175.0776s`, `129.3573s`가 나왔다. 특히 `rerun_c=129.3573s`는 Exp13의 `128.9104s`와 거의 같으므로, team 실행 자체가 계속 속도를 떨어뜨렸다는 증거는 약하다. Exp12의 `98.6966s`와 Exp13의 `128.9104s` 차이는 현재 benchmark 구조상 충분히 발생 가능한 runtime variance로 봐야 한다.

핵심 원인은 현재 측정값이 순수 GPU embedding kernel 시간이 아니기 때문이다. `embedding_cold_benchmark.py`의 `build_sec`는 다음을 한 번에 잰다.

```text
JSONL 로드
SentenceTransformer 모델 로드
CUDA/PyTorch 초기화
첫 batch warmup
전체 batch encode
numpy -> Python list 변환
68MB CSV 저장
temp config write
```

특히 first batch가 Exp13/Exp14에서 `42-56s` 수준으로 관찰됐다. 첫 batch 이후에는 batch 속도가 빠르게 안정화된다. 즉 전체 차이의 상당 부분은 CUDA/model warmup, GPU clock/power state, WSL `/mnt/c` I/O, Python object conversion/CSV write 변동이 섞인 결과다.

## 결론

- Exp3 원본 `gte step3=3.1252s`는 cold embedding/vector DB build 값으로 볼 수 없다.
- 같은 Q41/GTE/cache-off 조건의 실제 cold-ish build time은 현재 관측상 `98.7-175.1s` 범위다. team을 쓰지 않은 leader GPU rerun C가 `129.3573s`로 Exp13과 거의 같았으므로, team worker 때문에 지속적으로 느려진 것으로 보기는 어렵다.
- 단일 cold run 하나로 대표값을 잡으면 안 된다. 최소 3회 이상 반복하고, 가능하면 stage를 `model_load`, `encode`, `prepare_documents`, `csv_save`로 분리해야 한다.
- 운영 판단에는 평균/범위를 함께 쓰는 것이 맞다. 현재 다섯 관측값 평균은 `136.4003s`다.

## 속도 개선 방향

상세 권장 설정은 `experiments/outputs/exp14_cold_repeat_variance/embedding_speed_optimization.md`에 정리했다.

우선순위:

1. `batch_size`를 명시하고 sweep한다. GTE는 `64 -> 96 -> 128` 순서로 GPU memory를 보며 올린다.
2. CUDA에서는 `torch_dtype=float16`을 먼저 시도한다. BF16도 가능하지만 retrieval metric을 확인해야 한다.
3. `max_seq_length`를 명시한다. GTE는 `512`부터, BGE-M3는 dense-only 기준 `512/1024`부터 확인한다.
4. BGE-M3를 쓸 때는 `FlagEmbedding`의 `use_fp16=True`, dense-only, 작은 `max_length`를 우선한다.
5. 길이 기반 bucketing으로 batch 내 padding 낭비를 줄인다.
6. PyTorch 튜닝 이후에 ONNX/OpenVINO를 별도 후보로 비교한다.
