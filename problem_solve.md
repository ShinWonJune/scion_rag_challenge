# SHRAG 실험 진행 중 마주친 문제와 해결

> 작성: 2026-05-06
> 범위: EXPERIMENT_DESIGN.md 기반 retrieval 실험 (Exp1 ~ Exp7) 실행 중 발견한 시스템·설계·인프라 문제와 해결 기록

---

## P1. ScienceON BG 잡 silent 사망

### 증상
워커 1이 `python -m shrag.pipeline.steps.step1_search ... --emit-frozen-queries ... &`로 백그라운드 실행 후 idle 전환. 출력 파일은 0 bytes. 프로세스 ID 추적 시 부모 셸 종료와 함께 자식이 죽어 있음.

### 원인
`Bash` 도구의 `run_in_background=true` 위에 다시 `&`을 wrap하여 이중 백그라운드. 외부 wrapper bash가 `echo PID=$!` 후 즉시 exit → 프로세스 그룹 신호로 자식 python도 SIGHUP/SIGPIPE 받음.

### 해결
- BG 실행은 `run_in_background=true`만 사용하고 명령 안에 `&` 사용하지 않음.
- python 프로세스 PID는 ps로 직접 확인. `nohup`도 옵션이지만 `run_in_background`만으로 충분.

### 재발 방지
새 BG 명령 launch 직후 `ps -p <pid> -o etime,stat`으로 1회 sanity check. 출력 파일 사이즈가 일정 시간 내 변하지 않으면 디버깅.

---

## P2. step1_search default extractor=Gemini, GOOGLE_API_KEY 누락

### 증상
첫 cold cell 실행 시 `ValueError: API 키가 필요합니다`로 step1 실패. 다른 frozen-queries 사용임에도 keyword extractor가 생성되어야 함.

### 원인
- `step1_search.py:268` — `--frozen-queries` 사용 여부와 무관하게 `create_keyword_extractor()`가 항상 호출됨 (코드는 frozen 사용 시 extractor 출력 무시하지만 객체 생성은 시도).
- 기본 `--extractor=gemini`는 `GOOGLE_API_KEY` 필요. shell이 `.env`를 source하지 않으면 미설정.

### 해결
- run_cell.sh에 `source scripts/run_experiment_setup.sh`로 `.env` + conda 활성화.
- `--extractor vllm` + `--vllm-url` + `--vllm-model` 명시 (API key 불필요한 path).

### 재발 방지
환경 의존성 검증을 cell 직전 1회 수행: `--extractor vllm` + `--frozen-queries` 조합으로 smoke 1Q 실행. 통과하면 풀 sweep.

---

## P3. step1/step3 출력 timestamp subdir로 인한 follow-up 실패

### 증상
Cold cell 첫 실행: step1 success → search_documents.jsonl이 `search/<TS>/search_documents.jsonl`에 생성. step3가 `search/search_documents.jsonl` (subdir 없는 경로)을 찾다 FileNotFoundError. step3 출력도 `OUT_DIR/<TS>/vector_db_*.csv`로 timestamp subdir 생성 → step4가 vectordb를 못 찾음.

### 원인
step1과 step3 모두 내부에서 `datetime.now().strftime("%y%m%d_%H%M%S")` 서브디렉토리를 자동 생성. 호출자가 지정한 output 경로의 직접 파일이 아니라 그 아래 timestamp 폴더에 저장.

### 해결
run_cell.sh에서 step1, step3 출력 후 `find ... -name "search_documents.jsonl" -size +0 | head -1` / `find ... -name "vector_db_*.csv"`로 동적 위치 탐색.

### 재발 방지
- 새 step 호출 시 정확한 출력 경로 패턴 확인.
- Idempotency: cell이 timing.json을 만들면 다음 실행 시 스킵하도록 driver 설계.

---

## P4. set -euo pipefail + find 실패가 silent exit

### 증상
Phase A 25-cell driver 실행 시 24 cells 모두 7초 안에 "FAIL"로 종료. cell directory 자체가 안 만들어짐 (mkdir 도달 못함).

### 원인
run_cell.sh 첫 줄 `set -euo pipefail`. cell 디렉토리 새로 생기는 첫 실행 시 `find $OUT_DIR/search 2>/dev/null`이 디렉토리 없음으로 실패 → `set -o pipefail` + `head -1` 파이프 → 전체 exit code 1 → `set -e`로 즉시 종료. 그래서 echo 한 줄만 찍히고 종료.

### 해결
- `set -euo pipefail` → `set -uo pipefail`로 완화 (errexit 해제).
- `mkdir -p "$OUT_DIR" "$OUT_DIR/search" "$OUT_DIR/retrieval"`로 사전 생성.
- 명시적 ERR trap으로 에러 라인 표시: `trap 'echo "[exp1_kmn] FATAL at line $LINENO (rc=$?)"' ERR`.

### 재발 방지
드라이버 스크립트는 첫 cell만 foreground로 verbose 실행해서 sanity 확인 후 25-cell 본격 실행.

---

## P5. step4 retrieval 출력이 per-question JSON × 41 (단일 jsonl 아님)

### 증상
retrieval_eval CLI가 `retrieval.jsonl`을 기대하는데 step4는 `{qid}_<encoder>_<hash>.json` 41개 파일을 timestamp subdir에 출력.

### 원인
- step4는 `shrag.retrieval.main`을 subprocess 호출. 내부 출력 포맷이 per-question JSON dict (keys: id, retrieval_results[], meta, ...).
- `embed_benchmark.py`는 자체 retrieval loop를 돌리고 단일 retrieval.jsonl을 만들지만, 우리는 `shrag.retrieval`을 직접 사용해서 포맷이 다름.

### 해결
`experiments/exp1_kmn/aggregate_retrieval.py` 추가 — 41개 per-question JSON을 단일 retrieval.jsonl로 합침. `retrieval_results[type=="original"].hits` 추출, `{question_id, hits:[{doc_id, rank, score}]}` 포맷으로 정규화.

### 재발 방지
새 평가 도구 통합 시 입출력 포맷 spec을 코드 한 줄로 검증 (예: 첫 record key set 확인).

---

## P6. Frozen-queries slicer 버그 — search_terms만 슬라이스, search_terms_by_lang는 풀 리스트

### 증상 (가장 큰 문제)
Phase A에서 (k=50, n=1), (k=50, n=8), (k=50, n=10) 3 cells 모두 Hit@5 = 0.854로 정확히 동일. (k=10, n=1) vs (k=10, n=10)도 동일. n axis 효과 0으로 보였음.

진단 시 `search_meta_results.json`의 query별 docs 카운트 확인: n=1 cell도 query당 평균 73 docs (m=70 cap에 hit), 16개 search_terms 모두 가용한 상태로 동작.

### 원인
초기 slicer (`slice_frozen_queries.py` v1)가 frozen-queries JSONL의 `search_terms` 필드만 슬라이스 (`out["search_terms"] = ko + en`). 그러나 `step1_search.py:173`은:
```python
terms_by_lang = frozen.get("search_terms_by_lang") or _build_search_terms_by_lang(keywords)
```
즉 **`search_terms_by_lang` 우선**. 슬라이서는 이 필드를 풀 리스트 그대로 두었음. 결과적으로 n=1 의도한 모든 cell이 ko 7 + en 9 = 16 terms를 사용. m=70 cap에 의해 binding되어 동일 corpus 도출.

이로써 5 cells × ~30분 = 2.5시간 GPU 시간이 잘못된 가설("n axis 무관") 측정에 낭비됨.

### 해결
- `slice_frozen_queries.py` v2: `search_terms_by_lang.korean`, `search_terms_by_lang.english` 둘 다 top-N으로 슬라이스. 동시에 `search_terms = ko_sliced + en_sliced` 재계산. `search_terms_meta` 필드 추가.
- 단위 테스트 추가: 슬라이스 결과를 다시 읽어 두 필드 모두 N개인지 assert.

### 재발 방지 (메모리에 저장됨)
**새 로직 구현 시 입출력 단위 테스트 말고 downstream consumer까지 추적한 effect 검증 필수.** 비싼 실험 launch 전 반드시 1Q smoke run → meta 출력 sanity check (예: query당 docs 카운트가 의도한 범위인지) 후 풀 sweep.

`/home/wonjune/.claude/projects/.../memory/feedback_verify_implementation.md`에 영구 저장.

---

## P7. step3 인코딩 GPU 병목 (cell당 28분)

### 증상
4938 docs 인코딩에 cold cell 168s, 후속 동일 corpus cell 1698s (10배 느려짐). RTX 4070 12GB GPU util 100%, memory 11876/12282 MiB.

### 원인
- WSL2 + Windows GPU 공유 환경에서 sentence-transformers SentenceTransformer가 12GB VRAM에 가까이 압박.
- 첫 batch 421s (model load + warmup), 이후 9-15s/batch가 정상 (155 batches → 30분).
- Cold 168s 측정은 CSV 작성까지의 짧은 측정 — 전체 4938 docs 인코딩이 그 시간에 끝났다고 보기 어려움. 실제로는 두 번째 cell의 30분이 정상 GPU 처리 시간.

### 해결
**Embedding cache** (P-cache) 도입. 같은 doc은 (encoder, mode) 조합당 한 번만 인코딩하고 SQLite에 저장 → 후속 cell은 즉시 재사용.

상세는 EXPERIMENT_DESIGN.md §13a + `shrag/utils/embedding_cache.py`.

검증된 효과:
- 100% cache hit: 모델 로드 자체 스킵, 10 docs cell 9.1s → 3.2s.
- Vectorequivalence: cached vs fresh encoding max abs diff = 0.0.

### 추가 대응
- Cell 실행 순서: **큰 cell 먼저** (k=50 → k=10, m=70 → m≤70) → 후속은 100% hit.
- Strategy sweep (Exp4): 같은 corpus 8개 mode → 첫 mode만 cold, 나머지는 mode key 분리되어 새로 인코딩하지만 doc text 변환만 다르므로 빠름.
- Encoder sweep (Exp3): encoder 전환 시 캐시 분리, 어차피 새로 인코딩 필요. 인코더당 1회만 cold.

---

## P8. 풀 sweep scope 비현실성

### 증상
EXPERIMENT_DESIGN.md 원래 plan: Exp1 25 + Exp2 6 + Exp3 3 + Exp4 16 + Exp7 6 = 56 cells. 평균 cell당 5-30분 → 5-15시간. 실측 cell당 28분 (GPU 병목) → 26시간 비현실.

### 해결
**Reduced scope**:
- Phase A: 25 → 5 corner cells (이후 슬라이서 버그로 사실상 2 distinct cells)
- Phase C: skip
- Exp4: 16 → 8 cells (gte only, runner-up encoder skip)
- Exp7: 6 → 3 cells

Cache 도입 후 효과:
- Exp4 8 cells: 24분 → 4분 (100% mode가 다르더라도 corpus 동일하니 부분 절감 가능). 단 mode가 cache key의 일부라 fresh encode. 그러나 corpus 작아서(k10_n1_m30) ~수분 / cell.
- Phase B m sweep: 15분 → 3분.

### 재발 방지
- 풀 sweep 전 1 cell 실측 후 환산.
- 1Q timing harness로 빠른 cost prediction.

---

## P9. Worker 워커 idle 후 BG 잡 죽음

### 증상
worker-1 (Claude agent)가 frozen-queries BG 잡 launch 후 SendMessage로 보고하고 idle. 그 후 잡이 죽어 있음.

### 원인
Claude agent 워커는 idle 시 프로세스가 종료될 수 있음. 자식 BG 프로세스가 워커 프로세스 그룹에 묶여 있으면 함께 종료.

### 해결
- 장시간 BG 잡은 lead가 직접 BG 실행 + Monitor로 추적.
- 워커는 prep 작업 / 분석 / 보고처럼 짧은 task만 담당.

### 재발 방지
- 30분+ 예상 BG 잡은 워커가 아니라 lead가 실행.
- 워커가 BG 띄울 경우 nohup + setsid로 부모 분리.

---

## 요약: 핵심 lesson

1. **검증 없는 구현은 금물** — 새 도구 만들 때 단위 입출력 말고 downstream consumer까지 추적해서 effect 확인 (P6).
2. **리소스 비싼 실험은 1 cell smoke 후 sweep** — 25 cells 후 발견하면 늦음 (P4, P6).
3. **bash strict mode 주의** — `set -e` + pipe + 부재 디렉토리 = silent exit (P4).
4. **timestamp subdir 패턴 인지** — step1/step3 모두 자동 생성, find로 동적 추적 (P3).
5. **GPU 병목은 캐시로 우회** — encoder/mode/text_sha1 4중 키로 안전 (P7).
6. **워커 BG 잡 신뢰 X** — 장시간 작업은 lead 직접 (P1, P9).

---

## 산출 코드

| 파일 | 역할 |
|---|---|
| `shrag/utils/embedding_cache.py` | SQLite 임베딩 캐시 클래스 |
| `shrag/features/embedding_processor.py:generate_batch_embeddings_cached` | 캐시 사용 wrapper |
| `shrag/pipeline/_impl/build_vectordb.py` | config의 cache_db_path 트리거 |
| `experiments/exp1_kmn/slice_frozen_queries.py` | (수정) frozen JSONL n-slicer |
| `experiments/exp1_kmn/aggregate_retrieval.py` | step4 per-question JSON → retrieval.jsonl |
| `experiments/exp1_kmn/timing_1q.py` | 1Q latency harness |
| `experiments/exp1_kmn/run_cell.sh` | (수정) idempotency + dynamic path lookup |
| `experiments/exp1_kmn/aggregate.py` | cell별 metrics 집계 |
| `experiments/exp1_kmn/make_plots.py` | heatmap/Pareto/curve |
