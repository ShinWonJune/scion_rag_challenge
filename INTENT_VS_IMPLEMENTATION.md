# 의도(Intent) vs 구현(Implementation) 정합성 분석서

> 작성일: 2026-05-01
> 기반: `README.md`, `AGENTS.md`, `EXPERIMENT_BLUEPRINT.md`, `PROJECT_IMPROVEMENT_PLAN.md` ↔ 실제 import 그래프
> 목적: 리팩토링 전 dead code 제거 / 통합 / 의도 문서 보완

---

## 0. 분석 방법

진입점(`pipeline/run_pipeline.py`, 5개 step CLI, `src.preprocess_and_generate_answer`, `src.multi_hop_to_single_hop`, `src.build_vectordb_search`, `src.retrieval_system.main`)에서 시작해 import 그래프를 따라가 **활성 모듈(reachable)** 집합을 만들고, 그 외를 **orphan 후보**로 분류했다. 각 orphan은 해석을 붙여 (1) 삭제, (2) 통합, (3) 보존 + 의도 보완 중 하나로 라벨링한다.

---

## 1. 활성 의존성 체인 (현재 의도 그대로 동작)

```
run_pipeline.py
├─ step1 → src/search/{factories,scienceon_adapter,clients/*,cache,throttle,base_client}
│         + src/search_pipeline/core/{extractor_factory, base_extractor,
│           keyword_extractor, chatgpt_keyword_extractor, vllm_keyword_extractor,
│           codex_keyword_extractor*}
│         + src/utils/dedup
├─ step2 → src/multi_hop_to_single_hop → src/features/llm_question_decomposer
│         + src/llm_client/{init_gemini, call_gemini, prompts_tools}
│         + src/prompts/general/breakdown_multi_hop_question_as_single_hop
├─ step3 → src/build_vectordb_search → src/utils/{load_json, load_jsonl_..., create_class_from_schema,
│         save_as_csv_with_metadata} + src/features/embedding_processor + src/data_handler/for_embedding
├─ step4 → src/retrieval_system/{main,data_loader,query_encoder,result_saver,utils,retrievers/*}
└─ step5 → src/preprocess_and_generate_answer
          + src/llm_client/{init_gemini, call_gemini}
          + src/prompts/{general/generate_answer_base_v3*, scifact/generate_scifact_prompt}
          + src/{vllm_client, openai_client*, codex_client*}
          + src/llm_client/llm_factory  (← 여기서 만드는 객체는 step5가 사용 안 함)
```

`*` = git untracked (실제 디스크엔 있으나 추적되지 않음 → 5장 참조)

**평가**: 5단계 의도(question → search → decompose → embed/build → dense retrieve → answer)는 코드에 일대일 대응된다. **의도-구현 정합 점수: 높음.**

---

## 2. Dead Code (활성 체인에서 도달 불가)

### 2.1 Legacy 검색 체인 — **DELETE 후보**

`pipeline/step1_search.py`가 `ScienceONAdapter` 직접 호출로 단순화되면서 `SearchMetaSystem` 기반 구체계는 더 이상 호출되지 않는다.

| 파일/디렉토리 | 호출자 | 처리 |
|---|---|---|
| `src/search/services/search_meta_system.py` | `src/search_pipeline/legacy/main.py` 만 | DELETE |
| `src/search/services/__init__.py` | (위와 함께) | DELETE |
| `src/search_pipeline/legacy/` (디렉토리 전체) | 없음 | DELETE |
| `src/search_pipeline/processors/` (전체, 4 파일) | `legacy/main.py` 와 `search_meta_system.py` 만 | DELETE |
| `src/search_pipeline/utils/` (전체, 3 파일) | 위와 동일 | DELETE |
| `src/search_pipeline/settings.py` | `search_meta_system.py` 만 | DELETE |

**근거**: `grep`으로 외부 호출자 0건 확인. AGENTS.md "Invariant"의 `scienceon_api_example.py` 수정 금지 규칙은 영향 없음(adapter 통해 계속 사용).

### 2.2 Orphan 키워드 추출기 — **DELETE 후보**

`extractor_factory`는 4개 backend(`gemini`, `chatgpt`, `vllm`, `codex`)만 와이어링한다.

- `src/search_pipeline/core/llm_keyword_extractor copy.py` — **DELETE (파일명에 공백+"copy", 명백한 백업본)**

> **정정 (2026-05-01)**: 1차 분석에서 `llm_keyword_extractor.py` 와 `search_terms.py` 를 orphan 으로 잘못 분류했음. 정확한 module import 검증 결과:
> - `llm_keyword_extractor.py` 는 4개 활성 extractor(gemini/chatgpt/vllm/codex) 모두가 `from .llm_keyword_extractor import LLMKeywordExtractor` 로 사용 중 — **활성 유지**.
> - `search_terms.py` 는 `llm_keyword_extractor.py` 가 `from .search_terms import build_search_terms` 로 사용 — **활성 유지**.
> - 단, `llm_keyword_extractor copy.py` (파일명에 공백+"copy") 는 백업본이며 import 되지 않음 — DELETE 그대로 유지.

### 2.3 일회성 데이터 준비/리포트 스크립트 — **MOVE 또는 DELETE**

`src/` 루트의 스크립트 다수는 5단계 파이프라인 외부의 ad-hoc 작업(데이터셋 다운로드/변환, 결과 후처리)이다. 의도(README/AGENTS)에는 언급 없음.

| 파일 | 성격 | 처리 |
|---|---|---|
| `src/main.ipynb` (244 KB) | 탐색용 노트북 | `notebooks/` 로 이동 또는 DELETE (git history에 보존됨) |
| `src/final_result.py`, `final_result_pubmed.py`, `final_result_scifact.py` | 결과 합산 일회성 | `scripts/legacy/` 이동 |
| `src/extract_questions.py`, `extract_scifact_claims.py`, `extract_scifact_claims_distributed.py`, `extract_subcorpus.py` | 데이터셋 추출 일회성 | `scripts/data_prep/` 이동 |
| `src/run_extract_and_convert.py` | `/workspace/...` 절대경로 하드코딩, 실행 불가 추정 | DELETE |
| `src/convert_original_scifact.py`, `download_scifact_abstracts.py` | SciFact 일회성 | `scripts/data_prep/` 이동 (또는 DELETE — 현재 RAG 타깃은 ScienceON) |
| `src/rerank_inplace_by_config.py`, `src/rerank/retrieve_singlehop_contexts.py` | 활성 파이프라인 미사용 | `scripts/legacy/` 이동 |
| `src/inspector/inspec_json_in_csv_column.py` | 디버깅 도구 | `scripts/debug/` 이동 또는 DELETE |
| `src/llm_unified_client.py` | **0 byte 빈 파일** | DELETE |
| `src/outputs/elapsed_times.json` | 코드 디렉토리에 산출물 누수 | DELETE |
| `src/preprocess_and_generate_answer.py` | step5에서 활성 사용 | **유지 (활성)** |
| `src/build_vectordb_search.py` | step3 + embed_benchmark에서 활성 | **유지 (활성)** |
| `src/multi_hop_to_single_hop.py` | step2에서 활성 | **유지 (활성)** |

### 2.4 Legacy 평가 자료 — **DELETE 후보**

- `pipeline/evaluate/legacy/miracl_wiki_compare/` 의 4개 CSV: MIRACL 벤치마크 비교용 raw 자료. 현재 평가는 `eval_answers.py` / `eval_search.py` 가 담당. 의도(README) 에 MIRACL 비교 언급 없음 → DELETE.

### 2.5 깨진 테스트 — **FIX**

- `src/tests/test_history.py` 가 import 하는 `src.llm_client.history`, `src.llm_client.types` 는 **존재하지 않음**. 테스트 자체가 import 단계에서 실패 → 모듈 추가 또는 테스트 삭제. (history 기능을 의도한 흔적은 다른 곳에 없음 → **DELETE**)

### 2.6 통합 가능 thin wrapper — **MERGE**

- `pipeline/evaluate/eval_answers.py` (6 줄) → `src.evaluate.eval_answers.evaluate_answers` 호출만
- `pipeline/evaluate/eval_search.py` (6 줄) → `src.evaluate.eval_search.evaluate_search` 호출만

→ **방안 A (추천)**: 평가 진입점을 `pipeline/evaluate/`만 남기고, 실제 로직을 그대로 옮긴 뒤 `src/evaluate/` 삭제 (단방향 → step CLI 일관성).
→ **방안 B**: 반대로 `pipeline/evaluate/` 래퍼 삭제, `python -m src.evaluate.eval_answers` 로 직접 호출 안내. README 수정 필요.

---

## 3. 의도에는 있으나 구현되지 않은 부분

| 항목 | 의도 위치 | 상태 |
|---|---|---|
| HyDE / query transform | BLUEPRINT §1: 구 WP13 **제거 명시** | working tree 에선 이미 `D` (삭제됨), 미스테이지 — **commit으로 정리** |
| `exp3_query_ablation/` | BLUEPRINT 디렉토리 트리에는 없음, 그러나 git 추적엔 남음 | working tree `D` — **commit으로 정리** |
| 모든 WP0~WP16 본 작업 | BLUEPRINT WP 표 | 코드에 모두 존재 ✅ |

→ **즉시 조치**: `git rm` 으로 working tree 삭제분 staging + commit (별 PR).

---

## 4. 구현되어 있으나 의도 문서에 빠진 부분 — **의도 보완 필요**

다음은 git untracked 또는 의도 문서 미언급. 실제 동작에 필요/사용되므로 **추적 등록 + 의도 문서 보완**.

### 4.1 활성 코드 누락 (긴급)

| 파일 | 호출자 | 상태 | 조치 |
|---|---|---|---|
| `src/prompts/general/generate_answer_base_v3.py` | `src/preprocess_and_generate_answer.py:15` | UNTRACKED | `git add` (현재 step5의 실제 프롬프트) |
| `src/openai_client.py` | step5 chain | UNTRACKED | `git add` |
| `src/codex_client.py` | step5, target_sweep, closed_book | UNTRACKED | `git add` |
| `src/search_pipeline/core/codex_keyword_extractor.py` | `extractor_factory` (backend "codex") | UNTRACKED | `git add` |

> 위 4개 누락은 **현재 추적된 상태로 git checkout 시 step5/extractor 가 ImportError**. 즉시 stage.

### 4.2 부분 구현되었으나 의도가 모호한 부분

- **`src/llm_client/llm_factory.py`**: `create_llm_client` 가 반환하는 객체는 모두 `NotImplementedError`만 던지는 stub. 실제 LLM 호출은 `src/preprocess_and_generate_answer.py` 가 `VLLMClient`/`OpenAIClient`/`CodexClient` 를 **직접** import 한다.
  - **모순**: factory 패턴이 의도이나 실제 코드는 우회.
  - **두 갈래 결정 필요**:
    - (a) factory를 실 구현으로 채워서 step5가 factory 통하도록 리팩토링 (의도 정합 ↑)
    - (b) llm_factory.py 삭제, 직접 import 패턴 인정 (현실 정합 ↑)
  - 추천: (b) 후 BLUEPRINT 에 "LLM 호출은 step CLI 직접 import" 명시. 이유: factory가 어차피 backend별 분기만 하고 추가 가치 없음.

### 4.3 의도 문서에 없는 신규 실험 디렉토리 — **BLUEPRINT 보완 필요**

| 디렉토리 | 성격 | 의도 매핑 |
|---|---|---|
| `experiments/exp0_concurrency_sweep/` | PROJECT_IMPROVEMENT_PLAN §1.3 Step 2 (max_concurrency sweep) | BLUEPRINT WP4 와 동일 → **WP4 항목에 "구현 위치: experiments/exp0_concurrency_sweep" 추가** |
| `experiments/closed_book/` (untracked) | RAG 가치 측정의 closed-book 베이스라인 | BLUEPRINT 에 **신규 WP "Closed-book baseline" 추가** |
| `experiments/eval_gold_judge/` (untracked, artifacts 포함) | gold answer LLM judge | BLUEPRINT WP14 의 구체화 → **WP14 본문에 명시** |
| `experiments/gold_scienceon/` (untracked) | gold 문서셋 빌드/검증 | BLUEPRINT 에 **신규 WP "Gold dataset construction" 추가** | 
| `experiments/target_sweep/` (untracked) | target_documents 파라미터 sweep | BLUEPRINT 에 **신규 WP "target_documents sweep" 추가** |

→ **조치**: 이 5개 실험 디렉토리를 git add 한 뒤, BLUEPRINT 의 WP 표에 5개 항(또는 WP4 보강 + 신규 WP 4개)을 추가.

### 4.4 의도와 미세 불일치

- **`src/prompts/general/generate_answer_base{,_v2}.py`**: 활성은 `_v3`. v1·v2 는 미사용. → DELETE.
- **`README.md`**: `--llm` choices 에 `gemini, chatgpt, vllm` 만 표기되었으나 step5 실제는 `chatgpt` → OpenAI 경로. 그리고 codex backend는 keyword extractor 에만 노출. step5에는 codex 불필요.

---

## 5. 정리 후 예상 효과 (정량)

| 지표 | 현재 | 정리 후 |
|---|---|---|
| `src/` 추적 파일 수 | 약 70 | 약 40 (–30) |
| `src/search_pipeline/` 모듈 | 17 | 6 (legacy/processors/utils 제거) |
| 미사용 파일 | 25+ | 0 |
| Untracked 활성 파일 | 4 | 0 |
| 깨진 테스트 | 1 (`test_history.py`) | 0 |

---

## 6. 실행 계획 (위험도순)

### Step A — 즉시 안전 작업 (auto-execute 가능)

1. **누락 활성 파일 stage** (4.1): `git add` 4개 파일 + commit `chore: track active step5/extractor sources`
2. **이미 working tree 에서 삭제된 파일 stage** (3): HyDE, exp3_query_ablation 등 → commit `chore: prune deprecated query transform sources`
3. **분석 문서 stage**: 본 문서 + `REPO_CLEANUP_PLAN.md` → commit `docs: add repo cleanup analysis`

### Step B — 사용자 확인 후 (위험도 중)

4. **Dead code 삭제** (2.1, 2.2, 2.4, 2.5): legacy 검색 체인, orphan 추출기, MIRACL CSV, 깨진 테스트
5. **`src/` 루트 ad-hoc 스크립트 이동** (2.3): `notebooks/`, `scripts/{legacy,data_prep,debug}/` 신설
6. **thin wrapper 통합** (2.6): 평가 모듈 한 군데로
7. **llm_factory 결정** (4.2): (b) 채택 시 factory + 4개 stub class 삭제

### Step C — 의도 문서 보완 (낮은 위험)

8. **EXPERIMENT_BLUEPRINT.md 업데이트** (4.3): 5개 실험 디렉토리 매핑 추가
9. **README.md 업데이트** (4.4): `--llm` 옵션, codex extractor, 신규 실험 위치
10. **AGENTS.md 업데이트**: 정리 후 디렉토리 트리 반영

---

## 7. 결론

- 의도(5단계 RAG 파이프라인) **자체 구조는 코드와 잘 정합**한다.
- 그러나 (a) 구버전 검색 체인이 dead 상태로 남아있고, (b) 활성 코드 4개가 git untracked 며, (c) 5개 실험 디렉토리는 BLUEPRINT 에 미반영이다.
- Step A는 위험 없이 즉시 실행 가능하며, Step B/C 는 사용자 확인 1회 후 일괄 가능하다.
- 본 정리가 끝난 후에야 `REPO_CLEANUP_PLAN.md` 의 Phase 1 (코드 재구조화) 가 안전하게 수행된다.
