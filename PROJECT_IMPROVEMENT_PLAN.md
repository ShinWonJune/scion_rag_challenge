# SHRAG 보완 계획서

작성일: 2026-04-22

## 0. 현재 검증 상태 요약

`data/test.csv`를 입력으로 `pipeline.run_pipeline` 단일 명령을 실행하여 ScienceON 검색, 문서 수집, embedding, dense retrieval, vLLM 답변 생성을 처음부터 끝까지 검증했다.

실행 결과:

- 입력 질문: 50개
- ScienceON 수집 문서: 1070개
- retrieval 결과: 50개
- final answer 결과: 50개
- 멀티홉 분해: 생략
- 실행 메모: `outputs/e2e_vllm20b_testcsv_scienceon_fullrun/RUN_NOTES.md`

다만 ScienceON 검색 중 HTTP 429가 다수 발생했다. 현재 파이프라인은 검색 API 실패를 로그로 남기고 넘어가므로 전체 파이프라인은 완주하지만, 질문별 검색 품질은 불균등해진다.

---

## 1. 429로 직접 검색 0건인 질문 대응책

### 1.1 문제 정의

full run의 `search_meta_results.json` 기준:

- 전체 질문: 50개
- 질문별 직접 ScienceON 검색 문서 수 평균: 22.5개
- 직접 검색 문서 0건 질문: 5개
- 목표 문서 수 50개를 채운 질문: 0개

현재 구조에서는 Step1이 질문별 문서를 수집한 뒤 전체 문서를 global VectorDB로 합친다. 따라서 어떤 질문의 직접 검색 문서가 0건이어도, Step4는 다른 질문에서 수집된 문서까지 포함한 global VectorDB에서 검색한다. 이 방식은 파이프라인 완주에는 유리하지만, 질문별 provenance가 약해진다.

### 1.2 429가 유발할 수 있는 문제

1. 질문별 recall 저하

   429로 특정 query/search term/page 호출이 실패하면 그 질문에 대해 원래 들어왔어야 할 후보 문서가 VectorDB에 들어오지 않는다.

2. run-to-run 재현성 저하

   같은 입력이라도 API rate limit 시점에 따라 수집 문서 풀이 달라진다. 결과적으로 embedding 대상, retrieval ranking, 최종 답변이 달라진다.

3. 잘못된 성공처럼 보이는 결과

   직접 검색 0건 질문도 global VectorDB에서 문서를 찾기 때문에 답변 파일은 생성된다. 하지만 그 문서가 해당 질문의 ScienceON 검색으로 수집된 문서인지 보장되지 않는다.

4. 평가 왜곡

   현재 검색 평가는 “수집 문서 중 정답 문서 포함 여부”를 본다. 429가 많이 나면 모델 성능이 아니라 API throttling이 QSR을 낮출 수 있다.

5. 후속 디버깅 어려움

   현재 metadata에 429 횟수, 실패 search term, retry 여부가 구조화되어 있지 않으면 실패 원인을 후처리하기 어렵다.

### 1.3 설계안: 요청률 제어 + 재시도 (+ 캐시)

**기본 인식**

HTTP 429 는 서버가 "지금 요청이 너무 많다"고 알려주는 응답이다. 근본 해결은 클라이언트 요청률을 서버 한도 아래로 낮추는 것(throttle)이고, 재시도/백오프는 그 아래서 남는 burst 를 흡수하는 2차 방어선이다. 요청 캐시는 중복 호출 자체를 제거해 1·2차의 부담을 줄이는 3차 축이다. 순서대로 1차부터 적용해 해결되는 지점을 찾는다.

**1차 — 요청률 제어 (스로틀)**

- `ScienceONAdapter` / wrapper 계층에 단일 프로세스 기준 `max_concurrency`(동시 요청 수 상한)와 `min_interval_sec`(연속 호출 사이 최소 간격)을 둔다.
- 기본값은 보수적으로 설정(예: `max_concurrency=2`, `min_interval_sec=0.5`).
- 429 관측 시 AIMD(Additive Increase, Multiplicative Decrease)로 `min_interval_sec` 을 adaptive 하게 증가. 연속 성공 N회 후 점진 회복.
- Throttle 단독으로 대부분의 429 가 해소되는지 먼저 확인한다.
- 이후 최대 max_concurrency를 도출하여 최대 효율 유지

**2차 — 재시도 + 백오프**

- 1차를 뚫고 올라온 잔여 429 만 처리한다.
- HTTP 429 감지 시 exponential backoff (`retry_base_sleep_sec × 2^n`), `max_retries` 상한.
- `Retry-After` 헤더가 있으면 우선 사용.
- 재시도 실패 시 fail-soft: 해당 term 은 로그 + metadata 기록 후 다음으로 넘어간다.

**3차 — 요청 캐시**

- 동일 `(source, term, page, row_count, fields)` 요청은 디스크 캐시에서 응답. 실험을 반복할 때 재요청이 아예 없도록 한다.
- 저장 위치: `outputs/_shared_cache/`. run 간 공유.

**설정 예**

```yaml
scienceon:
  # 1차: throttle
  max_concurrency: 2
  min_interval_sec: 0.5
  throttle_aimd_increase: 0.25          # 429 시 interval 증가폭
  throttle_aimd_decrease_after: 20      # 연속 성공 N회 후 interval 감소
  interval_cap_sec: 10.0
  # 2차: retry
  max_retries: 5
  retry_base_sleep_sec: 2
  retry_max_sleep_sec: 60
  # 3차: cache
  cache_enabled: true
  cache_root: outputs/_shared_cache
  # 기타
  fail_on_zero_docs: false
```

**실험 방법**

두 단계로 진행한다. Step 1에서 방어선 구성을 확정하고, Step 2에서 운영 파라미터를 튜닝한다.

**Step 1 — 방어선별 누적 비교**

동일 50개 질문을 3회 반복 실행하며 1차부터 쌓아가는 방식으로 비교한다.

- A. 현재 방식 (no throttle, no retry, no cache)
- B. throttle only (`min_interval_sec=0.5`, `max_concurrency=2`)
- C. throttle + exponential retry
- D. throttle + retry + cache

측정:

- 총 429 수 (1차 목표: B 에서 0 또는 극소)
- 질문별 document_count 평균/분산
- zero-doc 질문 수
- 전체 실행 시간
- QSR 변화

**Step 2 — max_concurrency sweep (throughput ceiling 탐색)**

Step 1에서 채택된 조합(보통 D = throttle+retry+cache) 위에서 `max_concurrency` 만 변화시키며 서버 한도와 처리량의 tradeoff 를 측정한다. 목표는 *429 rate 를 허용치 이하로 유지하면서 throughput을 최대화하는 운영 지점* 을 찾는 것이다.

- 조건: `min_interval_sec`, retry, cache 는 Step 1 채택값으로 고정. 캐시는 초기화(cold start)하여 실제 요청이 발생하도록 한다.
- sweep: `max_concurrency ∈ {1, 2, 3, 5, 8}` (초기값; 상단에서 429 rate 가 폭증하면 더 늘리지 않는다).
- 각 값에서 동일 질문 세트를 1회 실행(AIMD를 끄고 고정 concurrency 로 측정).
- 측정:
  - 429 rate = (429 응답 수) / (총 요청 수)
  - 평균 throughput (요청/초, 캐시 히트 제외)
  - 전체 실행 시간
  - zero-doc 질문 수
  - Retry-After 헤더가 권고한 누적 대기 시간
- 선정 규칙: *최대 throughput 을 내면서 429 rate ≤ ε* (예: ε = 1%, 또는 절대값으로 429 수 ≤ N). 이 값을 운영 default 로 config 에 반영하고, AIMD 의 상한도 이 값으로 clamp.
- 보고: `max_concurrency` 축으로 throughput·429 rate 를 겹쳐 그린 그래프를 산출.

**판정 기준**

- Step 1: B(throttle only) 에서 429 가 0에 수렴하면 1차만으로 충분 — 2·3차는 재현성·비용 절감 용도로 유지. B 에서 잔여 429 가 있으면 C(+retry)로 흡수. 여전히 남으면 `min_interval_sec` 을 상향.
- Step 2: 429 rate 가 `max_concurrency=1` 에서도 유의하게 나오면 `min_interval_sec` 자체가 부족한 것이므로 Step 1로 회귀해 interval 을 키운다.
- 잔여 zero-doc 질문이 throttle+retry 이후에도 남으면, 그건 429 문제가 아니라 keyword/query 문제이므로 별도 분석 (본 절의 scope 밖).
- 실행 시간이 늘더라도 QSR·재현성이 개선되면 채택.


## 2. 보완사항 1: Embedding 모델 성능 비교 설계

### 2.1 요구 조건

이 프로젝트의 embedding 모델은 다음 조건을 만족해야 한다.

- 한국어-영어 cross-lingual 검색 성능
- 한국어/영어 혼합 학술 질의 처리
- 논문 제목 + 초록 수준의 긴 문서 처리
- 실험 반복 가능성
- CPU/GPU 자원 대비 처리 속도
- FAISS 또는 NumPy retrieval과 호환 가능한 dense vector 출력

현재 모델은 `Alibaba-NLP/gte-multilingual-base`이며, Hugging Face model card 기준 70개 이상 언어와 8192 token 입력을 지원하고, embedding dimension은 768이다. [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)

### 2.2 후보 모델군

#### 2.2.1 현재 baseline

1. `Alibaba-NLP/gte-multilingual-base`

   장점:
   - 8192 token 지원
   - 70개 이상 언어
   - 305M 수준으로 비교적 가벼움
   - 현재 파이프라인과 이미 통합됨

   리스크:
   - 실제 실행에서 1024 설정과 실제 embedding 768 차원 불일치가 보임
   - `trust_remote_code=True` 계열 사용성 검토 필요

#### 2.2.2 Long-context multilingual 후보

2. `BAAI/bge-m3`

   근거:
   - BGE 문서 기준 dense retrieval, sparse retrieval, multi-vector retrieval을 지원
   - 100개 이상 언어
   - 최대 8192 token
   - embedding dimension 1024
   - long document retrieval 실험에 적합  
   참고: [BGE-M3 documentation](https://bge-model.com/bge/bge_m3.html)

   실험 포인트:
   - dense-only로 현재 파이프라인과 동일 조건 비교
   - sparse/hybrid 기능은 2차 실험으로 분리

3. `jinaai/jina-embeddings-v3`

   근거:
   - 논문/모델 카드 기준 multilingual 및 long-context retrieval 지향
   - 최대 8192 token
   - task-specific LoRA adapter와 Matryoshka representation 지원
   - 570M parameter, default dimension 1024  
   참고: [jina-embeddings-v3 paper](https://huggingface.co/papers/2409.10173), [model card](https://huggingface.co/jinaai/jina-embeddings-v3)

   실험 포인트:
   - retrieval adapter 사용 여부 비교
   - 1024/768/512 dimension truncation 비교

4. `Alibaba-NLP/gte-multilingual-mlm-base`

   근거:
   - mGTE 계열 backbone
   - 75개 언어
   - 최대 8192 token  
   참고: [gte-multilingual-mlm-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-mlm-base)

   실험 포인트:
   - embedding 모델이 아니라 MLM backbone 성격이므로 직접 embedding 품질은 별도 pooling/adapter 필요 여부 확인
   - baseline 후보보다는 ablation 후보로 취급

#### 2.2.3 Strong multilingual but short-context 후보

5. `intfloat/multilingual-e5-large`

   근거:
   - 100개 언어 계열 지원
   - training data에 S2ORC title/abstract pair와 MIRACL이 포함됨
   - dimension 1024  
   참고: [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)

   리스크:
   - model card에 long texts are truncated to at most 512 tokens라고 명시되어 있어 논문 초록이 긴 경우 불리할 수 있음

6. `intfloat/multilingual-e5-large-instruct`

   근거:
   - multilingual E5 instruct 계열
   - query instruction을 줄 수 있어 학술 검색 질의에 적합할 가능성
   - dimension 1024  
   참고: [multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)

   리스크:
   - long abstract truncation 문제
   - query/passsage prefix 및 instruction 설계가 성능에 큰 영향을 줄 수 있음

#### 2.2.4 2차 후보

7. Qwen3 Embedding 계열

   근거:
   - 공개 정보 기준 multilingual, long context embedding/reranker 계열로 보고됨
   - 119 languages, 32k context를 지원한다는 보도가 있음

   리스크:
   - 이 문서 작성 시점에는 1차 후보보다 프로젝트 내 재현성 검증 비용이 큼
   - 모델 크기별 비용 차이가 클 수 있음

   처리:
   - 1차 실험에서는 제외하고, GPU 자원 확보 후 2차 후보로 검토

### 2.3 실험 데이터셋 설계

실험은 세 계층으로 나눈다.

#### Layer A: 기존 ScienceON 테스트셋 기반

입력:

- `data/test.csv`
- 질문 50개
- 기존 `retrieved_article_name_*` 필드
- ScienceON 검색 결과

평가 목표:

- 실제 프로젝트 입력과 가장 유사한 end-to-end 성능
- ko/en 혼합 질의와 ScienceON 문서 title/abstract 대상 검색

문제:

- 정답 문서의 gold label이 완전하지 않을 수 있음
- ScienceON API 429가 embedding 모델 비교를 오염시킬 수 있음

대응:

- 검색 수집 corpus를 먼저 고정한다.
- 같은 고정 corpus에 대해 embedding 모델만 바꿔 retrieval을 비교한다.

#### Layer B: MIRACL 기반 검색 플랫폼 평가셋

입력:

- MIRACL ko/en query-document relevance 또는 기존 프로젝트의 MIRACL 기반 평가 포맷

평가 목표:

- 표준 IR 지표로 embedding 모델 retrieval 성능 비교
- ko/en multilingual 성능 분리 평가

지표:

- Recall@5, Recall@10, Recall@50
- MRR@10
- nDCG@10
- QSR: Top-K 안에 정답 문서 포함 여부

#### Layer C: 직접 구축한 ScienceON gold set

입력:

- `data/test.csv`에서 질문별 기대 논문 title
- 사람이 검수한 positive document set
- hard negative: 같은 키워드지만 다른 주제 논문

평가 목표:

- 학술 플랫폼 특화 검색 성능 검증
- title/abstract 기반 relevance 평가

구축 방법:

- 각 질문당 positive 1~3개
- 각 질문당 hard negative 5~10개
- 한국어 질문, 영어 질문, cross-lingual 질문으로 층화

### 2.4 Embedding 모델 비교 실험 구조

고정할 요소:

- 동일한 검색 corpus
- 동일한 chunking/document schema
- 동일한 query text
- 동일한 top-k
- 동일한 retrieval implementation
- 동일한 random seed

변경할 요소:

- embedding model
- query instruction/prefix
- max length
- pooling strategy
- embedding dimension

실험 matrix:

| ID | Model | Max Tokens | Dim | Query Instruction | Corpus |
|---|---:|---:|---:|---|---|
| E0 | gte-multilingual-base | 8192 | 768 | none | fixed ScienceON |
| E1 | bge-m3 dense | 8192 | 1024 | none | fixed ScienceON |
| E2 | jina-embeddings-v3 retrieval | 8192 | 1024 | retrieval adapter | fixed ScienceON |
| E3 | multilingual-e5-large | 512 | 1024 | `query:` / `passage:` | fixed ScienceON |
| E4 | multilingual-e5-large-instruct | 512 | 1024 | academic search instruction | fixed ScienceON |

### 2.5 논문 초록 길이 대응 실험

목표:

- 초록이 512 tokens를 넘을 때 short-context 모델이 어느 정도 손해를 보는지 확인한다.

조건:

1. title only
2. title + first 512 tokens of abstract
3. title + full abstract
4. `3*title + abstract`
5. title + abstract + source metadata

지표:

- Recall@5
- nDCG@10
- embedding throughput docs/sec
- memory footprint

판정:

- long-context 모델이 full abstract에서 이득을 보이면 bge-m3/jina/gte 계열 우선
- short-context 모델이 title+first512에서 충분히 강하면 비용/속도 기준으로 후보 유지

### 2.6 실험 산출물 구조

실험 코드와 일반 파이프라인 코드는 물리적으로 분리한다. 실험 전용 코드·runbook·case YAML·산출물은 모두 최상위 `experiments/` 아래에 둔다. 상세 규약은 `EXPERIMENT_BLUEPRINT.md` §0.3 참고.

- 실험 코드: `experiments/shared/` (evaluate / rerank / retrievers / query_transform / prompts / cases)
- 실험 runbook: `experiments/expN_*/run.sh`
- 실험 산출물: `experiments/outputs/<experiment_name>/<run_id>/`
- 일반 파이프라인 산출물은 기존대로 `outputs/run_<timestamp>/`
- `src/`·`pipeline/` → `experiments.*` import 금지 (단방향)

```text
experiments/outputs/embedding_benchmark/{run_id}/
  config.yaml
  corpus.jsonl
  queries.jsonl
  models/
    gte-multilingual-base/
      vectordb.csv
      retrieval.jsonl
      metrics.json
    bge-m3/
      vectordb.csv
      retrieval.jsonl
      metrics.json
  report.md
```

### 2.7 1차 결론 가설

- 현재 `gte-multilingual-base`는 long-context와 multilingual 조건을 만족하므로 baseline으로 타당하다.
- `bge-m3`는 multilingual, long document, hybrid 확장 가능성 때문에 가장 중요한 비교 후보이다.
- `jina-embeddings-v3`는 long-context와 retrieval adapter가 있어 강한 후보이다.
- `multilingual-e5-large` 계열은 학술 abstract pair와 MIRACL 학습 이력이 장점이지만 512 token truncation 때문에 긴 초록에서 불리할 수 있다.

---

## 3. 보완사항 2: 평가 방법 설계

### 3.1 현재 평가 철학 검토

현재 생각:

> 프로젝트의 새로움은 학술 검색 플랫폼 문서 수집 부분에 있고, retrieve와 응답 생성은 기존 RAG와 동일하므로, 수집 문서 중 정답 문서가 포함되어 있는지를 평가하는 것이 핵심이다.

이 생각은 상당히 타당하다. 특히 본 프로젝트가 “검색 플랫폼과 RAG를 접목”한 시스템이라면, novelty는 다음에 있다.

- 질문에서 검색어를 추출하는 방식
- ScienceON/PubMed/Wikipedia 같은 외부 플랫폼 API를 호출하는 방식
- 플랫폼별 결과를 공통 스키마로 수집하는 방식
- RAG용 후보 corpus를 구성하는 방식

따라서 “정답 문서가 수집 후보군에 들어왔는가”는 시스템의 가장 앞단이자 가장 프로젝트 고유한 부분을 직접 평가한다.

하지만 이것만으로 충분하지는 않다. 이유는 다음과 같다.

- 정답 문서가 수집되어도 embedding/retrieval에서 밀릴 수 있다.
- 올바른 문서가 top-k에 있어도 LLM이 답변에 반영하지 못할 수 있다.
- 반대로 정답 문서가 없어도 global VectorDB에서 유사 문서를 찾아 그럴듯한 답변을 만들 수 있다.
- 학술 검색 플랫폼 융합 시스템은 “문서 수집 성공”과 “근거 기반 답변 성공”을 분리해서 보여주는 편이 설득력이 높다.

결론:

- 검색 플랫폼 성능 평가는 반드시 유지한다.
- 단, end-to-end 답변 평가는 보조 축으로 추가해야 한다.
- 논문/보고서에서는 “platform acquisition”, “retrieval ranking”, “answer generation”을 분리해 제시하는 것이 가장 타당하다.

### 3.2 평가 축 제안

#### Axis 1: Platform Acquisition Evaluation

질문:

- 외부 학술 검색 플랫폼에서 정답 후보 문서를 가져왔는가?

지표:

- Acquisition Success Rate@N
- Gold Title Recall@N
- Gold DOI/CN Recall@N
- Query zero-doc rate
- API error rate
- 429 rate
- 평균 수집 문서 수

현재 MIRACL 기반 QSR은 이 축에 해당한다.

#### Axis 2: Retrieval Ranking Evaluation

질문:

- 수집된 corpus 안에서 정답 문서가 top-k에 올라왔는가?

지표:

- Recall@k
- MRR@k
- nDCG@k
- Precision@k

중요:

- Acquisition과 Retrieval을 분리해야 한다.
- 검색 실패 때문에 정답 문서가 corpus에 없었던 경우와, corpus에는 있었지만 retrieval이 실패한 경우를 구분해야 한다.

#### Axis 3: Context Quality Evaluation

질문:

- 최종 답변에 들어간 context가 질문에 관련 있고 충분한가?

지표:

- Context relevance
- Context precision
- Context recall
- Evidence coverage
- Citation/support coverage

RAGAS는 RAG 평가에서 context relevance, faithfulness, answer relevance 같은 reference-free 지표를 제안한다. [RAGAS paper](https://huggingface.co/papers/2309.15217)

#### Axis 4: Answer Quality Evaluation

질문:

- 생성 답변이 context에 충실하고 질문에 답하는가?

지표:

- Faithfulness
- Answer relevance
- Completeness
- Citation correctness
- Hallucination rate
- Korean/English style compliance

ARES는 context relevance, answer faithfulness, answer relevance를 평가 축으로 삼고, 소량의 human annotation으로 보정하는 방법을 제안한다. [ARES paper](https://huggingface.co/papers/2311.09476)

### 3.3 LLM-as-a-Judge 설계

LLM judge는 단독 점수보다 “근거 문서 기반 채점”으로 제한해야 한다.

Judge 입력:

```json
{
  "question": "...",
  "retrieved_contexts": [
    {"title": "...", "abstract": "...", "rank": 1}
  ],
  "answer": "...",
  "gold_answer": "... optional ...",
  "gold_doc_titles": ["... optional ..."]
}
```

Judge 출력:

```json
{
  "context_relevance": 1-5,
  "answer_faithfulness": 1-5,
  "answer_completeness": 1-5,
  "unsupported_claims": ["..."],
  "missing_key_points": ["..."],
  "verdict": "pass|partial|fail"
}
```

Judge prompt 원칙:

- 외부 지식 사용 금지
- context에 없는 내용은 unsupported로 표시
- 긴 답변 선호 금지
- gold answer가 있으면 semantic equivalence를 평가하되, context faithfulness를 별도 평가
- Korean question은 Korean answer 품질도 평가

실험 방법:

- 50개 `data/test.csv` 질문에 대해 사람이 10개 샘플 annotation
- LLM judge 2종 비교:
  - vLLM local judge
  - stronger external judge 가능 시 별도 비교
- human annotation과 judge score 상관 측정
- judge disagreement case를 수동 분석

판정 기준:

- judge score를 최종 유일 평가로 쓰지 않는다.
- 자동 회귀 테스트 및 모델 비교 보조 지표로 사용한다.

### 3.4 HYDE와 현재 파이프라인 비교 타당성 검토

HyDE는 query를 hypothetical document로 확장한 뒤 그 문서를 embedding하여 dense retrieval을 수행하는 zero-shot dense retrieval 기법이다. HyDE 논문은 relevance label이 없는 zero-shot retrieval에서 효과를 보였고, generated hypothetical document의 false detail은 dense bottleneck을 통해 실제 corpus에 grounding된다고 설명한다. [HyDE ACL paper](https://aclanthology.org/2023.acl-long.99/), [HF paper page](https://huggingface.co/papers/2212.10496)

비교가 타당한 경우:

- 동일한 fixed corpus가 있을 때, query embedding 방식의 성능을 비교하는 경우
- 예: `raw query embedding` vs `HyDE document embedding`
- embedding/retrieval component 개선을 보고 싶은 경우

비교가 부적절한 경우:

- ScienceON API 문서 수집 성능과 HyDE를 직접 비교하는 경우
- HyDE는 검색 플랫폼 API 호출 전략이 아니라 dense retrieval query transformation이다.
- 따라서 본 프로젝트의 novelty인 “학술 플랫폼에서 어떤 문서를 수집해 corpus를 만드는가”를 HyDE가 대체 평가하지 못한다.

권장 비교 설계:

1. Acquisition 단계는 고정

   동일한 ScienceON 수집 corpus를 사용한다.

2. Retrieval query formulation만 비교

   - Baseline: original question embedding
   - Keyword pipeline: extracted keywords/search terms 기반 question
   - HyDE: generated hypothetical abstract embedding
   - Multi-HyDE: 여러 hypothetical abstract 평균 또는 union retrieval

3. 동일한 embedding model 사용

   embedding 모델 차이와 HyDE 효과를 분리한다.

4. 지표

   - Recall@5/10
   - MRR@10
   - nDCG@10
   - LLM judge context relevance
   - latency/cost

결론:

- HyDE와 비교하는 것은 “retrieval query representation 개선” 평가로는 타당하다.
- 하지만 “학술 검색 플랫폼 융합 시스템 전체의 novelty” 평가로는 불충분하다.
- 따라서 논문/보고서에서는 HyDE를 main baseline이 아니라 retrieval ablation baseline으로 두는 것이 적절하다.

### 3.5 최종 평가 설계안

#### Experiment 1: Platform Acquisition

목표:

- ScienceON/PubMed/Wikipedia 검색 클라이언트와 keyword extraction의 문서 수집 성능 평가

조건:

- extractor: gemini, vllm, raw query
- source: ScienceON
- retry policy: off/on

지표:

- QSR@50
- zero-doc rate
- avg collected docs
- 429 rate
- exact title match / normalized title match

#### Experiment 2: Embedding Retrieval

목표:

- 고정 corpus에서 embedding 모델별 retrieval ranking 비교

조건:

- gte-multilingual-base
- bge-m3
- jina-embeddings-v3
- multilingual-e5-large
- multilingual-e5-large-instruct

지표:

- Recall@5/10
- MRR@10
- nDCG@10
- throughput
- memory

#### Experiment 3: Query Representation Ablation

목표:

- raw query, keyword query, HyDE query가 retrieval 성능에 미치는 영향 비교

조건:

- same corpus
- same embedding model
- same top-k

비교군:

- Raw question
- Extracted keywords joined
- vLLM rewritten search query
- HyDE hypothetical abstract
- Multi-HyDE 3 samples

지표:

- Recall@5/10
- MRR@10
- context relevance judge score
- query generation latency

#### Experiment 4: End-to-End Answer Evaluation

목표:

- 최종 답변이 retrieved context에 충실한지 평가

조건:

- best acquisition + best embedding + best query representation
- vs baseline current pipeline

지표:

- LLM judge faithfulness
- LLM judge answer relevance
- unsupported claim count
- human spot-check pass rate
- optional BLEU/METEOR는 보조로만 사용

### 3.6 보고서 구조 제안

최종 실험 보고서는 아래처럼 분리한다.

```text
1. Platform Acquisition Performance
   - 학술 검색 플랫폼 융합의 핵심 성과

2. Retrieval Ranking Performance
   - embedding/query representation 성능

3. Answer Faithfulness and Usefulness
   - RAG 최종 사용자 경험 품질

4. Error Analysis
   - 429, zero-doc, wrong-doc, hallucination

5. Cost and Latency
   - API 호출 수, vLLM 호출 수, embedding 시간
```

---

## 4. 우선순위 로드맵

### P0: 안정성

- ScienceON 요청률 제어 (throttle / AIMD, §1.3 Step 1)
- Retry + exponential backoff (§1.3 Step 1)
- 요청 캐시 (§1.3 Step 1)
- max_concurrency sweep 으로 운영 지점 확정 (§1.3 Step 2)

### P1: 평가 신뢰성

- Acquisition/Retrieval/Answer 평가 분리
- QSR@N, Recall@K, MRR, nDCG 구현
- LLM-as-a-judge schema 구현
- human annotation 10~20개로 judge calibration

### P2: Embedding 모델 비교

- fixed corpus benchmark 구축
- gte/bge/jina/e5 비교
- long abstract truncation 실험

### P3: Retrieval 개선

- HyDE ablation
- reranker 도입 실험

### P4: 논문화/보고서화

- 학술 검색 플랫폼 수집 성능을 핵심 novelty로 제시
- retrieval/generation은 보조 실험으로 분리
- API 장애/429에 대한 robustness 분석 포함

---

## 5. 참고 자료

- [Alibaba-NLP/gte-multilingual-base model card](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
- [BGE-M3 documentation](https://bge-model.com/bge/bge_m3.html)
- [jina-embeddings-v3 paper](https://huggingface.co/papers/2409.10173)
- [jinaai/jina-embeddings-v3 model card](https://huggingface.co/jinaai/jina-embeddings-v3)
- [intfloat/multilingual-e5-large model card](https://huggingface.co/intfloat/multilingual-e5-large)
- [intfloat/multilingual-e5-large-instruct model card](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
- [HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels](https://aclanthology.org/2023.acl-long.99/)
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://huggingface.co/papers/2309.15217)
- [ARES: Automated Evaluation Framework for RAG Systems](https://huggingface.co/papers/2311.09476)

---

## 6. 검토 보완 사항 (Review)

작성일: 2026-04-22

이 절은 섹션 0~5를 실제 코드(`pipeline/`, `src/`, `configs/`)와 대조하여 검토한 결과이며, 원 설계를 유지한다는 전제에서 빠져 있던 선행 조건/위험/우선순위를 추가한다.

> **갱신 주기 (2026-04-23)**: 본 절은 plan 초판(설계안 A–D를 모두 가정)에 대한 리뷰다. 이후 §1.3 을 *throttle+retry+cache + concurrency sweep* 중심으로 재편하면서 구 설계안 B(per-question `search_status`), C(zero-doc fallback ladder), D(retrieval provenance) 는 제거되었다. 아래 항목 중 구 §1.4/1.5/1.6 / `fallback_*` / `provenance` 에 의존한 논의(6.1.6, 6.4의 "provenance" 맥락, 6.8 P3 provenance 실험, 6.9 의 fallback/provenance 정렬)는 **더 이상 유효하지 않다**. dedup 키 전환(6.4의 실체) 과 BM25·reranker·재현성 관련 서술(6.5–6.7) 은 계속 유효하다.

### 6.1 계획 전반 평가

탄탄한 부분:

- Platform Acquisition / Retrieval / Context / Answer의 4축 분리(3.2)는 본 프로젝트의 novelty(학술 플랫폼 융합)와 정확히 정렬된다. 한 지표(QSR)만 보고 결론을 내리지 않도록 막는 가장 중요한 구조이다.
- HyDE를 main baseline이 아니라 retrieval query ablation으로 위치시킨 판단(3.4)은 정확하다. Acquisition과 HyDE를 직접 비교하는 것은 범주 오류이다.
- 1.2의 "API 장애가 retrieval/answer 평가 점수를 오염시킬 수 있다"는 분리 분석은 평가 신뢰성의 핵심이며, 이 인식이 섹션 3 축 분리와도 일관된다.

보완이 필요한 부분(아래 6.2~6.8에서 순차적으로 다룸):

1. P0로 올려야 할 선행 버그 — embedding_dim 설정 오류.
2. Rate limit 대응이 retry 중심이며 concurrency control / 요청 캐시 / resume이 빠져 있다.
3. Retrieval baseline에 BM25 sparse가 없고 reranker가 P3로 밀려 있다.
4. LLM-as-judge의 bias 통제와 통계적 유의성 판정 기준이 없다.
5. 재현성/관측가능성(seed, keyword extractor 결정성, per-question trace) 설계가 없다.
6. 1.6 provenance 스키마와 현재 `src/utils/dedup.py` 로직이 정면으로 충돌한다.
7. 계획서와 `AGENTS.md`의 리팩토링 순서(T7/T8/T9)가 연동되지 않았다.

### 6.2 [P0 선행] embedding_dim 설정 정정

현재 `configs/query_encoder/config_gte-multilingual-base.json`:

```json
{ "model_name": "Alibaba-NLP/gte-multilingual-base", "embedding_dim": 1024, ... }
```

`Alibaba-NLP/gte-multilingual-base`의 native dimension은 **768**이다(Matryoshka로 128~768 절단 가능). `src/features/embedding_processor.py` 는 이 값을 `SentenceTransformer(..., truncate_dim=embedding_dim)` 로 그대로 전달한다. 현재는 sentence-transformers가 native(768)로 silently fallback하여 파이프라인 자체는 완주하지만, 섹션 2의 임베딩 모델 비교 실험에서 이 불일치가 잠복하면 모델별 결과 해석이 오염된다(특히 Matryoshka를 의도적으로 쓰는 경우와 구분 불가).

조치:

- 768로 정정하거나, Matryoshka truncation 실험용이면 별도 config 파일(예: `config_gte-multilingual-base_d256.json`)로 분기.
- 모든 embedding config에서 `embedding_dim`이 "모델 native dim" 인지 "의도한 Matryoshka 절단 차원" 인지 주석으로 명시.
- `build_vectordb_search` 말미에 `assert embeddings.shape[1] == config["embedding_dim"]` 추가 → 향후 silent mismatch 차단.

이 수정이 없으면 섹션 2.4 실험 매트릭스(E0~E4) 결과의 신뢰성을 확보할 수 없다.

### 6.3 섹션 1 보강: Rate limit 전략

섹션 1.3은 per-request retry + backoff에 집중하지만, HTTP 429의 근본 원인은 대부분 *throughput 초과*이다. retry-only 설계는 같은 요청을 여러 번 보내 오히려 전체 QPS를 끌어올린다. 다음 세 층을 추가해야 한다.

1) Global throttle (concurrency control)

- `ScienceONAdapter` 생성자에 `max_concurrency`, `min_interval_sec` 추가. 단일 프로세스 기준 `threading.Semaphore` + `time.monotonic()` 기반.
- 429 발생 시 `min_interval_sec`을 AIMD(Additive Increase, Multiplicative Decrease)로 adaptive 조정.

2) 요청 수준 캐시

- 동일 `search_term`은 ScienceON 기준 동일 응답. 50 질문이 질문당 3~5 term을 쓰면 150~250 호출이지만 term 중복이 높다. 캐시 hit이 상당히 생긴다.
- 캐시 키: `(source, normalized_term, cur_page, row_count, fields)`, 값: raw JSON 응답.
- 저장 위치: `outputs/<run>/search/_cache/<sha1>.json`. Run 간 재사용하려면 `configs/credentials/` 옆이 아닌 `outputs/_shared_cache/`로 승격.
- 효과: 429 빈도↓, 재현성↑, 비용↓ — 1.3 실험에서 "총 429 수" 지표에 직접 영향.

3) Idempotent resume

- `search_meta_results.json`을 질문 1개 처리할 때마다 append-flush.
- 재실행 시 `question_id not in processed AND status != "success"`인 질문만 재호출.
- 1.4의 `search_status` 스키마와 결합하면 `api_failed` 상태 질문만 선택 재시도하는 "repair run" 모드를 자연스럽게 구성할 수 있다.

### 6.4 섹션 1.6 provenance —  dedup 키 변경


- 현재 키 `title` — 학회 proceedings 동명 논문, 서평, 번역판 등 "같은 title 다른 문서"가 실제로 존재. title 키 dedup은 서로 다른 문서를 하나로 합치는 false positive를 만든다.
- 키를 ScienceON의 `CN`(ScienceONAdapter가 `doc_id`로 매핑) 또는 source별 고유 id로 변경. PubMed는 PMID, Wikipedia는 pageid 또는 title+lang.
- 이는 `AGENTS.md` T9(중복 제거 로직 통일)에 포함되는 것이 자연스럽다.

### 6.5 섹션 2 보강: BM25 baseline, reranker, Layer C 비용

1) BM25 / sparse baseline 부재

학술 검색에서 BM25는 여전히 강한 baseline이며, monolingual 슬라이스(한국어 질문 → 한국어 abstract)에서는 dense를 종종 이긴다. 섹션 2.4 실험 매트릭스에 `E-BM25` 행을 추가해, dense의 개선이 sparse 대비 *실제* 이득인지를 확인해야 한다. 1070 문서 규모에서는 `rank_bm25` 또는 `bm25s`로 즉시 실행 가능하며, tokenizer 선택(Mecab vs 공백)에 대한 ko/en 각각의 비교만 추가로 설계하면 된다.

2) Reranker가 P3로 밀려 있는 문제

`bge-reranker-v2-m3`, `jina-reranker-v2-base-multilingual` 같은 multilingual cross-encoder reranker는 Top-50 → Top-5 재정렬에서 일관된 이득이 있다. retrieval ranking 축(Axis 2)의 필수 비교군으로 올려야 한다. 실험 3(Query Representation Ablation)의 각 조건에 `+rerank` 변형을 붙이는 구성이 정보량이 가장 높다.

3) Layer C(자체 gold set) 주석 비용 누락

50 questions × (positive 1~3 + hard negative 5~10) = **300~650 판정**. 건당 2~5분으로 보면 10~50시간 분량. 실질적으로 한 주 내 1인 완성이 어렵다. 파일럿으로 10~15 질문 서브셋을 먼저 구축하고, Layer B(MIRACL)로 대부분의 embedding 비교를 마친 뒤 Layer C는 최종 후보 1~2개 모델 검증용으로 제한하는 것이 현실적이다.

### 6.6 섹션 3.3 보강: Judge bias & 통계 유의성

현재 judge 설계에 빠진 방어선:

- **Self-preference bias**: 답변 생성에 `gpt-oss-20b`를 쓰고 judge도 동일 계열이면 자기편향이 들어간다. Judge 모델은 다른 family(예: `gemini-2.5-flash` 또는 외부 strong judge)로 고정하거나, 같은 family로 평가할 경우 반대 방향(gemini로 생성 → gpt-oss로 judge)을 교차 수행해 편향 크기를 정량화한다.
- **Position / length bias**: pairwise 비교라면 A-B, B-A 양 방향을 평균. pointwise 채점이라면 prompt에 "길이는 점수에 무관" 명시 + 답변 길이 분포를 metadata로 남겨 사후 회귀.
- **통계 유의성**: 50 질문에서 Recall@10 0.62 vs 0.68 의 절대차이가 실제 유의한지 판정하려면 **paired bootstrap**(1000 resample, 95% CI) 또는 sign test가 필요하다. 계획서에는 "판정 기준"은 있지만 "유의성 기준"이 없다. 제안 판정 규칙: *(절대차이 ≥ 0.03) **AND** (95% CI 하한 > 0)* 이어야 "우세", 그 외는 tie. Tie인 경우 latency/cost를 tie-breaker로 쓴다.

### 6.7 추가 축: 재현성 & 관측가능성

계획서 전반에 명시적으로 빠져 있는 operational 요건.

- **Keyword extractor 결정성**: vLLM/Gemini keyword 추출은 샘플링 기반이라 run-to-run으로 `search_terms`가 바뀌고, 이것이 1.2의 "run-to-run 재현성 저하"에 본인이 기여한다. 모든 benchmark 실행은 `temperature=0`, seed 고정(vLLM은 `seed` 지원, Gemini는 불완전) 을 명시한다. 엄격한 재현성이 필요한 비교 실험에서는 keyword extraction을 1회만 수행하고 산출된 `search_terms`를 JSONL로 고정한 뒤 이후 run에서 재사용하는 "frozen queries" 모드를 만든다.
- **Per-질문 trace 로그**: 한 `question_id`가 5단계를 통과하는 동안 생성된 모든 산출물을 단일 키로 join할 수 있어야 한다. 현재 step5 결과에 `used_context: [{doc_id, rank, from_step: "step4"}]` 가 남지 않으면 Axis 3(context quality) 평가 데이터를 재구성할 수 없다. step5 schema에 해당 필드를 추가.
- **Run manifest**: `outputs/<run>/run_manifest.json` 에 git SHA, `requirements.txt` 해시, vLLM 모델/버전, 전체 CLI 인수, 시작/종료 시각, ScienceON credential fingerprint(값 제외, SHA1)를 저장. 2.6 실험 산출물 구조와 통합.

### 6.8 우선순위 재조정

섹션 4 P0~P4에 다음을 삽입/조정한다.

- **P0 (안정성)** 추가:
  - `embedding_dim` config 정정 + assertion (6.2) — **이것이 없으면 섹션 2 전체 실험 결과가 오염**
  - ScienceON 요청 캐시 + resume checkpoint (6.3-2, 6.3-3)
  - Dedup 키를 `title` → `doc_id`(source별 고유 id)로 변경 (6.4) — merge 전환은 Option A 채택 시에만 후속 조치
  - run_manifest + step5 `used_context` 필드 (6.7)

- **P1 (평가 신뢰성)** 추가:
  - BM25 baseline 구현 (6.5-1)
  - Paired bootstrap 유의성 검정 유틸 (6.6)
  - Judge 교차검증 프로토콜(생성-judge 모델 family 분리) (6.6)
  - Frozen queries 모드(keyword extraction 결정성) (6.7)

- **P2 (Embedding 비교)** 수정:
  - Reranker를 P3에서 P2로 상향. Embedding 비교와 같은 매트릭스에서 `+rerank` 변형까지 한 번에 측정하면 2차 실행 비용이 크게 준다 (6.5-2).
  - Layer C는 10~15 질문 파일럿으로 제한 (6.5-3).

- **P3 (Retrieval 개선)** 유지:
  - HyDE / Multi-HyDE ablation
  - (구 provenance 실험은 §1.3 개편으로 제거됨 — §6 서두 갱신 주기 참조.)

### 6.9 AGENTS.md 리팩토링 순서와 정합성

계획서의 새 메타데이터·fallback·provenance는 `AGENTS.md`의 리팩토링 작업과 순서 의존성이 있다.

- **T9(dedup 통합) → 6.4(dedup 키 `title` → `doc_id`) → 1.6(provenance)** 이 정상 순서. 1.6의 provenance는 런타임 join(6.4 Option B)으로 먼저 구현하고, corpus 내장(Option A)이 필요한 시점에만 merge 로직을 T9 위에 덧붙인다.
- **T7(BaseSearchClient 베이스)** 완료 전에 1.5 zero-doc fallback을 adapter 안쪽에 넣으면 각 source별로 fallback 로직이 중복된다. T7 이후 `SearchMetaSystem` 레벨에서 fallback 오케스트레이션(T8 방향: 키워드 추출을 상위에서 수행)으로 구현하는 것이 일관된다.
- **T8(Integration 순환 의존 제거)** 는 1.6의 `collected_by_terms` 기록 위치 결정과 연결된다. Keyword 추출이 `SearchMetaSystem` 레벨로 올라오면 `collected_by_terms`도 그 레벨에서 자연스럽게 주입된다.
