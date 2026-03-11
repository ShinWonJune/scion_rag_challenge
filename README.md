# SHRAG: A Framework for Combining Human-Inspired Search with RAG

SHRAG (Search like Human with RAG) 프레임워크의 사용 설명서 입니다.

## Architecture

- `pipeline/`: CLI 엔트리포인트 + 단계별 오케스트레이션
- `src/`: 재사용 가능한 로직(검색 클라이언트, 임베딩, 검색, 평가)


## Core Paths

- 설정: `configs/`
- 데이터: `data/`
- 실행 산출물: `outputs/`
- 평가 결과(선택): `results/`

## Setup

```bash
pip install -r requirements.txt
```

환경변수 파일은 `.env.example`을 기준으로 준비합니다.

```bash
cp .env.example .env
```

로컬 LLM(vLLM) 사용 시 `docker-compose.yml` 기준으로 서버를 먼저 실행합니다.

1. `docker-compose.yml`에서 아래를 환경에 맞게 수정
- `volumes`의 모델 경로
- `device_ids`의 GPU 번호
- 필요 시 `--model`, `--port`, `--tensor-parallel-size`

2. 서버 실행

```bash
docker compose up -d
```

3. 서버 상태 확인

```bash
docker compose logs -f vllm-rag
curl http://localhost:8004/v1/models
```

## E2E Run

```bash
python pipeline/run_pipeline.py \
  --questions data/test.csv \
  --encoder configs/query_encoder/config_gte-multilingual-base.json \
  --extractor vllm \
  --llm vllm \
  --sources scienceon \
  --vllm-url http://localhost:8004/v1 \
  --vllm-model openai/gpt-oss-20b
```

질문 분해 단계(step2_decompose) 포함 실행:

```bash
python pipeline/run_pipeline.py \
  --questions data/test.csv \
  --encoder configs/query_encoder/config_gte-multilingual-base.json \
  --extractor vllm \
  --llm vllm \
  --sources scienceon \
  --vllm-url http://localhost:8004/v1 \
  --decompose
```

## Step Entrypoints

- `pipeline/step1_search.py`
- `pipeline/step2_decompose.py`
- `pipeline/step3_build_vectordb.py`
- `pipeline/step4_retrieve.py`
- `pipeline/step5_generate.py`

## Output Paths

경우별 생성 경로는 아래와 같습니다.

1. `run_pipeline.py` 실행
- `--output` 지정 시: 지정한 폴더를 실행 루트로 사용
  - 예: `--output outputs/e2e_vllm20b`
- `--output` 미지정 시: `outputs/run_<timestamp>/`
- 내부 구조:
  - `search/`
  - `decompose/` (옵션)
  - `retrieval/`
  - `final/`
- 참고: `run_pipeline` 실행 시 Step4는 `retrieval/` 아래에 추가 타임스탬프 폴더를 만들지 않습니다.

2. `step1_search.py` 단독 실행
- 기본 루트: `outputs/search/`
- 내부에 타임스탬프 하위 폴더가 생성됨
- 생성 파일:
  - `search_meta_results.json`
  - `search_documents.jsonl`
- 커스텀: `--output-dir <path>`

3. `step2_decompose.py` 단독 실행
- 기본 루트: `outputs/decompose/`
- 내부에 타임스탬프 하위 폴더가 생성되며 `singlehop_decompose.jsonl` 저장
- 커스텀: `--output-dir <dir_path>`
- 호환 옵션: `--output <dir_or_file_path>`

4. `step4_retrieve.py` 단독 실행
- 기본 루트: `outputs/retrieval/`
- 내부에 타임스탬프 하위 폴더가 생성되며, 질의별 JSON 저장
- 커스텀: `--output-dir <path>`
- 호환 옵션: `--output-root <path>`

5. `step5_generate.py` 단독 실행
- 기본 루트: `outputs/final/`
- 내부에 타임스탬프 하위 폴더가 생성됨
- 커스텀: `--output-dir <path>`

## Evaluation

Step1 SEARCH 평가:

```bash
python pipeline/evaluate/eval_search.py \
  --results outputs/search/search_meta_results.json \
  --ground_truth <ground_truth_json> \
  --source scienceon
```

Step5 GENERATE 평가:

```bash
python pipeline/evaluate/eval_answers.py \
  --predictions <predictions_csv> \
  --ground_truth <ground_truth_csv> \
  --dataset scienceon
```

## Notes

- 현재 Step1은 단일 source 실행 정책입니다.
- 과거 실험/분석 산출물은 `outputs/old_outputs/` 및 `pipeline/evaluate/legacy/`로 분리 관리하는 것을 권장합니다.
- 평가 CLI는 `run_pipeline.py`에서 자동 호출되지 않으며, 실행 후 별도로 수행합니다.
