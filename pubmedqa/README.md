# PubMedQA Dataset for RAG Evaluation

이 디렉토리는 HuggingFace의 PubMedQA 데이터셋(`qiaojin/PubMedQA`)을 사용하여 RAG(Retrieval-Augmented Generation) 시스템 평가를 위한 데이터를 준비한 것입니다.

## 데이터셋 개요

PubMedQA는 생물의학 분야의 질문-답변 데이터셋으로, PubMed 논문을 기반으로 한 질문과 답변이 포함되어 있습니다. 각 질문에 대해 yes/no/maybe의 답변과 상세한 설명이 제공됩니다.

## 파일 구조

```
pubmedqa/
├── README.md                          # 이 파일
├── download_and_prepare_data.py       # 원본 데이터셋 다운로드 스크립트
├── create_sample_csvs.py             # CSV 파일 생성 스크립트
├── add_context_to_csv.py             # context 데이터 추가 스크립트
├── dataset_info.json                 # 원본 데이터셋 정보
├── subquestion.csv                   # 질문만 포함된 CSV (batch processing용)
├── pubmedqa_complete.csv             # 전체 데이터가 포함된 CSV
└── pubmedqa_evaluation.csv           # 평가용 CSV
```

## CSV 데이터셋 구조

### 1. subquestion.csv
RAG 시스템의 batch processing을 위한 질문 데이터입니다.

| 컬럼 | 설명 | 예시 |
|------|------|------|
| question | 생물의학 관련 질문 | "Are complex coronary lesions more frequent in patients with diabetes mellitus?" |

- **용도**: `batch_query_processor._load_queries_from_csv()` 메서드와 호환
- **샘플 수**: 50개
- **특징**: 첫 번째 컬럼에 질문만 포함

### 2. pubmedqa_complete.csv
전체 정보가 포함된 완전한 데이터셋입니다.

| 컬럼 | 타입 | 설명 | 예시 |
|------|------|------|------|
| question | string | 생물의학 관련 질문 | "Are complex coronary lesions more frequent in patients with diabetes mellitus?" |
| context | string | 답변을 위한 abstract의 모음 | "Coronary atherosclerotic burden is excessive in diabetic patients..." |
| long_answer | string | 질문에 대한 상세한 답변 | "Complex coronary lesions such as bifurcation and ostial lesions were significantly more common..." |
| final_decision | string | 최종 결정 (yes/no/maybe) | "yes" |
| pubid | integer | PubMed 논문 ID | 16971978 |

- **용도**: RAG 시스템 개발 및 상세 분석
- **샘플 수**: 50개
- **특징**: 질문, 컨텍스트, 답변, 결정이 모두 포함

### 3. pubmedqa_evaluation.csv
RAG 시스템 성능 평가를 위한 ground truth 데이터입니다.

| 컬럼 | 타입 | 설명 | 예시 |
|------|------|------|------|
| question | string | 평가할 질문 | "Are complex coronary lesions more frequent in patients with diabetes mellitus?" |
| ground_truth_answer | string | 정답 (긴 답변) | "Complex coronary lesions such as bifurcation and ostial lesions were significantly more common..." |
| ground_truth_decision | string | 정답 결정 (yes/no/maybe) | "yes" |

- **용도**: RAG 시스템 출력과 정답 비교
- **샘플 수**: 50개
- **특징**: 평가 메트릭 계산에 최적화

## 사용 방법

### 1. 데이터 다운로드 및 준비
```bash
cd /app/pubmedqa
python download_and_prepare_data.py  # 원본 데이터셋 다운로드
python create_sample_csvs.py         # 샘플 CSV 파일 생성
python add_context_to_csv.py         # context 데이터 추가
```

### 2. RAG 시스템에서 활용
```bash
# batch processing 실행
cd /app/search_science_on_challenge
TARGET_DOCUMENTS=50 python main.py --use-vllm --use-pubmed batch ../pubmedqa/subquestion.csv
```

### 3. 평가 데이터로 활용
- `pubmedqa_evaluation.csv`를 사용하여 RAG 시스템의 성능 평가
- 의미적 유사도, BLEU 점수, 결정 정확도 등 다양한 메트릭 계산 가능

## 데이터 특징

### 질문 유형
- 생물의학, 임상, 약물학 관련 질문
- Yes/No 답변이 가능한 구체적 질문
- 논문 데이터를 기반으로 한 사실적 질문

### 답변 형태
- **final_decision**: yes (긍정), no (부정), maybe (불확실)
- **long_answer**: 결정에 대한 상세한 근거와 설명
- **context**: 논문의 배경, 방법론, 결과 등 상세 정보

### 샘플링 정보
- 원본 데이터셋: `pqa_labeled` (1,000개 샘플)
- 추출된 샘플: 50개 (random seed=42로 재현 가능)
- 데이터 분포: 다양한 생물의학 주제 포함

## 주의사항

1. **인코딩**: 모든 CSV 파일은 UTF-8로 인코딩되어 있습니다.
2. **컨텍스트 길이**: context 필드는 매우 길 수 있으므로 (평균 1,500자) 처리 시 주의가 필요합니다.
3. **결정 값**: final_decision은 소문자로 통일되어 있습니다 (yes/no/maybe).
4. **CSV 구분자**: 쉼표(,)를 사용하며, 텍스트 내 쉼표는 따옴표로 감싸져 있습니다.

## 라이선스 및 출처

- **원본 데이터셋**: qiaojin/PubMedQA (HuggingFace)
- **논문**: Jin et al., "PubMedQA: A Dataset for Biomedical Research Question Answering"
- **라이선스**: 원본 데이터셋의 라이선스를 따름