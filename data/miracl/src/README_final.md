# MIRACL 한국어 & 영어 데이터셋

이 폴더는 MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) 데이터셋의 한국어와 영어 데이터를 포함합니다.

## 📊 데이터 통계

### 한국어 (Korean)
- **Dev 쿼리**: 213개
- **Dev QRELs**: 3,057개  
- **Train 쿼리**: 868개
- **Train QRELs**: 12,767개
- **코퍼스 샘플**: 500개 문서

### 영어 (English)  
- **Train 쿼리**: 2,863개
- **Train QRELs**: 29,416개
- **코퍼스 샘플**: 500개 문서

## 📁 파일 구조

```
miracl/
├── korean 데이터
│   ├── miracl-v1.0-ko/
│   │   ├── topics/
│   │   │   ├── topics.miracl-v1.0-ko-dev.tsv     # Dev 쿼리
│   │   │   └── topics.miracl-v1.0-ko-train.tsv   # Train 쿼리  
│   │   └── qrels/
│   │       ├── qrels.miracl-v1.0-ko-dev.tsv      # Dev 관련성 판정
│   │       └── qrels.miracl-v1.0-ko-train.tsv    # Train 관련성 판정
│   ├── topics_ko_dev.jsonl                       # Dev 쿼리 (JSONL)
│   ├── qrels_ko_dev.jsonl                        # Dev QRELs (JSONL)
│   └── corpus_ko_real_sample.jsonl               # 한국어 문서 샘플
├── english 데이터  
│   ├── miracl-v1.0-en/
│   │   ├── topics/
│   │   │   └── topics.miracl-v1.0-en-train.tsv   # Train 쿼리
│   │   └── qrels/
│   │       └── qrels.miracl-v1.0-en-train.tsv    # Train 관련성 판정
│   └── corpus_en_sample.jsonl                    # 영어 문서 샘플
└── metadata_final.json                           # 전체 메타데이터
```

## 🔍 데이터 형식

### Topics (쿼리)
```tsv
query_id    query_text
2          합성생물학을 연구하는 방식은 탑다운 외 다른 방식은 무엇이 있나요?
```

### QRELs (관련성 판정) 
```tsv
query_id   Q0   doc_id      relevance
2         0    317339#6    1
```

### Corpus (문서)
```json
{
    "docid": "5#0",
    "title": "지미 카터", 
    "text": "제임스 얼 \"지미\" 카터 주니어는..."
}
```

## 🚀 사용 예시

### Python으로 데이터 로드
```python
import json

# 한국어 Dev 쿼리 로드
with open('topics_ko_dev.jsonl', 'r', encoding='utf-8') as f:
    ko_queries = [json.loads(line) for line in f]

# 영어 코퍼스 로드  
with open('corpus_en_sample.jsonl', 'r', encoding='utf-8') as f:
    en_corpus = [json.loads(line) for line in f]

print(f"한국어 쿼리 수: {len(ko_queries)}")
print(f"영어 문서 수: {len(en_corpus)}")
```

### pandas로 TSV 파일 로드
```python
import pandas as pd

# 한국어 Train QRELs
ko_qrels = pd.read_csv('miracl-v1.0-ko/qrels/qrels.miracl-v1.0-ko-train.tsv', 
                      sep='\t', names=['query_id', 'Q0', 'doc_id', 'relevance'])

# 영어 Train Topics  
en_topics = pd.read_csv('miracl-v1.0-en/topics/topics.miracl-v1.0-en-train.tsv',
                       sep='\t', names=['query_id', 'query'])
```

## 📚 참고자료

- **논문**: [MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages](https://arxiv.org/abs/2210.09984)
- **공식 사이트**: [miracl.ai](http://miracl.ai)  
- **HuggingFace**: [miracl/miracl](https://huggingface.co/datasets/miracl/miracl)
- **GitHub**: [project-miracl/miracl](https://github.com/project-miracl/miracl)

## 💡 활용 방법

이 데이터셋은 다음과 같은 정보검색 태스크에 사용할 수 있습니다:

1. **다국어 정보검색** - 한국어/영어 쿼리에 대한 문서 검색
2. **크로스링구얼 검색** - 한국어 쿼리로 영어 문서 검색 (또는 반대)  
3. **검색 시스템 평가** - QRELs를 이용한 검색 성능 측정
4. **임베딩 모델 학습** - 쿼리-문서 쌍을 이용한 학습 데이터

점수 체계:
- **0**: 관련 없음
- **1**: 부분적으로 관련  
- **2**: 매우 관련