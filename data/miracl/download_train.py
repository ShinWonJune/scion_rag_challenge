#!/usr/bin/env python3
"""
MIRACL 한국어 train 데이터 다운로드 및 변환
"""

from huggingface_hub import hf_hub_download
import json
from pathlib import Path

def download_train_data():
    """MIRACL 한국어 train 데이터를 다운로드하고 변환합니다."""
    
    print('📥 MIRACL 한국어 train qrels 다운로드 중...')
    try:
        train_qrels_path = hf_hub_download(
            repo_id='miracl/miracl',
            filename='miracl-v1.0-ko/qrels/qrels.miracl-v1.0-ko-train.tsv',
            local_dir='.',
            repo_type='dataset'
        )
        print(f'✅ Train qrels 다운로드 완료: {train_qrels_path}')
        
        # 파일 크기와 라인 수 확인
        with open(train_qrels_path, 'r', encoding='utf-8') as f:
            qrels_lines = f.readlines()
        
        print(f'📊 총 {len(qrels_lines)}개의 train qrels')
        
        # JSONL 변환
        print('🔄 JSONL 변환 중...')
        
        # Topics JSONL 변환
        train_topics_jsonl = []
        with open('miracl-v1.0-ko/topics/topics.miracl-v1.0-ko-train.tsv', 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    train_topics_jsonl.append({
                        'query_id': parts[0],
                        'query': parts[1]
                    })
        
        with open('topics_ko_train.jsonl', 'w', encoding='utf-8') as f:
            for topic in train_topics_jsonl:
                f.write(json.dumps(topic, ensure_ascii=False) + '\n')
        
        # QRELs JSONL 변환
        train_qrels_jsonl = []
        with open(train_qrels_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    train_qrels_jsonl.append({
                        'query_id': parts[0],
                        'corpus_id': parts[2],
                        'score': int(parts[3])
                    })
        
        with open('qrels_ko_train.jsonl', 'w', encoding='utf-8') as f:
            for qrel in train_qrels_jsonl:
                f.write(json.dumps(qrel, ensure_ascii=False) + '\n')
        
        print(f'✅ JSONL 변환 완료')
        print(f'   - topics_ko_train.jsonl: {len(train_topics_jsonl)}개')
        print(f'   - qrels_ko_train.jsonl: {len(train_qrels_jsonl)}개')
        
        # 메타데이터 업데이트
        metadata = {
            "dataset": "MIRACL Korean (Complete with Train)",
            "language": "ko",
            "source": "HuggingFace: miracl/miracl & miracl/miracl-corpus",
            "train_queries": len(train_topics_jsonl),
            "train_qrels": len(train_qrels_jsonl),
            "dev_queries": 213,
            "dev_qrels": 3057,
            "corpus_sample": 500,
            "download_date": "2025-09-27",
            "files": {
                "train_topics_tsv": "miracl-v1.0-ko/topics/topics.miracl-v1.0-ko-train.tsv",
                "train_qrels_tsv": "miracl-v1.0-ko/qrels/qrels.miracl-v1.0-ko-train.tsv",
                "train_topics_jsonl": "topics_ko_train.jsonl",
                "train_qrels_jsonl": "qrels_ko_train.jsonl",
                "dev_topics_tsv": "miracl-v1.0-ko/topics/topics.miracl-v1.0-ko-dev.tsv",
                "dev_qrels_tsv": "miracl-v1.0-ko/qrels/qrels.miracl-v1.0-ko-dev.tsv",
                "dev_topics_jsonl": "topics_ko_dev.jsonl",
                "dev_qrels_jsonl": "qrels_ko_dev.jsonl",
                "corpus_sample": "corpus_ko_real_sample.jsonl"
            },
            "note": "실제 MIRACL 데이터셋의 train/dev 쿼리, 관련성 판정, 그리고 코퍼스 샘플을 포함합니다."
        }
        
        with open('metadata_final.json', "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        print(f'📄 최종 메타데이터 업데이트: metadata_final.json')
        
    except Exception as e:
        print(f'❌ 오류 발생: {e}')

if __name__ == "__main__":
    download_train_data()