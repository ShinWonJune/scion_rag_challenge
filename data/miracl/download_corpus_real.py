#!/usr/bin/env python3
"""
실제 MIRACL 한국어 코퍼스 샘플 다운로드
"""

import gzip
import json
from huggingface_hub import hf_hub_download
from pathlib import Path

def download_korean_corpus_real():
    """실제 MIRACL 한국어 코퍼스의 첫 번째 파일을 다운로드합니다."""
    
    base_dir = Path(__file__).parent
    
    print('📚 실제 MIRACL 한국어 코퍼스 다운로드 중...')
    
    try:
        # 첫 번째 문서 파일 다운로드
        print('첫 번째 문서 파일 다운로드 중...')
        corpus_file = hf_hub_download(
            repo_id="miracl/miracl-corpus",
            filename="miracl-corpus-v1.0-ko/docs-0.jsonl.gz",
            local_dir=str(base_dir),
            repo_type="dataset"
        )
        
        print(f'✅ 다운로드 완료: {corpus_file}')
        
        # gzip 파일 압축 해제 및 처리
        print('압축 해제 및 샘플 추출 중...')
        corpus_sample = []
        
        with gzip.open(corpus_file, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 500:  # 처음 500개만 샘플로
                    break
                    
                doc = json.loads(line)
                corpus_sample.append(doc)
                
                if (i + 1) % 100 == 0:
                    print(f'   처리됨: {i + 1}개 문서')
        
        # 샘플을 JSONL로 저장
        sample_path = base_dir / "corpus_ko_real_sample.jsonl"
        with open(sample_path, 'w', encoding='utf-8') as f:
            for doc in corpus_sample:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f'\n✅ 실제 한국어 코퍼스 샘플 저장 완료!')
        print(f'📊 총 {len(corpus_sample)}개 문서')
        print(f'📁 파일: {sample_path}')
        
        # 샘플 확인
        print('\n🔍 실제 문서 샘플:')
        for i, doc in enumerate(corpus_sample[:3]):
            print(f'   문서 {i+1}: {doc["docid"]} - {doc["title"]}')
            print(f'      내용: {doc["text"][:100]}...')
            
        # 최종 메타데이터 업데이트
        metadata = {
            "dataset": "MIRACL Korean (Complete Sample)",
            "language": "ko",
            "source": "HuggingFace: miracl/miracl & miracl/miracl-corpus",
            "total_queries": 213,
            "total_qrels": 3057,
            "total_corpus_sample": len(corpus_sample),
            "download_date": "2025-09-27",
            "files": {
                "topics_tsv": "miracl-v1.0-ko/topics/topics.miracl-v1.0-ko-dev.tsv",
                "qrels_tsv": "miracl-v1.0-ko/qrels/qrels.miracl-v1.0-ko-dev.tsv",
                "topics_jsonl": "topics_ko_dev.jsonl",
                "qrels_jsonl": "qrels_ko_dev.jsonl",
                "corpus_sample": "corpus_ko_real_sample.jsonl"
            },
            "note": "실제 MIRACL 데이터셋의 쿼리, 관련성 판정, 그리고 코퍼스 샘플을 포함합니다."
        }
        
        with open(base_dir / "metadata_complete.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        print(f'📄 최종 메타데이터: metadata_complete.json')
        
        return len(corpus_sample)
        
    except Exception as e:
        print(f'❌ 오류 발생: {e}')
        return 0

if __name__ == "__main__":
    download_korean_corpus_real()