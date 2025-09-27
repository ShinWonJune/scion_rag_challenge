#!/usr/bin/env python3
"""
MIRACL 한국어 코퍼스 샘플 다운로드 스크립트
"""

import json
from datasets import load_dataset
from pathlib import Path

def download_miracl_korean_corpus_sample():
    """MIRACL 한국어 코퍼스의 샘플을 다운로드합니다."""
    
    base_dir = Path(__file__).parent
    
    print("📚 MIRACL 한국어 코퍼스 샘플 다운로드 중...")
    
    try:
        # datasets 라이브러리의 새로운 방법 시도
        print("HuggingFace에서 한국어 코퍼스 로딩 중...")
        
        # 한국어 코퍼스 로드 (스트리밍 모드로 메모리 절약)
        dataset = load_dataset("miracl/miracl-corpus", "ko", split="train", streaming=True)
        
        # 첫 1000개 문서만 샘플로 저장
        corpus_sample = []
        print("샘플 문서 수집 중 (최대 1000개)...")
        
        for i, doc in enumerate(dataset):
            if i >= 1000:
                break
            
            corpus_sample.append({
                'docid': doc['docid'],
                'title': doc['title'], 
                'text': doc['text']
            })
            
            if (i + 1) % 100 == 0:
                print(f"   처리됨: {i + 1}개 문서")
        
        # JSONL로 저장
        corpus_path = base_dir / "corpus_ko_sample.jsonl"
        with open(corpus_path, 'w', encoding='utf-8') as f:
            for doc in corpus_sample:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"\n✅ 한국어 코퍼스 샘플 다운로드 완료!")
        print(f"📊 총 {len(corpus_sample)}개 문서 저장")
        print(f"📁 파일: {corpus_path}")
        
        # 샘플 확인
        print("\n🔍 샘플 문서 확인:")
        for i, doc in enumerate(corpus_sample[:3]):
            print(f"   문서 {i+1}: {doc['docid']} - {doc['title']}")
            print(f"      내용: {doc['text'][:100]}...")
            
        return len(corpus_sample)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("💡 대신 더 간단한 방법으로 시도해보겠습니다...")
        return 0

if __name__ == "__main__":
    download_miracl_korean_corpus_sample()