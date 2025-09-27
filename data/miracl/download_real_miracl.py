#!/usr/bin/env python3
"""
실제 MIRACL 한국어 데이터셋 다운로드 스크립트
"""

import os
import json
import requests
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_miracl_korean_real():
    """실제 MIRACL 한국어 데이터셋을 HuggingFace에서 다운로드합니다."""
    
    # 현재 디렉토리 설정
    base_dir = Path(__file__).parent
    
    print("🌍 실제 MIRACL 한국어 데이터셋 다운로드를 시작합니다...")
    
    try:
        # 1. Topics (쿼리) 다운로드
        print("1. 한국어 Topics (쿼리) 다운로드 중...")
        
        # Dev topics
        topics_dev_path = hf_hub_download(
            repo_id="miracl/miracl",
            filename="miracl-v1.0-ko/topics/topics.miracl-v1.0-ko-dev.tsv",
            local_dir=str(base_dir),
            repo_type="dataset"
        )
        print(f"   ✅ Dev topics 다운로드 완료: {topics_dev_path}")
        
        # 2. QRELs (관련성 판정) 다운로드
        print("2. 한국어 QRELs (관련성 판정) 다운로드 중...")
        
        # Dev qrels
        qrels_dev_path = hf_hub_download(
            repo_id="miracl/miracl", 
            filename="miracl-v1.0-ko/qrels/qrels.miracl-v1.0-ko-dev.tsv",
            local_dir=str(base_dir),
            repo_type="dataset"
        )
        print(f"   ✅ Dev qrels 다운로드 완료: {qrels_dev_path}")
        
        # 3. 다운로드된 파일들 확인 및 통계
        print("3. 다운로드된 데이터 분석 중...")
        
        # Topics 분석
        topics_count = 0
        topics_data = []
        with open(topics_dev_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    topics_data.append({
                        'query_id': parts[0],
                        'query': parts[1]
                    })
                    topics_count += 1
        
        print(f"   📊 총 {topics_count}개의 한국어 쿼리")
        
        # QRELs 분석  
        qrels_count = 0
        qrels_data = []
        with open(qrels_dev_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    qrels_data.append({
                        'query_id': parts[0],
                        'corpus_id': parts[2], 
                        'score': int(parts[3])
                    })
                    qrels_count += 1
        
        print(f"   📊 총 {qrels_count}개의 관련성 판정")
        
        # 4. JSONL 형식으로도 저장
        print("4. JSONL 형식으로 변환 중...")
        
        # Topics를 JSONL로 저장
        topics_jsonl = base_dir / "topics_ko_dev.jsonl"
        with open(topics_jsonl, 'w', encoding='utf-8') as f:
            for topic in topics_data:
                f.write(json.dumps(topic, ensure_ascii=False) + '\n')
        
        # QRELs를 JSONL로 저장
        qrels_jsonl = base_dir / "qrels_ko_dev.jsonl" 
        with open(qrels_jsonl, 'w', encoding='utf-8') as f:
            for qrel in qrels_data:
                f.write(json.dumps(qrel, ensure_ascii=False) + '\n')
        
        # 5. 샘플 확인
        print("5. 샘플 데이터 확인:")
        print("   🔍 첫 5개 쿼리:")
        for i, topic in enumerate(topics_data[:5]):
            print(f"      {topic['query_id']}: {topic['query']}")
        
        print("   🎯 첫 5개 관련성 판정:")
        for i, qrel in enumerate(qrels_data[:5]):
            print(f"      쿼리 {qrel['query_id']} -> 문서 {qrel['corpus_id']} (점수: {qrel['score']})")
        
        # 6. 메타데이터 저장
        metadata = {
            "dataset": "MIRACL Korean (Real)",
            "language": "ko",
            "source": "HuggingFace: miracl/miracl",
            "total_queries": topics_count,
            "total_qrels": qrels_count,
            "download_date": "2025-09-27",
            "files": {
                "topics_tsv": str(Path(topics_dev_path).relative_to(base_dir)),
                "qrels_tsv": str(Path(qrels_dev_path).relative_to(base_dir)),
                "topics_jsonl": "topics_ko_dev.jsonl",
                "qrels_jsonl": "qrels_ko_dev.jsonl"
            },
            "note": "이는 실제 MIRACL 데이터셋입니다. 코퍼스는 별도로 다운로드해야 합니다."
        }
        
        with open(base_dir / "metadata_real.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 실제 MIRACL 한국어 데이터셋 다운로드 완료!")
        print(f"📁 저장 위치: {base_dir}")
        print(f"❓ 총 쿼리: {topics_count}개")
        print(f"🎯 총 QRELs: {qrels_count}개")
        print(f"📄 메타데이터: metadata_real.json")
        
        print("\n⚠️  참고: 이는 쿼리와 관련성 판정만 포함합니다.")
        print("   실제 문서 코퍼스는 매우 크므로(GB 단위) 별도로 다운로드해야 합니다.")
        print("   코퍼스 다운로드: https://huggingface.co/datasets/miracl/miracl-corpus")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise

if __name__ == "__main__":
    download_miracl_korean_real()