#!/usr/bin/env python3
"""
subquestion.csv의 doc_id에 해당하는 corpus 데이터를 추출하여 subcorpus.jsonl 생성
"""

import json
import csv
import os
from typing import Set, Dict


def load_doc_ids_from_subquestion(csv_path: str) -> Set[str]:
    """
    subquestion.csv에서 doc_id들을 추출
    
    Args:
        csv_path: subquestion.csv 파일 경로
        
    Returns:
        Set[str]: 고유한 doc_id들
    """
    doc_ids = set()
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_id = row.get('doc_id', '').strip()
                if doc_id:
                    doc_ids.add(str(doc_id))
                    
        print(f"✅ subquestion.csv에서 {len(doc_ids)}개의 고유한 doc_id 발견")
        return doc_ids
        
    except Exception as e:
        print(f"❌ subquestion.csv 로드 오류: {e}")
        return set()


def extract_matching_corpus(corpus_path: str, target_doc_ids: Set[str], output_path: str) -> int:
    """
    corpus.jsonl에서 target_doc_ids와 일치하는 문서들을 추출하여 subcorpus.jsonl 생성
    
    Args:
        corpus_path: corpus.jsonl 파일 경로
        target_doc_ids: 추출할 doc_id들
        output_path: 출력 subcorpus.jsonl 파일 경로
        
    Returns:
        int: 추출된 문서 수
    """
    found_docs = []
    found_doc_ids = set()
    
    try:
        # corpus.jsonl 읽기
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    doc = json.loads(line.strip())
                    doc_id = str(doc.get('doc_id', ''))
                    
                    if doc_id in target_doc_ids:
                        found_docs.append(doc)
                        found_doc_ids.add(doc_id)
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ 라인 {line_num} JSON 파싱 오류: {e}")
                    continue
        
        # subcorpus.jsonl 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in found_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"✅ subcorpus.jsonl 생성 완료: {output_path}")
        print(f"   - 추출된 문서: {len(found_docs)}개")
        print(f"   - 일치한 doc_id: {len(found_doc_ids)}개")
        
        # 찾지 못한 doc_id들 보고
        missing_doc_ids = target_doc_ids - found_doc_ids
        if missing_doc_ids:
            print(f"⚠️ corpus에서 찾지 못한 doc_id들 ({len(missing_doc_ids)}개):")
            for doc_id in sorted(missing_doc_ids)[:10]:  # 처음 10개만 표시
                print(f"     - {doc_id}")
            if len(missing_doc_ids) > 10:
                print(f"     - ... 및 {len(missing_doc_ids) - 10}개 더")
        
        return len(found_docs)
        
    except Exception as e:
        print(f"❌ corpus 추출 오류: {e}")
        return 0


def main():
    # 파일 경로 설정
    subquestion_csv_path = "/app/scifact/questions/subquestion.csv"
    corpus_path = "/app/scifact/data/corpus.jsonl"  # 원본 corpus 사용
    output_path = "/app/scifact/questions/subcorpus.jsonl"
    
    print(f"📖 subquestion.csv에서 doc_id 추출 중...")
    
    # 1. subquestion.csv에서 doc_id 추출
    target_doc_ids = load_doc_ids_from_subquestion(subquestion_csv_path)
    
    if not target_doc_ids:
        print("❌ 추출할 doc_id가 없습니다.")
        return 1
    
    print(f"📋 corpus.jsonl에서 해당 문서들 추출 중...")
    
    # 2. corpus에서 매칭되는 문서들 추출
    extracted_count = extract_matching_corpus(corpus_path, target_doc_ids, output_path)
    
    if extracted_count == 0:
        print("❌ 추출된 문서가 없습니다.")
        return 1
    
    print(f"\n🎉 subcorpus.jsonl 생성 완료!")
    print(f"   - 대상 doc_id: {len(target_doc_ids)}개")
    print(f"   - 추출된 문서: {extracted_count}개")
    print(f"   - 출력 파일: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())