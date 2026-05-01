#!/usr/bin/env python3
"""
원본 SciFact 데이터를 우리 형식으로 변환하는 스크립트

사용법:
    python convert_original_scifact.py --input_dir /app/scifact/original_data --output_dir /app/scifact/converted --split train
"""

import os
import json
import csv
import argparse
from typing import Dict, List, Tuple
from pathlib import Path


def load_corpus(corpus_path: str) -> Dict[str, Dict]:
    """
    SciFact corpus.jsonl을 로드
    
    Args:
        corpus_path: corpus.jsonl 파일 경로
        
    Returns:
        Dict[str, Dict]: {doc_id: document_data}
    """
    corpus = {}
    
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                doc_id = str(doc['doc_id'])
                corpus[doc_id] = doc
                
        print(f"✅ Corpus 로드 완료: {len(corpus)}개 문서")
        return corpus
        
    except Exception as e:
        print(f"❌ Corpus 로드 오류: {e}")
        return {}


def load_claims(claims_path: str) -> List[Dict]:
    """
    SciFact claims.jsonl을 로드
    
    Args:
        claims_path: claims.jsonl 파일 경로
        
    Returns:
        List[Dict]: claims 리스트
    """
    claims = []
    
    try:
        with open(claims_path, 'r', encoding='utf-8') as f:
            for line in f:
                claim = json.loads(line.strip())
                claims.append(claim)
                
        print(f"✅ Claims 로드 완료: {len(claims)}개 클레임")
        return claims
        
    except Exception as e:
        print(f"❌ Claims 로드 오류: {e}")
        return []


def extract_label_from_evidence(evidence: Dict) -> str:
    """
    Evidence에서 주요 라벨을 추출
    
    Args:
        evidence: evidence 딕셔너리
        
    Returns:
        str: SUPPORT, CONTRADICT, 또는 NOT_ENOUGH_INFO
    """
    if not evidence:
        return "NOT_ENOUGH_INFO"
    
    # 모든 evidence의 라벨을 수집
    labels = []
    for doc_id, doc_evidence in evidence.items():
        for sent_evidence in doc_evidence:
            labels.append(sent_evidence.get('label', ''))
    
    # 라벨 우선순위: CONTRADICT > SUPPORT > NOT_ENOUGH_INFO
    if 'CONTRADICT' in labels:
        return 'CONTRADICT'
    elif 'SUPPORT' in labels:
        return 'SUPPORT'
    else:
        return 'NOT_ENOUGH_INFO'


def convert_to_csv_format(claims: List[Dict], output_path: str, split_name: str):
    """
    Claims를 CSV 형식으로 변환
    
    Args:
        claims: claims 리스트
        output_path: 출력 CSV 파일 경로
        split_name: 데이터 분할 이름 (train/dev/test)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # CSV 헤더
        writer.writerow(['question', 'label', 'doc_id', 'claim_id', 'evidence_docs'])
        
        support_count = 0
        contradict_count = 0
        nei_count = 0
        
        for claim in claims:
            claim_id = claim['id']
            question = claim['claim']
            evidence = claim.get('evidence', {})
            cited_docs = claim.get('cited_doc_ids', [])
            
            # 라벨 추출
            label = extract_label_from_evidence(evidence)
            
            # 라벨 정규화 (우리 형식에 맞게)
            if label == 'CONTRADICT':
                normalized_label = 'REFUTE'
                contradict_count += 1
            elif label == 'SUPPORT':
                normalized_label = 'SUPPORT' 
                support_count += 1
            else:
                normalized_label = 'NOT_ENOUGH_INFO'
                nei_count += 1
            
            # 주요 cited_doc (첫 번째 것 사용)
            main_doc_id = cited_docs[0] if cited_docs else ''
            
            # evidence에 포함된 문서들
            evidence_doc_ids = '|'.join(evidence.keys()) if evidence else ''
            
            writer.writerow([
                question,
                normalized_label,
                main_doc_id,
                claim_id,
                evidence_doc_ids
            ])
    
    print(f"✅ {split_name} CSV 변환 완료: {output_path}")
    print(f"   - SUPPORT: {support_count}개")
    print(f"   - REFUTE: {contradict_count}개") 
    print(f"   - NOT_ENOUGH_INFO: {nei_count}개")
    print(f"   - 총합: {len(claims)}개")


def extract_abstracts_for_claims(claims: List[Dict], corpus: Dict[str, Dict], output_path: str):
    """
    Claims에서 사용된 문서들의 abstract를 추출
    
    Args:
        claims: claims 리스트
        corpus: corpus 딕셔너리
        output_path: 출력 JSON 파일 경로
    """
    used_doc_ids = set()
    
    # 모든 claims에서 사용된 문서 ID 수집
    for claim in claims:
        # cited_doc_ids에서 수집
        cited_docs = claim.get('cited_doc_ids', [])
        for doc_id in cited_docs:
            used_doc_ids.add(str(doc_id))
            
        # evidence에서도 수집
        evidence = claim.get('evidence', {})
        for doc_id in evidence.keys():
            used_doc_ids.add(str(doc_id))
    
    # 해당하는 abstract들 추출
    abstracts = {}
    found_count = 0
    
    for doc_id in used_doc_ids:
        if doc_id in corpus:
            doc = corpus[doc_id]
            abstracts[doc_id] = {
                'doc_id': doc_id,
                'title': doc.get('title', ''),
                'abstract': doc.get('abstract', []),
                'structured': doc.get('structured', False)
            }
            found_count += 1
        else:
            print(f"⚠️ doc_id {doc_id}를 corpus에서 찾을 수 없음")
    
    # JSON으로 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(abstracts, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Abstract 추출 완료: {output_path}")
    print(f"   - 요청된 문서: {len(used_doc_ids)}개")
    print(f"   - 성공적으로 추출: {found_count}개")


def main():
    parser = argparse.ArgumentParser(
        description="원본 SciFact 데이터를 우리 형식으로 변환"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="원본 SciFact 데이터 디렉토리"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="변환된 데이터를 저장할 디렉토리"
    )
    parser.add_argument(
        "--split",
        choices=['train', 'dev', 'test', 'all'],
        default='all',
        help="변환할 데이터 분할 (기본값: all)"
    )
    
    args = parser.parse_args()
    
    # 입력 디렉토리 검증
    if not os.path.exists(args.input_dir):
        print(f"❌ 입력 디렉토리가 존재하지 않습니다: {args.input_dir}")
        return 1
    
    try:
        # Corpus 로드
        corpus_path = os.path.join(args.input_dir, "corpus.jsonl")
        corpus = load_corpus(corpus_path)
        
        if not corpus:
            print("❌ Corpus를 로드할 수 없습니다.")
            return 1
        
        # 변환할 분할들 결정
        splits_to_process = []
        if args.split == 'all':
            splits_to_process = ['train', 'dev', 'test']
        else:
            splits_to_process = [args.split]
        
        # 각 분할 처리
        for split in splits_to_process:
            claims_path = os.path.join(args.input_dir, f"claims_{split}.jsonl")
            
            if not os.path.exists(claims_path):
                print(f"⚠️ {split} 분할 파일이 존재하지 않습니다: {claims_path}")
                continue
            
            print(f"\n{'='*50}")
            print(f"처리 중: {split.upper()} 분할")
            print(f"{'='*50}")
            
            # Claims 로드
            claims = load_claims(claims_path)
            
            if not claims:
                print(f"❌ {split} claims를 로드할 수 없습니다.")
                continue
            
            # CSV 변환
            csv_output_path = os.path.join(args.output_dir, f"scifact_{split}.csv")
            convert_to_csv_format(claims, csv_output_path, split)
            
            # Abstract 추출
            abstracts_output_path = os.path.join(args.output_dir, f"abstracts_{split}.json")
            extract_abstracts_for_claims(claims, corpus, abstracts_output_path)
        
        print(f"\n🎉 변환 완료!")
        print(f"   - 입력 디렉토리: {args.input_dir}")
        print(f"   - 출력 디렉토리: {args.output_dir}")
        print(f"   - 처리된 분할: {', '.join(splits_to_process)}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    exit(main())