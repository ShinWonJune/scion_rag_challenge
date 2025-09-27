#!/usr/bin/env python3
"""
SciFact claims_train.jsonl에서 전체 범위에 걸쳐 골고루 SUPPORT/CONTRADICT 라벨의 클레임을 추출하여
subquestion.csv와 questions.jsonl 형태로 변환하는 스크립트
"""

import json
import csv
import os
from typing import List, Dict, Tuple
import random


def extract_label_from_evidence(evidence: Dict) -> str:
    """Evidence에서 주요 라벨을 추출"""
    if not evidence:
        return "NOT_ENOUGH_INFO"
    
    labels = []
    for doc_id, doc_evidence in evidence.items():
        for sent_evidence in doc_evidence:
            labels.append(sent_evidence.get('label', ''))
    
    if 'CONTRADICT' in labels:
        return 'CONTRADICT'
    elif 'SUPPORT' in labels:
        return 'SUPPORT'
    else:
        return 'NOT_ENOUGH_INFO'


def load_and_categorize_claims(claims_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    claims_train.jsonl에서 모든 SUPPORT/CONTRADICT 라벨의 클레임들을 카테고리별로 분류
    
    Args:
        claims_path: claims_train.jsonl 파일 경로
        
    Returns:
        Tuple[List[Dict], List[Dict]]: (support_claims, contradict_claims)
    """
    support_claims = []
    contradict_claims = []
    
    try:
        with open(claims_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                claim = json.loads(line.strip())
                evidence = claim.get('evidence', {})
                label = extract_label_from_evidence(evidence)
                
                # 위치 정보 추가
                claim['_line_number'] = line_num
                
                if label == 'SUPPORT':
                    support_claims.append(claim)
                elif label == 'CONTRADICT':
                    contradict_claims.append(claim)
        
        print(f"✅ 전체 클레임 분석 완료:")
        print(f"   - SUPPORT 클레임: {len(support_claims)}개")
        print(f"   - CONTRADICT 클레임: {len(contradict_claims)}개")
        
        return support_claims, contradict_claims
        
    except Exception as e:
        print(f"❌ 클레임 로드 오류: {e}")
        return [], []


def sample_claims_evenly(claims: List[Dict], target_count: int, seed: int = 42) -> List[Dict]:
    """
    전체 범위에서 골고루 클레임을 샘플링
    
    Args:
        claims: 클레임 리스트
        target_count: 목표 샘플 수
        seed: 랜덤 시드
        
    Returns:
        List[Dict]: 샘플링된 클레임들
    """
    if len(claims) <= target_count:
        return claims
    
    # 랜덤 시드 설정 (재현 가능한 결과를 위해)
    random.seed(seed)
    
    # 전체 범위를 target_count개 구간으로 나누어 각 구간에서 하나씩 선택
    interval = len(claims) / target_count
    sampled_claims = []
    
    for i in range(target_count):
        start_idx = int(i * interval)
        end_idx = min(int((i + 1) * interval), len(claims))
        
        # 해당 구간에서 랜덤하게 하나 선택
        if start_idx < end_idx:
            selected_claim = random.choice(claims[start_idx:end_idx])
            sampled_claims.append(selected_claim)
    
    # 라인 번호순으로 정렬
    sampled_claims.sort(key=lambda x: x['_line_number'])
    
    return sampled_claims


def create_subquestion_csv(claims: List[Dict], output_path: str):
    """
    subquestion.csv 형태로 저장
    
    Args:
        claims: 클레임 리스트
        output_path: 출력 CSV 파일 경로
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # CSV 헤더
        writer.writerow(['question', 'label', 'doc_id'])
        
        support_count = 0
        contradict_count = 0
        
        for claim in claims:
            question = claim['claim']
            evidence = claim.get('evidence', {})
            label = extract_label_from_evidence(evidence)
            
            # 라벨 변환: CONTRADICT -> REFUTE
            if label == 'CONTRADICT':
                normalized_label = 'REFUTE'
                contradict_count += 1
            elif label == 'SUPPORT':
                normalized_label = 'SUPPORT'
                support_count += 1
            else:
                continue  # NOT_ENOUGH_INFO는 제외
            
            # cited_doc_ids에서 첫 번째 doc_id 사용
            cited_docs = claim.get('cited_doc_ids', [])
            doc_id = cited_docs[0] if cited_docs else ''
            
            writer.writerow([question, normalized_label, doc_id])
    
    print(f"✅ subquestion.csv 생성 완료: {output_path}")
    print(f"   - SUPPORT: {support_count}개")
    print(f"   - REFUTE: {contradict_count}개")
    print(f"   - 총합: {len(claims)}개")


def create_questions_jsonl(claims: List[Dict], output_path: str):
    """
    questions.jsonl 형태로 저장
    
    Args:
        claims: 클레임 리스트  
        output_path: 출력 JSONL 파일 경로
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, claim in enumerate(claims, 1):
            evidence = claim.get('evidence', {})
            label = extract_label_from_evidence(evidence)
            
            # NOT_ENOUGH_INFO는 제외
            if label == 'NOT_ENOUGH_INFO':
                continue
            
            question_data = {
                "id": f"row_{i:06d}",
                "question": claim['claim']
            }
            
            f.write(json.dumps(question_data, ensure_ascii=False) + '\n')
    
    print(f"✅ questions.jsonl 생성 완료: {output_path}")


def extract_subcorpus(subquestion_csv_path: str, corpus_path: str, output_path: str):
    """
    subquestion.csv의 doc_id에 해당하는 corpus 데이터를 추출하여 subcorpus.jsonl 생성
    """
    # subquestion.csv에서 doc_id 추출
    doc_ids = set()
    with open(subquestion_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row.get('doc_id', '').strip()
            if doc_id:
                doc_ids.add(str(doc_id))
    
    # corpus에서 매칭되는 문서들 추출
    found_docs = []
    found_doc_ids = set()
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                doc = json.loads(line.strip())
                doc_id = str(doc.get('doc_id', ''))
                
                if doc_id in doc_ids:
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
    print(f"   - 고유 doc_id: {len(found_doc_ids)}개")
    
    # 찾지 못한 doc_id들 보고
    missing_doc_ids = doc_ids - found_doc_ids
    if missing_doc_ids:
        print(f"⚠️ corpus에서 찾지 못한 doc_id들 ({len(missing_doc_ids)}개):")
        for doc_id in sorted(missing_doc_ids)[:5]:  # 처음 5개만 표시
            print(f"     - {doc_id}")
        if len(missing_doc_ids) > 5:
            print(f"     - ... 및 {len(missing_doc_ids) - 5}개 더")


def main():
    # 파일 경로 설정
    claims_train_path = "/app/scifact/data/claims_train.jsonl"
    corpus_path = "/app/scifact/data/corpus.jsonl"
    output_dir = "/app/scifact/questions"
    
    subquestion_csv_path = os.path.join(output_dir, "subquestion.csv")
    questions_jsonl_path = os.path.join(output_dir, "questions.jsonl")
    subcorpus_jsonl_path = os.path.join(output_dir, "subcorpus.jsonl")
    
    # 입력 파일 검증
    if not os.path.exists(claims_train_path):
        print(f"❌ 입력 파일이 존재하지 않습니다: {claims_train_path}")
        return 1
    
    try:
        print(f"📖 SciFact claims_train.jsonl 전체 분석 중...")
        
        # 1. 모든 클레임을 카테고리별로 분류
        support_claims, contradict_claims = load_and_categorize_claims(claims_train_path)
        
        if not support_claims or not contradict_claims:
            print("❌ 충분한 클레임을 찾을 수 없습니다.")
            return 1
        
        print(f"\n🎯 전체 범위에서 골고루 샘플링 중...")
        print(f"   - 목표: SUPPORT 25개, CONTRADICT 25개")
        
        # 2. 전체 범위에서 골고루 샘플링
        sampled_support = sample_claims_evenly(support_claims, 25, seed=42)
        sampled_contradict = sample_claims_evenly(contradict_claims, 25, seed=43)
        
        # 라인 번호순으로 통합 정렬
        all_sampled_claims = sorted(sampled_support + sampled_contradict, 
                                  key=lambda x: x['_line_number'])
        
        print(f"✅ 샘플링 완료:")
        print(f"   - SUPPORT: {len(sampled_support)}개")
        print(f"   - CONTRADICT: {len(sampled_contradict)}개")
        print(f"   - 라인 범위: {all_sampled_claims[0]['_line_number']} ~ {all_sampled_claims[-1]['_line_number']}")
        
        # 3. subquestion.csv 생성
        create_subquestion_csv(all_sampled_claims, subquestion_csv_path)
        
        # 4. questions.jsonl 생성
        create_questions_jsonl(all_sampled_claims, questions_jsonl_path)
        
        # 5. subcorpus.jsonl 생성
        print(f"\n📋 subcorpus.jsonl 생성 중...")
        extract_subcorpus(subquestion_csv_path, corpus_path, subcorpus_jsonl_path)
        
        print(f"\n🎉 모든 파일 생성 완료!")
        print(f"   - 출력 디렉토리: {output_dir}")
        print(f"   - subquestion.csv: 50개 클레임 (전체 범위 분산)")
        print(f"   - questions.jsonl: 50개 질문")
        print(f"   - subcorpus.jsonl: 해당 논문 abstract들")
        
        return 0
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    exit(main())