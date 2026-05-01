#!/usr/bin/env python3
"""
SciFact claims_train.jsonl에서 SUPPORT/CONTRADICT 라벨의 클레임을 추출하여
subquestion.csv와 questions.jsonl 형태로 변환하는 스크립트
"""

import json
import csv
import os
from typing import List, Dict


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


def load_and_filter_claims(claims_path: str, support_count: int = 25, contradict_count: int = 25) -> List[Dict]:
    """
    claims_train.jsonl에서 SUPPORT/CONTRADICT 라벨의 클레임들을 추출
    
    Args:
        claims_path: claims_train.jsonl 파일 경로
        support_count: 추출할 SUPPORT 클레임 수
        contradict_count: 추출할 CONTRADICT 클레임 수
    
    Returns:
        List[Dict]: 필터링된 클레임 리스트
    """
    support_claims = []
    contradict_claims = []
    
    try:
        with open(claims_path, 'r', encoding='utf-8') as f:
            for line in f:
                claim = json.loads(line.strip())
                evidence = claim.get('evidence', {})
                label = extract_label_from_evidence(evidence)
                
                if label == 'SUPPORT' and len(support_claims) < support_count:
                    support_claims.append(claim)
                elif label == 'CONTRADICT' and len(contradict_claims) < contradict_count:
                    contradict_claims.append(claim)
                
                # 필요한 수만큼 수집되면 중단
                if len(support_claims) >= support_count and len(contradict_claims) >= contradict_count:
                    break
        
        print(f"✅ 추출된 클레임:")
        print(f"   - SUPPORT: {len(support_claims)}개")
        print(f"   - CONTRADICT: {len(contradict_claims)}개")
        
        return support_claims + contradict_claims
        
    except Exception as e:
        print(f"❌ 클레임 로드 오류: {e}")
        return []


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
        
        for claim in claims:
            question = claim['claim']
            evidence = claim.get('evidence', {})
            label = extract_label_from_evidence(evidence)
            
            # 라벨 변환: CONTRADICT -> REFUTE
            if label == 'CONTRADICT':
                normalized_label = 'REFUTE'
            elif label == 'SUPPORT':
                normalized_label = 'SUPPORT'
            else:
                continue  # NOT_ENOUGH_INFO는 제외
            
            # cited_doc_ids에서 첫 번째 doc_id 사용
            cited_docs = claim.get('cited_doc_ids', [])
            doc_id = cited_docs[0] if cited_docs else ''
            
            writer.writerow([question, normalized_label, doc_id])
    
    print(f"✅ subquestion.csv 생성 완료: {output_path}")


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


def main():
    # 파일 경로 설정
    claims_train_path = "/app/scifact/original_data/claims_train.jsonl"
    output_dir = "/app/scifact/data/questions_from_original"
    
    subquestion_csv_path = os.path.join(output_dir, "subquestion.csv")
    questions_jsonl_path = os.path.join(output_dir, "questions.jsonl")
    
    # 입력 파일 검증
    if not os.path.exists(claims_train_path):
        print(f"❌ 입력 파일이 존재하지 않습니다: {claims_train_path}")
        return 1
    
    try:
        print(f"📖 SciFact claims_train.jsonl에서 클레임 추출 중...")
        print(f"   - 대상: SUPPORT 25개, CONTRADICT 25개")
        
        # 1. 클레임 추출
        filtered_claims = load_and_filter_claims(claims_train_path, 25, 25)
        
        if not filtered_claims:
            print("❌ 추출된 클레임이 없습니다.")
            return 1
        
        # 2. subquestion.csv 생성
        create_subquestion_csv(filtered_claims, subquestion_csv_path)
        
        # 3. questions.jsonl 생성
        create_questions_jsonl(filtered_claims, questions_jsonl_path)
        
        print(f"\n🎉 데이터 추출 완료!")
        print(f"   - 출력 디렉토리: {output_dir}")
        print(f"   - subquestion.csv: {subquestion_csv_path}")
        print(f"   - questions.jsonl: {questions_jsonl_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    exit(main())