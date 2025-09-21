"""
pubmedqa_complete.csv에 context 데이터를 추가하는 스크립트
"""

import pandas as pd
import json
from datasets import load_dataset

def add_context_to_complete_csv():
    """기존 pubmedqa_complete.csv에 context 정보를 추가합니다."""
    
    # 현재 CSV 파일 로드
    current_df = pd.read_csv('/app/pubmedqa/pubmedqa_complete.csv')
    print(f"현재 CSV 파일: {len(current_df)} 개 행")
    
    # pqa_labeled 데이터셋 로드 (원본과 같은 데이터)
    print("pqa_labeled 데이터셋 로딩 중...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    data = dataset['train']
    
    # pubid를 키로 사용하여 context 매핑
    pubid_to_context = {}
    for item in data:
        pubid = item['pubid']
        context = item['context']
        
        # context를 문자열로 변환 (contexts 리스트를 합침)
        if isinstance(context, dict) and 'contexts' in context:
            context_text = ' '.join(context['contexts'])
        else:
            context_text = str(context)
        
        pubid_to_context[pubid] = context_text
    
    # 기존 DataFrame에 context 컬럼 추가
    contexts = []
    for _, row in current_df.iterrows():
        pubid = row['pubid']
        context = pubid_to_context.get(pubid, '')
        contexts.append(context)
    
    # context 컬럼을 question 다음에 추가
    new_df = current_df.copy()
    
    # 컬럼 순서 재정렬: question, context, long_answer, final_decision, pubid
    new_df['context'] = contexts
    new_df = new_df[['question', 'context', 'long_answer', 'final_decision', 'pubid']]
    
    # 새로운 CSV 파일로 저장
    new_df.to_csv('/app/pubmedqa/pubmedqa_complete.csv', index=False)
    print(f"context가 추가된 새로운 CSV 파일이 저장되었습니다.")
    
    # 결과 확인
    print(f"\n업데이트된 컬럼들: {list(new_df.columns)}")
    print(f"첫 번째 행의 context 길이: {len(new_df.iloc[0]['context'])} 문자")
    print(f"첫 번째 행 context 미리보기:")
    print(new_df.iloc[0]['context'][:200] + "...")

if __name__ == "__main__":
    add_context_to_complete_csv()