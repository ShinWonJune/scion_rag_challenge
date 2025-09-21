"""
PubMedQA 데이터에서 샘플 CSV 파일 생성
1. subquestion.csv: 첫 번째 컬럼에 질문만 포함 (50개 샘플)
2. pubmedqa_complete.csv: 질문, long_answer, final_decision 모두 포함
"""

import pandas as pd
import random
from datasets import load_dataset

def create_sample_csvs():
    """PubMedQA 데이터에서 CSV 파일들을 생성합니다."""
    
    # pqa_labeled 데이터셋 로드 (1000개 샘플, 가장 적절한 크기)
    print("pqa_labeled 데이터셋 로딩 중...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    data = dataset['train']
    
    print(f"총 {len(data)} 개의 샘플이 있습니다.")
    
    # 50개 샘플 랜덤 선택
    random.seed(42)  # 재현 가능한 결과를 위해 시드 설정
    sample_indices = random.sample(range(len(data)), min(50, len(data)))
    
    # 선택된 샘플 데이터 추출
    sampled_data = [data[i] for i in sample_indices]
    
    # 1. subquestion.csv 생성 (첫 번째 컬럼에 질문만)
    questions_df = pd.DataFrame({
        'question': [item['question'] for item in sampled_data]
    })
    
    questions_df.to_csv('/app/pubmedqa/subquestion.csv', index=False)
    print("subquestion.csv 파일이 생성되었습니다. (50개 질문)")
    
    # 2. pubmedqa_complete.csv 생성 (모든 정보 포함)
    complete_df = pd.DataFrame({
        'question': [item['question'] for item in sampled_data],
        'long_answer': [item['long_answer'] for item in sampled_data],
        'final_decision': [item['final_decision'] for item in sampled_data],
        'pubid': [item['pubid'] for item in sampled_data]
    })
    
    complete_df.to_csv('/app/pubmedqa/pubmedqa_complete.csv', index=False)
    print("pubmedqa_complete.csv 파일이 생성되었습니다. (질문, 답변, 결정 포함)")
    
    # 3. 평가용 데이터 생성 (질문과 정답만)
    evaluation_df = pd.DataFrame({
        'question': [item['question'] for item in sampled_data],
        'ground_truth_answer': [item['long_answer'] for item in sampled_data],
        'ground_truth_decision': [item['final_decision'] for item in sampled_data]
    })
    
    evaluation_df.to_csv('/app/pubmedqa/pubmedqa_evaluation.csv', index=False)
    print("pubmedqa_evaluation.csv 파일이 생성되었습니다. (평가용)")
    
    # 생성된 파일들 정보 출력
    print(f"\n생성된 파일들:")
    print(f"1. subquestion.csv: {len(questions_df)} 개 질문")
    print(f"2. pubmedqa_complete.csv: {len(complete_df)} 개 완전한 데이터")
    print(f"3. pubmedqa_evaluation.csv: {len(evaluation_df)} 개 평가용 데이터")
    
    # 샘플 데이터 출력
    print(f"\n첫 번째 질문 예시:")
    print(f"질문: {sampled_data[0]['question']}")
    print(f"답변: {sampled_data[0]['long_answer'][:200]}...")
    print(f"결정: {sampled_data[0]['final_decision']}")

if __name__ == "__main__":
    create_sample_csvs()