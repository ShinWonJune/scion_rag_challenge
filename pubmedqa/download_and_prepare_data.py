"""
PubMedQA 데이터셋 다운로드 및 준비 스크립트
HuggingFace datasets에서 qiaojin/PubMedQA 데이터셋을 다운로드합니다.
"""

import os
import pandas as pd
from datasets import load_dataset
import random
import json

def download_pubmedqa_dataset():
    """PubMedQA 데이터셋을 다운로드하고 구조를 확인합니다."""
    print("PubMedQA 데이터셋을 다운로드하는 중...")
    
    # 사용 가능한 설정들 확인
    configs = ['pqa_artificial', 'pqa_labeled', 'pqa_unlabeled']
    datasets = {}
    
    for config in configs:
        try:
            print(f"\n{config} 설정 로딩 중...")
            dataset = load_dataset("qiaojin/PubMedQA", config)
            datasets[config] = dataset
            
            print(f"{config} 데이터셋 분할: {list(dataset.keys())}")
            
            # 각 분할의 크기 확인
            for split in dataset.keys():
                print(f"  {split}: {len(dataset[split])} 샘플")
                
            # 첫 번째 샘플 구조 확인
            for split in dataset.keys():
                if len(dataset[split]) > 0:
                    sample = dataset[split][0]
                    print(f"\n{config} - {split} 샘플 데이터 구조:")
                    for key, value in sample.items():
                        print(f"    {key}: {type(value)} - {str(value)[:100]}...")
                    break
                    
        except Exception as e:
            print(f"{config} 데이터셋 다운로드 중 오류 발생: {e}")
            continue
    
    return datasets

def save_dataset_info(datasets):
    """데이터셋 정보를 JSON 파일로 저장합니다."""
    info = {}
    
    for config, dataset in datasets.items():
        info[config] = {}
        
        for split in dataset.keys():
            info[config][split] = {
                'size': len(dataset[split]),
                'features': list(dataset[split].features.keys())
            }
            
            # 샘플 데이터 저장
            if len(dataset[split]) > 0:
                info[config][split]['sample'] = dataset[split][0]
    
    with open('/app/pubmedqa/dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print("데이터셋 정보가 dataset_info.json에 저장되었습니다.")

if __name__ == "__main__":
    # 작업 디렉토리를 pubmedqa로 변경
    os.chdir('/app/pubmedqa')
    
    # 데이터셋 다운로드
    datasets = download_pubmedqa_dataset()
    
    if datasets:
        # 데이터셋 정보 저장
        save_dataset_info(datasets)
        print("PubMedQA 데이터셋 다운로드가 완료되었습니다.")
    else:
        print("데이터셋 다운로드에 실패했습니다.")