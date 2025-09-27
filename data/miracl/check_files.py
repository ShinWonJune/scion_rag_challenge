#!/usr/bin/env python3
"""
HuggingFace Hub를 통한 MIRACL 파일 확인
"""

from huggingface_hub import list_repo_files, hf_hub_download
import json

def check_miracl_files():
    """MIRACL 관련 파일들을 확인합니다."""
    
    print('📁 MIRACL 코퍼스 레포지토리 파일 목록 확인 중...')
    try:
        files = list_repo_files('miracl/miracl-corpus', repo_type='dataset')
        ko_files = [f for f in files if 'ko' in f]
        print(f'한국어 관련 파일들:')
        for f in ko_files[:10]:  # 처음 10개만 표시
            print(f'   - {f}')
        
        if len(ko_files) > 10:
            print(f'   ... 총 {len(ko_files)}개 파일')
            
        return ko_files
        
    except Exception as e:
        print(f'❌ 레포지토리 접근 실패: {e}')
        return []

if __name__ == "__main__":
    check_miracl_files()