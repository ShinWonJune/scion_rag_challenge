#!/usr/bin/env python3
"""ScienceON API 토큰 강제 갱신 스크립트"""

from pathlib import Path
from scienceon_api_example import ScienceONAPIClient
import logging

logging.basicConfig(level=logging.INFO)

def main():
    credentials_path = Path(__file__).parent / 'configs/scienceon_api_credentials.json'
    
    print("토큰 갱신 시작...")
    client = ScienceONAPIClient(credentials_path=credentials_path)
    
    # 토큰 갱신을 강제하기 위해 만료 시간을 과거로 설정
    client.credential_manager.credentials['access_token_expire'] = '2020-01-01 00:00:00'
    
    # get_access_token을 호출하면 자동으로 갱신됨
    token = client.credential_manager.get_access_token(client.session)
    
    print(f"✅ 토큰 갱신 완료!")
    print(f"새 Access Token (처음 20자): {token[:20]}...")
    print(f"만료 시간: {client.credential_manager.credentials.get('access_token_expire')}")
    
    client.close_session()

if __name__ == "__main__":
    main()
