"""
설정 관리 모듈
- 기본 설정값들
- 환경 변수 관리
- 설정 검증
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

class Settings:
    """설정 관리 클래스"""
    
    def __init__(self):
        """설정 초기화"""
        self._load_default_settings()
        self._load_environment_settings()
        self._validate_settings()
    
    def _load_default_settings(self):
        """기본 설정 로드"""
        self.defaults = {
            # API 설정
            "gemini_model": "gemini-2.5-flash",
            "gemini_temperature": 0.2,
            "gemini_candidate_count": 1,
            
            # 검색 설정
            "target_documents_per_query": 50,
            "max_search_pages": 5,
            "page_size": 20,
            "max_search_terms": 15,
            "max_keywords": 5,
            
            # 파일 설정
            "output_directory": "./outputs",
            "csv_filename_prefix": "search_results",
            "json_filename_prefix": "search_meta_results",
            "jsonl_filename_prefix": "search_documents",
            
            # 로깅 설정
            "log_level": "INFO",
            "log_format": "%(asctime)s - %(levelname)s - %(message)s",
            
            # 품질 필터링 설정
            "min_title_length": 5,
            "min_abstract_length": 10,
            "require_abstract": False,
            "require_source": True,
        }
    
    def _load_environment_settings(self):
        """환경 변수에서 설정 로드"""
        self.env_settings = {}
        
        # API 키
        if os.getenv("GEMINI_API_KEY"):
            self.env_settings["gemini_api_key"] = os.getenv("GEMINI_API_KEY")
        
        # 출력 디렉토리
        if os.getenv("OUTPUT_DIRECTORY"):
            self.env_settings["output_directory"] = os.getenv("OUTPUT_DIRECTORY")
        
        # 로그 레벨
        if os.getenv("LOG_LEVEL"):
            self.env_settings["log_level"] = os.getenv("LOG_LEVEL")
        
        # 목표 문서 수
        if os.getenv("TARGET_DOCUMENTS"):
            try:
                self.env_settings["target_documents_per_query"] = int(os.getenv("TARGET_DOCUMENTS"))
            except ValueError:
                logging.warning(f"잘못된 TARGET_DOCUMENTS 값: {os.getenv('TARGET_DOCUMENTS')}")
    
    def _validate_settings(self):
        """설정 검증"""
        # 필수 설정 확인
        if not self.get("gemini_api_key"):
            logging.warning("GEMINI_API_KEY가 설정되지 않았습니다")
        
        # 출력 디렉토리 생성
        output_dir = self.get("output_directory")
        if output_dir:
            Path(output_dir).mkdir(exist_ok=True)
        
        # 숫자 설정 검증
        numeric_settings = [
            "target_documents_per_query",
            "max_search_pages", 
            "page_size",
            "max_search_terms",
            "max_keywords",
            "min_title_length",
            "min_abstract_length"
        ]
        
        for setting in numeric_settings:
            value = self.get(setting)
            if value is not None and (not isinstance(value, int) or value <= 0):
                logging.warning(f"잘못된 {setting} 값: {value}, 기본값 사용")
                self.env_settings[setting] = self.defaults[setting]
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정값 가져오기"""
        # 환경 변수 우선, 그 다음 기본값
        return self.env_settings.get(key, self.defaults.get(key, default))
    
    def set(self, key: str, value: Any):
        """설정값 설정"""
        self.env_settings[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """모든 설정값 반환"""
        all_settings = self.defaults.copy()
        all_settings.update(self.env_settings)
        return all_settings
    
    def get_api_config(self) -> Dict[str, Any]:
        """API 관련 설정 반환"""
        return {
            "model": self.get("gemini_model"),
            "temperature": self.get("gemini_temperature"),
            "candidate_count": self.get("gemini_candidate_count"),
            "api_key": self.get("gemini_api_key")
        }
    
    def get_search_config(self) -> Dict[str, Any]:
        """검색 관련 설정 반환"""
        return {
            "target_documents": self.get("target_documents_per_query"),
            "max_pages": self.get("max_search_pages"),
            "page_size": self.get("page_size"),
            "max_search_terms": self.get("max_search_terms"),
            "max_keywords": self.get("max_keywords")
        }
    
    def get_quality_config(self) -> Dict[str, Any]:
        """품질 필터링 설정 반환"""
        return {
            "min_title_length": self.get("min_title_length"),
            "min_abstract_length": self.get("min_abstract_length"),
            "require_abstract": self.get("require_abstract"),
            "require_source": self.get("require_source")
        }
    
    def get_file_config(self) -> Dict[str, Any]:
        """파일 관련 설정 반환"""
        return {
            "output_directory": self.get("output_directory"),
            "csv_prefix": self.get("csv_filename_prefix"),
            "json_prefix": self.get("json_filename_prefix"),
            "jsonl_prefix": self.get("jsonl_filename_prefix")
        }
    
    def print_settings(self):
        """설정 정보 출력"""
        print("🔧 현재 설정:")
        print("=" * 50)
        
        # API 설정
        api_config = self.get_api_config()
        print(f"📡 API 설정:")
        print(f"   모델: {api_config['model']}")
        print(f"   온도: {api_config['temperature']}")
        print(f"   API 키: {'설정됨' if api_config['api_key'] else '미설정'}")
        
        # 검색 설정
        search_config = self.get_search_config()
        print(f"\n🔍 검색 설정:")
        print(f"   목표 문서 수: {search_config['target_documents']}개")
        print(f"   최대 페이지: {search_config['max_pages']}페이지")
        print(f"   페이지 크기: {search_config['page_size']}개")
        
        # 파일 설정
        file_config = self.get_file_config()
        print(f"\n📁 파일 설정:")
        print(f"   출력 디렉토리: {file_config['output_directory']}")
        
        print("=" * 50)
