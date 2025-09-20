"""
검색 메타데이터 시스템 - 메인 클래스
- 모든 컴포넌트 통합 관리
- 사용자 친화적 인터페이스 제공
- 설정 및 초기화 관리
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from core.keyword_extractor import KeywordExtractor
from core.document_searcher import DocumentSearcher
from processors.single_query_processor import SingleQueryProcessor
from processors.batch_query_processor import BatchQueryProcessor
from utils.file_manager import FileManager
from utils.result_converter import ResultConverter
from configs.settings import Settings

# ScienceON API 클라이언트 import
from scienceon_api_example import ScienceONAPIClient

class SearchMetaSystem:
    """검색 메타데이터 시스템 - 통합 관리"""
    
    def __init__(self, gemini_api_key: str = None, use_vllm: bool = False, 
             vllm_base_url: str = "http://localhost:8000/v1", 
             vllm_model: str = "openai/gpt-oss-120B",
             scienceon_credentials_path: str = "./configs/scienceon_api_credentials.json"):
        """
        검색 메타데이터 시스템 초기화
        
        Args:
            gemini_api_key: Gemini API 키 (use_vllm=False일 때 필수)
            use_vllm: vLLM 사용 여부
            vllm_base_url: vLLM 서버 URL
            vllm_model: vLLM 모델명
            scienceon_credentials_path: ScienceON API 자격증명 파일 경로
        """
        # 설정 로드
        self.settings = Settings()
        self.use_vllm = use_vllm
        
        if use_vllm:
            self.settings.set("use_vllm", True)
            self.settings.set("vllm_base_url", vllm_base_url)
            self.settings.set("vllm_model", vllm_model)
        else:
            self.settings.set("gemini_api_key", gemini_api_key)
        
        # 로깅 설정
        self._setup_logging()
        
        # ScienceON API 클라이언트 초기화
        self.scienceon_client = ScienceONAPIClient(Path(scienceon_credentials_path))
        
        # 핵심 컴포넌트 초기화
        self._initialize_components()
        
        logging.info("SearchMetaSystem 초기화 완료")
    
    def _setup_logging(self):
        """로깅 설정"""
        log_level = self.settings.get("log_level", "INFO")
        log_format = self.settings.get("log_format")
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format
        )
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        # 파일 관리자
        self.file_manager = FileManager(
            output_dir=self.settings.get("output_directory")
        )
        
        # 결과 변환기
        self.result_converter = ResultConverter(self.file_manager)
        
        # 키워드 추출기
        api_config = self.settings.get_api_config()
        if self.use_vllm:
            from core.vllm_keyword_extractor import VLLMKeywordExtractor
            self.keyword_extractor = VLLMKeywordExtractor(
                self.settings.get("vllm_base_url"),
                self.settings.get("vllm_model")
            )
        else:
            self.keyword_extractor = KeywordExtractor(self.settings.get("gemini_api_key"))
        
        # 문서 검색기
        self.document_searcher = DocumentSearcher(self.scienceon_client)
        search_config = self.settings.get_search_config()
        self.document_searcher.set_target_documents(search_config["target_documents"])
        self.document_searcher.set_max_pages(search_config["max_pages"])
        
        # 단일 쿼리 처리기
        self.single_processor = SingleQueryProcessor(
            self.keyword_extractor,
            self.document_searcher
        )
        
        # 배치 쿼리 처리기
        self.batch_processor = BatchQueryProcessor(
            self.single_processor,
            self.file_manager,
            self.result_converter
        )
    
    def process_single_query(self, query: str, target_documents: int = None) -> Dict[str, Any]:
        """
        단일 쿼리 처리
        
        Args:
            query: 처리할 질문
            target_documents: 목표 문서 수 (None이면 설정값 사용)
            
        Returns:
            처리 결과
        """
        if target_documents is None:
            target_documents = self.settings.get("target_documents_per_query")
        
        return self.single_processor.process_query(query, target_documents)
    
    def process_batch_from_csv(self, csv_path: str, max_queries: Optional[int] = None, 
                              target_documents: int = None) -> Dict[str, Any]:
        """
        CSV 파일에서 배치 처리
        
        Args:
            csv_path: CSV 파일 경로
            max_queries: 최대 처리할 질문 수
            target_documents: 질문당 목표 문서 수
            
        Returns:
            배치 처리 결과
        """
        if target_documents is None:
            target_documents = self.settings.get("target_documents_per_query")
        
        return self.batch_processor.process_queries_from_csv(
            csv_path, target_documents, max_queries
        )
    
    def process_batch_queries(self, queries: List[str], target_documents: int = None) -> Dict[str, Any]:
        """
        질문 리스트에 대한 배치 처리
        
        Args:
            queries: 처리할 질문 리스트
            target_documents: 질문당 목표 문서 수
            
        Returns:
            배치 처리 결과
        """
        if target_documents is None:
            target_documents = self.settings.get("target_documents_per_query")
        
        return self.batch_processor.process_queries(queries, target_documents)
    
    def convert_latest_results(self) -> Dict[str, str]:
        """
        가장 최근 결과를 CSV/JSONL로 변환
        
        Returns:
            변환된 파일 경로들
        """
        try:
            csv_file = self.result_converter.convert_latest_json_to_csv()
            jsonl_file = self.result_converter.convert_latest_json_to_jsonl()
            
            return {
                "csv": csv_file,
                "jsonl": jsonl_file
            }
        except Exception as e:
            logging.error(f"결과 변환 실패: {e}")
            raise
    
    def update_settings(self, **kwargs):
        """설정 업데이트"""
        for key, value in kwargs.items():
            self.settings.set(key, value)
        
        # 관련 컴포넌트 재초기화
        self._update_components()
    
    def _update_components(self):
        """설정 변경에 따른 컴포넌트 업데이트"""
        # 문서 검색기 설정 업데이트
        search_config = self.settings.get_search_config()
        self.document_searcher.set_target_documents(search_config["target_documents"])
        self.document_searcher.set_max_pages(search_config["max_pages"])
        
        # 파일 관리자 설정 업데이트
        self.file_manager = FileManager(
            output_dir=self.settings.get("output_directory")
        )
        self.result_converter = ResultConverter(self.file_manager)
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 반환"""
        return {
            "settings": self.settings.get_all(),
            "components": {
                "keyword_extractor": "초기화됨",
                "document_searcher": "초기화됨",
                "single_processor": "초기화됨",
                "batch_processor": "초기화됨",
                "file_manager": "초기화됨",
                "result_converter": "초기화됨"
            }
        }
    
    def print_system_info(self):
        """시스템 정보 출력"""
        print("🔧 검색 메타데이터 시스템 정보")
        print("=" * 60)
        self.settings.print_settings()
        
        print(f"\n📊 컴포넌트 상태:")
        print(f"   키워드 추출기: ✅ 초기화됨")
        print(f"   문서 검색기: ✅ 초기화됨")
        print(f"   단일 처리기: ✅ 초기화됨")
        print(f"   배치 처리기: ✅ 초기화됨")
        print(f"   파일 관리자: ✅ 초기화됨")
        print(f"   결과 변환기: ✅ 초기화됨")
        print("=" * 60)
    
    def cleanup(self):
        """리소스 정리"""
        logging.info("SearchMetaSystem 리소스 정리 완료")
