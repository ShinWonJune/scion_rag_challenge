"""
결과 변환기 모듈
- JSON 결과를 CSV/JSONL 형식으로 변환
- 중복 제거 및 데이터 정리
"""

import logging
from typing import Dict, Any, List
from src.search_pipeline.utils.file_manager import FileManager

class ResultConverter:
    """결과 변환기"""
    
    def __init__(self, file_manager: FileManager):
        """
        결과 변환기 초기화
        
        Args:
            file_manager: 파일 관리자
        """
        self.file_manager = file_manager
        
    def convert_to_csv(self, results: Dict[str, Any]) -> str:
        """
        JSON 결과를 CSV로 변환
        
        Args:
            results: 변환할 JSON 결과
            
        Returns:
            생성된 CSV 파일 경로
        """
        try:
            logging.info("🔄 JSON 결과를 CSV 형식으로 변환 중...")
            
            csv_file = self.file_manager.save_csv_results(results)
            
            # 통계 정보 계산
            total_queries = len(results.get("results", []))
            total_documents = sum(
                len(result.get("documents", [])) 
                for result in results.get("results", []) 
                if result.get("status") == "success"
            )
            
            logging.info(f"✅ CSV 파일 생성 완료: {csv_file}")
            logging.info(f"   총 {total_queries}개 질문, 각각 최대 50개 논문 정보 포함")
            
            return csv_file
            
        except Exception as e:
            logging.error(f"CSV 변환 실패: {e}")
            raise
    
    def convert_to_jsonl(self, results: Dict[str, Any]) -> str:
        """
        JSON 결과를 JSONL로 변환
        
        Args:
            results: 변환할 JSON 결과
            
        Returns:
            생성된 JSONL 파일 경로
        """
        try:
            logging.info("🔄 JSON 결과를 JSONL 형식으로 변환 중...")
            
            jsonl_file = self.file_manager.save_jsonl_results(results)
            
            # 통계 정보 계산
            unique_documents = self._count_unique_documents(results)
            
            logging.info(f"✅ JSONL 파일 생성 완료: {jsonl_file}")
            logging.info(f"   총 {unique_documents}개 중복 없는 논문")
            
            return jsonl_file
            
        except Exception as e:
            logging.error(f"JSONL 변환 실패: {e}")
            raise
    
    def convert_latest_json_to_csv(self) -> str:
        """가장 최근 JSON 파일을 CSV로 변환"""
        try:
            latest_json = self.file_manager.get_latest_json_file()
            if not latest_json:
                raise ValueError("변환할 JSON 파일을 찾을 수 없습니다")
            
            # JSON 파일 읽기
            import json
            with open(latest_json, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            return self.convert_to_csv(results)
            
        except Exception as e:
            logging.error(f"최신 JSON을 CSV로 변환 실패: {e}")
            raise
    
    def convert_latest_json_to_jsonl(self) -> str:
        """가장 최근 JSON 파일을 JSONL로 변환"""
        try:
            latest_json = self.file_manager.get_latest_json_file()
            if not latest_json:
                raise ValueError("변환할 JSON 파일을 찾을 수 없습니다")
            
            # JSON 파일 읽기
            import json
            with open(latest_json, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            return self.convert_to_jsonl(results)
            
        except Exception as e:
            logging.error(f"최신 JSON을 JSONL로 변환 실패: {e}")
            raise
    
    def _count_unique_documents(self, results: Dict[str, Any]) -> int:
        """중복 없는 문서 수 계산"""
        seen_titles = set()
        
        query_results = results.get("results", [])
        for result in query_results:
            if result.get("status") == "success":
                documents = result.get("documents", [])
                for doc in documents:
                    title = doc.get("title", "").strip()
                    if title:
                        seen_titles.add(title)
        
        return len(seen_titles)
    
    def get_conversion_stats(self, results: Dict[str, Any]) -> Dict[str, int]:
        """변환 통계 정보 반환"""
        total_queries = len(results.get("results", []))
        successful_queries = sum(
            1 for result in results.get("results", []) 
            if result.get("status") == "success"
        )
        total_documents = sum(
            len(result.get("documents", [])) 
            for result in results.get("results", []) 
            if result.get("status") == "success"
        )
        unique_documents = self._count_unique_documents(results)
        
        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "total_documents": total_documents,
            "unique_documents": unique_documents
        }
