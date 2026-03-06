"""
파일 관리자 모듈
- JSON 결과 저장
- 파일 경로 관리
- 출력 디렉토리 관리
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

class FileManager:
    """파일 관리자"""
    
    def __init__(self, output_dir: str = "./outputs"):
        """
        파일 관리자 초기화
        
        Args:
            output_dir: 출력 디렉토리 경로
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def save_json_results(self, results: Dict[str, Any], filename_prefix: str = "search_meta_results") -> str:
        """
        JSON 결과 저장
        
        Args:
            results: 저장할 결과 데이터
            filename_prefix: 파일명 접두사
            
        Returns:
            저장된 파일 경로
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logging.info(f"JSON 결과 저장 완료: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logging.error(f"JSON 결과 저장 실패: {e}")
            raise
    
    def save_csv_results(self, results: Dict[str, Any], filename_prefix: str = "search_results") -> str:
        """
        CSV 결과 저장
        
        Args:
            results: 저장할 결과 데이터
            filename_prefix: 파일명 접두사
            
        Returns:
            저장된 파일 경로
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.csv"
            filepath = self.output_dir / filename
            
            # CSV 데이터 생성
            csv_data = self._prepare_csv_data(results)
            
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                import csv
                if csv_data:
                    writer = csv.writer(f)
                    writer.writerows(csv_data)
            
            logging.info(f"CSV 결과 저장 완료: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logging.error(f"CSV 결과 저장 실패: {e}")
            raise
    
    def save_jsonl_results(self, results: Dict[str, Any], filename_prefix: str = "search_documents") -> str:
        """
        JSONL 결과 저장
        
        Args:
            results: 저장할 결과 데이터
            filename_prefix: 파일명 접두사
            
        Returns:
            저장된 파일 경로
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.jsonl"
            filepath = self.output_dir / filename
            
            # JSONL 데이터 생성
            jsonl_data = self._prepare_jsonl_data(results)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                for line in jsonl_data:
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')
            
            logging.info(f"JSONL 결과 저장 완료: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logging.error(f"JSONL 결과 저장 실패: {e}")
            raise
    
    def _prepare_csv_data(self, results: Dict[str, Any]) -> list:
        """CSV 데이터 준비 (기존 형식에 맞춤)"""
        csv_data = []
        
        # 헤더 생성 (Question + 50개 컬럼)
        header = ["Question"]
        for i in range(1, 51):
            header.append(f"Prediction_retrieved_article_name_{i}")
        csv_data.append(header)
        
        # 결과 데이터 추가
        query_results = results.get("results", [])
        for result in query_results:
            if result.get("status") == "success":
                query = result.get("question", "")
                documents = result.get("documents", [])
                
                # 각 질문당 최대 50개 문서
                row = [query]
                for i in range(50):
                    if i < len(documents):
                        doc = documents[i]
                        # 기존 형식에 맞춰 포맷팅
                        formatted_doc = f"Title: {doc.get('title', '')}, Abstract: {doc.get('abstract', '')}, Source: {doc.get('source', '')}"
                        row.append(formatted_doc)
                    else:
                        row.append("")
                
                csv_data.append(row)
        
        return csv_data
    
    def _prepare_jsonl_data(self, results: Dict[str, Any]) -> list:
        """JSONL 데이터 준비"""
        jsonl_data = []
        seen_documents = set()
        
        # 중복 제거를 위한 문서 추적
        query_results = results.get("results", [])
        for result in query_results:
            if result.get("status") == "success":
                documents = result.get("documents", [])
                
                for doc in documents:
                    # 중복 제거 (제목 기준)
                    title = doc.get("title", "").strip()
                    if title and title not in seen_documents:
                        seen_documents.add(title)
                        jsonl_data.append(doc)
        
        return jsonl_data
    
    def get_latest_json_file(self) -> str:
        """가장 최근 JSON 파일 경로 반환"""
        try:
            json_files = list(self.output_dir.glob("search_meta_results_*.json"))
            if not json_files:
                return ""
            
            # 파일명으로 정렬 (타임스탬프 기준)
            latest_file = max(json_files, key=lambda x: x.name)
            return str(latest_file)
            
        except Exception as e:
            logging.error(f"최신 JSON 파일 찾기 실패: {e}")
            return ""
    
    def save_elapsed_times(self, elapsed_times: Dict[str, float]):
        """실행 시간 저장"""
        try:
            filepath = self.output_dir / "elapsed_times.json"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(elapsed_times, f, ensure_ascii=False, indent=2)
            
            logging.info(f"실행 시간 저장 완료: {filepath}")
            
        except Exception as e:
            logging.error(f"실행 시간 저장 실패: {e}")
