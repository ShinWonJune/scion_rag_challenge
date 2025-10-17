"""
검색 결과에서 질문과 문서 제목을 추출하는 함수
- question, document titles, question_id 추출
- JSON 형태로 출력
"""

import json
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime


def extract_questions_and_titles(input_file: str, output_file: str = None) -> List[Dict[str, Any]]:
    """
    검색 결과 파일에서 질문과 문서 제목들을 추출
    
    Args:
        input_file: 입력 JSON 파일 경로
        output_file: 출력 JSON 파일 경로 (선택적)
        
    Returns:
        추출된 데이터 리스트
    """
    try:
        # JSON 파일 읽기
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        extracted_data = []
        
        # 각 결과에서 질문과 문서 제목들 추출
        for idx, result in enumerate(data.get('results', [])):
            # 성공한 결과는 'question', 실패한 결과는 'query' 키 사용
            question = result.get('question', result.get('query', ''))
            documents = result.get('documents', [])
            
            # 문서 제목들 추출
            document_titles = []
            for doc in documents:
                title = doc.get('title', '').strip()
                if title:
                    document_titles.append(title)
            
            # 데이터 구성
            question_data = {
                'question_id': idx,  # 0부터 시작
                'question': question,
                'document_titles': document_titles,
                'document_count': len(document_titles)
            }
            
            extracted_data.append(question_data)
        
        # 출력 파일이 지정된 경우 저장
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, ensure_ascii=False, indent=2)
            print(f"추출된 데이터가 {output_file}에 저장되었습니다.")
        
        return extracted_data
        
    except Exception as e:
        print(f"데이터 추출 중 오류 발생: {e}")
        return []


def print_summary(extracted_data: List[Dict[str, Any]]):
    """추출된 데이터 요약 출력"""
    if not extracted_data:
        print("추출된 데이터가 없습니다.")
        return
    
    total_questions = len(extracted_data)
    total_documents = sum(item['document_count'] for item in extracted_data)
    avg_docs_per_question = total_documents / total_questions if total_questions > 0 else 0
    
    print(f"\n=== 추출 결과 요약 ===")
    print(f"총 질문 수: {total_questions}")
    print(f"총 문서 수: {total_documents}")
    print(f"질문당 평균 문서 수: {avg_docs_per_question:.2f}")
    
    # 처음 3개 질문 미리보기
    print(f"\n=== 처음 3개 질문 미리보기 ===")
    for i, item in enumerate(extracted_data[:3]):
        print(f"\n[질문 {item['question_id']}]")
        print(f"질문: {item['question']}")
        print(f"문서 수: {item['document_count']}")
        if item['document_titles']:
            print(f"첫 번째 문서: {item['document_titles'][0][:100]}...")


def main():
    """메인 실행 함수"""
    # 입력 파일 경로
    input_file = "outputs/search_meta_results_20251017_080800.json"
    
    # 현재 시간을 포함한 출력 파일 경로
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputs/extracted_questions_titles_{timestamp}.json"
    
    # 데이터 추출
    print(f"파일에서 데이터 추출 중: {input_file}")
    extracted_data = extract_questions_and_titles(input_file, output_file)
    
    # 요약 출력
    print_summary(extracted_data)
    
    return extracted_data


if __name__ == "__main__":
    main()