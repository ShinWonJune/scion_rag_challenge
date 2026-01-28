"""
메타데이터로부터 문서 커버리지 분석을 수행하는 통합 스크립트
1. 검색 결과에서 질문과 문서 제목 추출
2. 정답 문서와 검색 결과 간의 포함/누락 분석
"""

import json
from typing import Dict, List, Tuple, Set, Any
import re
from datetime import datetime
from pathlib import Path


def normalize_title(title: str) -> str:
    """
    문서 제목 정규화 (대소문자, 공백, 특수문자 처리)
    """
    # 소문자로 변환, 연속된 공백을 하나로, 양끝 공백 제거
    normalized = re.sub(r'\s+', ' ', title.lower().strip())
    # 특수문자 일부 제거 (선택적)
    normalized = re.sub(r'[^\w\s가-힣]', '', normalized)
    return normalized


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


def load_answer_docs(file_path: str) -> Dict[str, str]:
    """정답 문서 JSON 파일 로딩"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"정답 문서 로딩 중 오류: {e}")
        return {}


def is_title_match(title1: str, title2: str) -> bool:
    """두 제목이 일치하는지 확인 (완전 일치 및 부분 일치)"""
    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)
    
    # 완전 일치
    if norm1 == norm2:
        return True
    
    # 부분 일치 (둘 중 하나가 다른 하나에 포함)
    if norm1 in norm2 or norm2 in norm1:
        return True
    
    return False


def find_missing_answer_docs_globally(answer_docs: Dict[str, str], 
                                    extracted_questions: List[Dict]) -> List[Tuple[str, str]]:
    """
    분석 1: 전체 검색 결과에 포함되지 않은 정답 문서 찾기
    
    Returns:
        [(question_id, answer_title), ...] 형태의 리스트
    """
    # 모든 검색된 문서 제목을 하나의 세트로 수집
    all_extracted_titles = set()
    for question_data in extracted_questions:
        for title in question_data.get('document_titles', []):
            all_extracted_titles.add(normalize_title(title))
    
    missing_docs = []
    
    # 각 정답 문서가 전체 검색 결과에 포함되는지 확인
    for question_id, answer_title in answer_docs.items():
        answer_normalized = normalize_title(answer_title)
        
        # 완전 일치 확인
        if answer_normalized in all_extracted_titles:
            continue
        
        # 부분 일치 확인
        found = False
        for extracted_title in all_extracted_titles:
            if answer_normalized in extracted_title or extracted_title in answer_normalized:
                found = True
                break
        
        if not found:
            missing_docs.append((question_id, answer_title))
    
    return missing_docs


def find_missing_answer_docs_per_question(answer_docs: Dict[str, str], 
                                        extracted_questions: List[Dict]) -> List[Tuple[str, str, List[str]]]:
    """
    분석 2: 각 질문별로 정답 문서가 해당 질문의 검색 결과에 포함되지 않은 경우 찾기
    
    Returns:
        [(question_id, answer_title, [extracted_titles]), ...] 형태의 리스트
    """
    missing_per_question = []
    
    # 질문 ID별로 검색 결과 매핑
    question_map = {str(q['question_id']): q for q in extracted_questions}
    
    for question_id, answer_title in answer_docs.items():
        # 해당 질문의 검색 결과가 있는지 확인
        if question_id not in question_map:
            # 질문 자체가 없는 경우
            missing_per_question.append((question_id, answer_title, []))
            continue
        
        question_data = question_map[question_id]
        extracted_titles = question_data.get('document_titles', [])
        
        # 정답 문서가 해당 질문의 검색 결과에 포함되는지 확인
        found = False
        for extracted_title in extracted_titles:
            if is_title_match(answer_title, extracted_title):
                found = True
                break
        
        if not found:
            missing_per_question.append((question_id, answer_title, extracted_titles))
    
    return missing_per_question


def print_extraction_summary(extracted_data: List[Dict[str, Any]]):
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


def analyze_document_coverage(answer_docs: Dict[str, str], extracted_questions: List[Dict], 
                            analysis_output_file: str = None, search_results_file: str = None) -> Dict:
    """
    문서 커버리지 분석 수행
    
    Args:
        answer_docs: 정답 문서 딕셔너리
        extracted_questions: 추출된 질문 데이터 리스트
        analysis_output_file: 분석 결과 저장 파일 경로 (선택적)
        search_results_file: 분석에 사용된 검색 결과 파일 경로 (선택적)
        
    Returns:
        분석 결과 딕셔너리
    """
    print(f"\n정답 문서 수: {len(answer_docs)}")
    print(f"검색된 질문 수: {len(extracted_questions)}")
    
    # 분석 1: 전체 검색 결과에 포함되지 않은 정답 문서
    print("\n" + "="*80)
    print("분석 1: 전체 검색 결과에 포함되지 않은 정답 문서")
    print("="*80)
    
    missing_globally = find_missing_answer_docs_globally(answer_docs, extracted_questions)
    
    print(f"\n총 {len(missing_globally)}개의 정답 문서가 전체 검색 결과에서 누락됨")
    print(f"누락률: {len(missing_globally) / len(answer_docs) * 100:.2f}%")
    
    if missing_globally:
        print("\n누락된 정답 문서 목록:")
        for i, (qid, title) in enumerate(missing_globally, 1):
            print(f"{i:2d}. [Q{qid}] {title}")
    
    # 분석 2: 각 질문별로 정답 문서가 누락된 경우
    print("\n" + "="*80)
    print("분석 2: 각 질문별로 정답 문서가 누락된 경우")
    print("="*80)
    
    missing_per_question = find_missing_answer_docs_per_question(answer_docs, extracted_questions)
    
    print(f"\n총 {len(missing_per_question)}개 질문에서 정답 문서가 누락됨")
    print(f"질문별 누락률: {len(missing_per_question) / len(answer_docs) * 100:.2f}%")
    
    if missing_per_question:
        print("\n질문별 누락 상세:")
        for i, (qid, answer_title, extracted_titles) in enumerate(missing_per_question, 1):
            print(f"\n{i:2d}. [Q{qid}] 정답: {answer_title}")
            if extracted_titles:
                print(f"    검색된 문서 수: {len(extracted_titles)}")
                print(f"    첫 번째 검색 문서: {extracted_titles[0][:60]}...")
            else:
                print(f"    검색 결과 없음")
    
    # 요약 통계
    print("\n" + "="*80)
    print("요약 통계")
    print("="*80)
    
    total_answer_docs = len(answer_docs)
    global_missing = len(missing_globally)
    question_missing = len(missing_per_question)
    
    global_coverage = (total_answer_docs - global_missing) / total_answer_docs * 100
    question_coverage = (total_answer_docs - question_missing) / total_answer_docs * 100
    
    summary = {
        'metadata': {},
        'total_answer_documents': total_answer_docs,
        'analysis_1': {
            'description': '전체 검색 결과에 포함되지 않은 정답 문서',
            'missing_count': global_missing,
            'coverage_rate': round(global_coverage, 2),
            'missing_docs': [{'question_id': qid, 'title': title} for qid, title in missing_globally]
        },
        'analysis_2': {
            'description': '각 질문별로 정답 문서가 누락된 경우',
            'missing_count': question_missing,
            'coverage_rate': round(question_coverage, 2),
            'missing_per_question': [
                {
                    'question_id': qid, 
                    'answer_title': title, 
                    'extracted_count': len(extracted)
                } 
                for qid, title, extracted in missing_per_question
            ]
        }
          # 메타데이터는 main()에서 추가됨
    }
    
    print(f"전체 정답 문서 수: {total_answer_docs}")
    print(f"전체 검색 결과 커버리지: {global_coverage:.2f}% ({total_answer_docs - global_missing}/{total_answer_docs})")
    print(f"질문별 검색 결과 커버리지: {question_coverage:.2f}% ({total_answer_docs - question_missing}/{total_answer_docs})")
    
    # 메타데이터 추가
    if search_results_file:
        summary['metadata']['search_results_file'] = search_results_file
    summary['metadata']['analysis_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 결과 저장
    if analysis_output_file:
        try:
            with open(analysis_output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n분석 결과가 {analysis_output_file}에 저장되었습니다.")
        except Exception as e:
            print(f"파일 저장 중 오류: {e}")
    
    return summary


def main():
    """메인 실행 함수"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 입력 파일 경로
    search_results_file = "outputs/search_meta_results_20251021_114157.json"
    answer_docs_file = "outputs/scion_answer_docs.json"
    
    # 출력 파일 경로 (타임스탬프 포함) - 최종 결과만 저장
    analysis_output_file = f"outputs/scion_document_coverage_analysis_{timestamp}.json"
    
    print("="*80)
    print("메타데이터로부터 문서 커버리지 분석 시작")
    print("="*80)
    
    # 1단계: 검색 결과에서 질문과 문서 제목 추출 (메모리에만 저장)
    print(f"\n1단계: 검색 결과에서 질문과 문서 제목 추출")
    print(f"입력 파일: {search_results_file}")
    
    extracted_questions = extract_questions_and_titles(search_results_file, output_file=None)
    
    if not extracted_questions:
        print("질문 데이터 추출에 실패했습니다.")
        return
    
    print_extraction_summary(extracted_questions)
    
    # 2단계: 정답 문서 로딩
    print(f"\n2단계: 정답 문서 로딩")
    print(f"정답 파일: {answer_docs_file}")
    
    answer_docs = load_answer_docs(answer_docs_file)
    
    if not answer_docs:
        print("정답 문서 로딩에 실패했습니다.")
        return
    
    print(f"정답 문서 {len(answer_docs)}개 로딩 완료")
    
    # 3단계: 문서 커버리지 분석
    print(f"\n3단계: 문서 커버리지 분석")
    
    analysis_results = analyze_document_coverage(
        answer_docs, 
        extracted_questions, 
        analysis_output_file,
        search_results_file
    )
    
    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80)
    print(f"분석 결과: {analysis_output_file}")
    
    return analysis_results


if __name__ == "__main__":
    main()