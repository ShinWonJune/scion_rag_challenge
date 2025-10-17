"""
정답 문서와 검색 결과 간의 포함/누락 분석
1. 전체 검색 결과에 포함되지 않은 정답 문서 찾기
2. 각 질문별로 정답 문서가 누락된 경우 찾기
"""

import json
from typing import Dict, List, Tuple, Set
import re


def normalize_title(title: str) -> str:
    """
    문서 제목 정규화 (대소문자, 공백, 특수문자 처리)
    """
    # 소문자로 변환, 연속된 공백을 하나로, 양끝 공백 제거
    normalized = re.sub(r'\s+', ' ', title.lower().strip())
    # 특수문자 일부 제거 (선택적)
    normalized = re.sub(r'[^\w\s가-힣]', '', normalized)
    return normalized


def load_answer_docs(file_path: str) -> Dict[str, str]:
    """정답 문서 JSON 파일 로딩"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"정답 문서 로딩 중 오류: {e}")
        return {}


def load_extracted_questions(file_path: str) -> List[Dict]:
    """검색된 질문 데이터 JSON 파일 로딩"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"검색 결과 로딩 중 오류: {e}")
        return []


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


def analyze_document_coverage(answer_docs_file: str, extracted_questions_file: str, output_file: str = None):
    """
    메인 분석 함수
    
    Args:
        answer_docs_file: 정답 문서 JSON 파일 경로
        extracted_questions_file: 검색 결과 JSON 파일 경로
        output_file: 결과 저장 파일 경로 (선택적)
    """
    print("데이터 로딩 중...")
    answer_docs = load_answer_docs(answer_docs_file)
    extracted_questions = load_extracted_questions(extracted_questions_file)
    
    print(f"정답 문서 수: {len(answer_docs)}")
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
    }
    
    print(f"전체 정답 문서 수: {total_answer_docs}")
    print(f"전체 검색 결과 커버리지: {global_coverage:.2f}% ({total_answer_docs - global_missing}/{total_answer_docs})")
    print(f"질문별 검색 결과 커버리지: {question_coverage:.2f}% ({total_answer_docs - question_missing}/{total_answer_docs})")
    
    # 결과 저장
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n분석 결과가 {output_file}에 저장되었습니다.")
        except Exception as e:
            print(f"파일 저장 중 오류: {e}")
    
    return summary


def main():
    """메인 실행 함수"""
    answer_docs_file = "outputs/scion_answer_docs.json"
    extracted_questions_file = "outputs/extracted_questions_titles.json"
    output_file = "outputs/document_coverage_analysis.json"
    
    print("문서 커버리지 분석을 시작합니다...")
    results = analyze_document_coverage(
        answer_docs_file, 
        extracted_questions_file, 
        output_file
    )
    
    return results


if __name__ == "__main__":
    main()