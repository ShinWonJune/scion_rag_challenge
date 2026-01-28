"""
scion_answer_docs.txt 파일을 파싱하여 query_id: title 형태의 JSON 생성
- 첫 번째 문자(O, o, X 등) 무시
- 숫자는 쿼리 인덱스로 사용
- 나머지 내용은 정답 문서 제목으로 사용
"""

import json
import re
from typing import Dict


def parse_answer_docs_to_json(input_file: str, output_file: str = None) -> Dict[str, str]:
    """
    scion_answer_docs.txt 파일을 파싱하여 JSON 형태로 변환
    
    Args:
        input_file: 입력 파일 경로 (scion_answer_docs.txt)
        output_file: 출력 JSON 파일 경로 (선택적)
        
    Returns:
        {query_id: title} 딕셔너리
    """
    answer_docs = {}
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            # 패턴 매칭: [첫 번째 문자] [숫자]. [제목]
            # 예: "O 0. 인공지능(Artificial Intelligence)과 대학수학교육"
            # 예: "X 39. 산업생태계 관점에서 바라본 IT융합 촉진전략 ㅌ"
            match = re.match(r'^[^\d]*(\d+)\.?\s*(.+)$', line)
            
            if match:
                query_id = match.group(1)  # 숫자 부분
                title = match.group(2).strip()  # 제목 부분
                
                if title:  # 빈 제목이 아닌 경우만 추가
                    answer_docs[query_id] = title
                    print(f"Query {query_id}: {title}")
                else:
                    print(f"경고: Line {line_num}에서 빈 제목 발견: {line}")
            else:
                print(f"경고: Line {line_num}에서 패턴 매칭 실패: {line}")
        
        print(f"\n총 {len(answer_docs)}개의 정답 문서를 파싱했습니다.")
        
        # JSON 파일로 저장
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(answer_docs, f, ensure_ascii=False, indent=2)
            print(f"결과가 {output_file}에 저장되었습니다.")
        
        return answer_docs
        
    except Exception as e:
        print(f"파일 파싱 중 오류 발생: {e}")
        return {}


def print_summary(answer_docs: Dict[str, str]):
    """파싱 결과 요약 출력"""
    if not answer_docs:
        print("파싱된 데이터가 없습니다.")
        return
    
    print(f"\n=== 파싱 결과 요약 ===")
    print(f"총 정답 문서 수: {len(answer_docs)}")
    
    # 쿼리 ID 범위 확인
    query_ids = [int(qid) for qid in answer_docs.keys()]
    min_id, max_id = min(query_ids), max(query_ids)
    print(f"쿼리 ID 범위: {min_id} ~ {max_id}")
    
    # 누락된 쿼리 ID 확인
    expected_ids = set(range(min_id, max_id + 1))
    actual_ids = set(query_ids)
    missing_ids = expected_ids - actual_ids
    
    if missing_ids:
        print(f"누락된 쿼리 ID: {sorted(missing_ids)}")
    else:
        print("모든 쿼리 ID가 연속적으로 존재합니다.")
    
    # 처음 5개 샘플 출력
    print(f"\n=== 처음 5개 샘플 ===")
    for i, (qid, title) in enumerate(sorted(answer_docs.items(), key=lambda x: int(x[0]))[:5]):
        print(f"Query {qid}: {title[:60]}{'...' if len(title) > 60 else ''}")


def main():
    """메인 실행 함수"""
    input_file = "scion_answer_docs.txt"
    output_file = "outputs/answer_docs.json"
    
    print(f"파싱 시작: {input_file}")
    answer_docs = parse_answer_docs_to_json(input_file, output_file)
    
    # 결과 요약 출력
    print_summary(answer_docs)
    
    return answer_docs


if __name__ == "__main__":
    main()