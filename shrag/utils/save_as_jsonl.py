# /utils/save_as_jsonl.py

import json
import os
from dataclasses import asdict, is_dataclass
from typing import List, Any


def save_documents_to_jsonl_batch(
    documents: List[Any], output_file: str, document_class: type, mode: str = "w"
):
    """
    여러 데이터 객체를 JSONL 파일에 배치 저장합니다.

    Args:
        documents (List[Any]): 저장할 객체들의 리스트.
        output_file (str): 저장할 파일 경로.
        document_class (type): 데이터 객체의 타입 (데이터 클래스).
        mode (str): 파일 저장 모드. 'w' (덮어쓰기) 또는 'a' (이어쓰기).
    """
    # --- 입력값 유효성 검사 ---
    if not documents:
        print("경고: 저장할 데이터가 없어 작업을 중단합니다.")
        return
    if not is_dataclass(document_class):
        raise TypeError("오류: 'document_class'는 데이터 클래스 타입이어야 합니다.")
    if mode not in ["w", "a"]:
        raise ValueError("오류: mode는 'w' 또는 'a'여야 합니다.")

    for doc in documents:
        if not isinstance(doc, document_class):
            raise TypeError(
                f"오류: 저장할 데이터는 모두 '{document_class.__name__}' 타입이어야 합니다."
            )

    # --- 파일 처리 ---
    try:
        # JSONL은 CSV처럼 mode ('w' or 'a')를 직접 사용할 수 있음
        with open(output_file, mode, encoding="utf-8") as f:
            for doc in documents:
                # dataclass를 dict로 변환 후 json 문자열로 변환
                doc_dict = asdict(doc)
                json_string = json.dumps(doc_dict, ensure_ascii=False)
                # 파일에 한 줄로 쓴다
                f.write(json_string + "\n")

    except IOError as e:
        raise IOError(
            f"파일 쓰기 오류: '{output_file}' 파일에 쓸 수 없습니다. 권한을 확인하세요. ({e})"
        )

    print(
        f"✅ 총 {len(documents)}개 항목을 '{output_file}'에 '{mode}' 모드로 저장했습니다."
    )


def save_document_to_jsonl_single(
    document: Any, output_file: str, document_class: type, mode: str = "w"
):
    """
    하나의 데이터 객체를 JSONL 파일에 저장합니다. (내부적으로 batch 함수 재사용)
    """
    save_documents_to_jsonl_batch([document], output_file, document_class, mode=mode)
