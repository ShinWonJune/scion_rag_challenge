# /utils/save_as_json.py

import json
import os
from dataclasses import asdict, is_dataclass
from typing import List, Any


def save_documents_to_json_batch(
    documents: List[Any], output_file: str, document_class: type, mode: str = "w"
):
    """
    여러 데이터 객체를 JSON 파일에 배치 저장합니다.

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

    # --- 데이터 준비 ---
    # dataclass 객체를 dictionary 리스트로 변환
    new_data = [asdict(doc) for doc in documents]

    # --- 파일 처리 ---
    final_data = []
    file_exists = os.path.exists(output_file)

    try:
        # 'a' (이어쓰기) 모드이고 파일이 이미 존재할 경우
        if mode == "a" and file_exists and os.path.getsize(output_file) > 0:
            with open(output_file, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        final_data.extend(existing_data)
                    else:
                        # 기존 데이터가 리스트가 아니면 오류 발생
                        raise ValueError(
                            "JSON 파일의 최상위 객체는 리스트여야 이어쓰기가 가능합니다."
                        )
                except json.JSONDecodeError:
                    # 파일이 비어있거나 손상된 경우, 새 파일처럼 처리
                    print(
                        f"경고: '{output_file}' 파일이 비어있거나 유효한 JSON이 아니므로 덮어씁니다."
                    )

        final_data.extend(new_data)

        # 최종 데이터를 파일에 쓰기 (항상 'w' 모드로 덮어씀)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)

    except (IOError, ValueError) as e:
        raise IOError(
            f"파일 처리 오류: '{output_file}' 파일에 접근 중 문제가 발생했습니다. ({e})"
        )

    print(
        f"✅ 총 {len(documents)}개 항목을 추가하여 '{output_file}'에 저장했습니다. (최종 {len(final_data)}개)"
    )


def save_document_to_json_single(
    document: Any, output_file: str, document_class: type, mode: str = "w"
):
    """
    하나의 데이터 객체를 JSON 파일에 저장합니다. (내부적으로 batch 함수 재사용)
    """
    save_documents_to_json_batch([document], output_file, document_class, mode=mode)
