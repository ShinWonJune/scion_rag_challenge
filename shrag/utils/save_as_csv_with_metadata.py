import csv
import json
import os
from dataclasses import make_dataclass, fields, is_dataclass
from typing import List, Any, Dict

from .create_class_from_schema import create_class_from_schema

# --- 2. 핵심 CSV 저장 로직 ---


def save_documents_batch(
    documents: List[Any], output_file: str, document_class: type, mode: str = "w"
):
    """
    여러 데이터 객체를 CSV 파일에 배치 저장합니다.

    Args:
        documents (List[Any]): 저장할 객체들의 리스트.
        output_file (str): 저장할 파일 경로.
        document_class (type): 데이터 객체의 타입 (동적으로 생성된 클래스).
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
    file_exists = os.path.exists(output_file)
    headers = [field.name for field in fields(document_class)]

    try:
        with open(output_file, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # 'w' 모드이거나, 'a' 모드인데 파일이 새로 생성될 때만 헤더 작성
            if mode == "w" or not file_exists:
                writer.writerow(headers)

            for doc in documents:
                row_data = [getattr(doc, header) for header in headers]
                writer.writerow(row_data)

    except IOError as e:
        raise IOError(
            f"파일 쓰기 오류: '{output_file}' 파일에 쓸 수 없습니다. 권한을 확인하세요. ({e})"
        )

    print(
        f"✅ 총 {len(documents)}개 항목을 '{output_file}'에 '{mode}' 모드로 저장했습니다."
    )


def save_document_single(
    document: Any, output_file: str, document_class: type, mode: str = "w"
):
    """
    하나의 데이터 객체를 CSV 파일에 저장합니다. (내부적으로 batch 함수 재사용)
    """
    # 단일 저장은 "객체가 1개인 배치 저장"과 동일하므로, batch 함수를 호출
    save_documents_batch([document], output_file, document_class, mode=mode)


# --- 3. 실제 사용 예시 ---

if __name__ == "__main__":
    try:
        # 1. 외부 스키마로부터 'Document' 클래스를 동적으로 생성
        DynamicDocument = create_class_from_schema(
            "Document", "/workspace/configs/csv_schema/test.json"
        )

        print(
            f"'{'schema.json'}'으로부터 '{DynamicDocument.__name__}' 클래스를 성공적으로 생성했습니다."
        )
        print("-" * 30)

        # 2. 동적으로 생성된 클래스를 사용하여 데이터 객체 생성
        docs_1 = [
            DynamicDocument("doc1", "첫 번째 문서", "요약1", [0.1, 0.2]),
            DynamicDocument("doc2", "두 번째 문서", "요약2", [0.3, 0.4]),
        ]
        doc_3 = DynamicDocument("doc3", "세 번째 문서", "요약3", [0.5, 0.6])
        docs_4 = [
            DynamicDocument("doc4", "네 번째 문서", "요약4", [0.7, 0.8]),
            DynamicDocument("doc5", "다섯 번째 문서", "요약5", [0.9, 1.0]),
        ]

        output_filename = "documents_output.csv"

        # 3. 시나리오별 함수 호출
        print("\n--- 시나리오 1: 배치 데이터로 새 파일 생성 (mode='w') ---")
        save_documents_batch(docs_1, output_filename, DynamicDocument, mode="w")

        print("\n--- 시나리오 2: 기존 파일에 단일 데이터 이어쓰기 (mode='a') ---")
        save_document_single(doc_3, output_filename, DynamicDocument, mode="a")

        print("\n--- 시나리오 3: 기존 파일에 배치 데이터 이어쓰기 (mode='a') ---")
        save_documents_batch(docs_4, output_filename, DynamicDocument, mode="a")

        print("\n--- 최종 파일 내용 확인 ---")
        with open(output_filename, "r", encoding="utf-8") as f:
            print(f.read().strip())
        print("-" * 30)

        print("\n--- 시나리오 4: 잘못된 타입의 데이터를 저장 시도 (오류 발생) ---")

        class WrongType:
            pass

        wrong_doc = WrongType()
        save_document_single(wrong_doc, "temp.csv", DynamicDocument)

    except (FileNotFoundError, TypeError, ValueError, IOError) as e:
        print(f"\n💥 프로그램 실행 중 오류가 발생했습니다: {e}")
