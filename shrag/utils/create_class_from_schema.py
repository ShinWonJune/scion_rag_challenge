import csv
import json
import os
from dataclasses import make_dataclass, fields, is_dataclass
from typing import List, Any, Dict

# --- 1. 동적 클래스 생성을 위한 헬퍼 ---

# JSON의 타입 문자열을 실제 파이썬 타입으로 변환하는 맵
TYPE_MAPPING = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "List[str]": List[str],
    "List[int]": List[int],
    "List[float]": List[float],
    "Dict": Dict,
    "Any": Any,
}


def create_class_from_schema(class_name: str, schema_path: str = "schema.json") -> type:
    """
    JSON 스키마 파일로부터 동적으로 데이터 클래스를 생성합니다.

    Args:
        class_name (str): 생성할 클래스의 이름 (예: "Document").
        schema_path (str): 스키마 파일의 경로.

    Returns:
        type: 동적으로 생성된 데이터 클래스.

    Raises:
        FileNotFoundError: 스키마 파일이 존재하지 않을 경우.
        TypeError: 스키마에 지원하지 않는 타입이 정의된 경우.
    """
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"오류: 스키마 파일 '{schema_path}'을 찾을 수 없습니다."
        )
    except json.JSONDecodeError:
        raise ValueError(f"오류: '{schema_path}' 파일이 올바른 JSON 형식이 아닙니다.")

    fields_for_class = []
    for field_name, field_type_str in schema.items():
        actual_type = TYPE_MAPPING.get(field_type_str)
        if actual_type is None:
            raise TypeError(
                f"스키마 오류: 지원하지 않는 타입 '{field_type_str}'입니다."
            )
        fields_for_class.append((field_name, actual_type))

    # make_dataclass를 사용하여 동적으로 데이터 클래스 생성
    return make_dataclass(class_name, fields_for_class)
