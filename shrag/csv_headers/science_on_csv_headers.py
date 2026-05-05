import csv
from dataclasses import dataclass, fields
from typing import List, Any


@dataclass
class Document:
    # 각 데이터의 타입을 명시합니다.
    cn: str
    title: str
    abstract: str
    source: str
    embedding_mode: str
    embedding_text: str
    embedding: List[float]  # 임베딩은 float 값의 리스트
