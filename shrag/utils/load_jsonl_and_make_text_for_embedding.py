import jsonlines
from typing import List, Dict
import json


def load_jsonl_2(jsonl_path: str) -> List[Dict]:
    """
    JSONL 파일에서 모든 문서를 로드

    Args:
        jsonl_path: JSONL 파일 경로

    Returns:
        documents: 문서 원본 데이터 리스트
    """
    documents = []
    with jsonlines.open(jsonl_path) as reader:
        for doc in reader:
            print(reader)
            documents.append(doc)
    return documents


def load_jsonl(path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                docs.append(json.loads(s))
            except Exception as e:
                # 👉 여기서 구체적으로 알려줌
                raise ValueError(
                    f"Invalid JSON at line {i}: {e}\nLine content (truncated): {s[:200]}"
                ) from e
    return docs


def make_text_for_embedding(
    documents: List[Dict], embedding_mode: str = "title+abstract"
) -> List[Dict]:
    """
    문서 리스트에서 임베딩할 텍스트 생성

    Args:
        documents: 문서 리스트
        embedding_mode: 텍스트 생성 방식 (예: "title+abstract", "3*title+abstract")

    Returns:
        documents_data: 임베딩용 텍스트가 포함된 문서 리스트
    """
    documents_data = []

    for doc in documents:
        cn = doc.get("doc_id") or doc.get("CN") or doc.get("cn", "")
        title = doc.get("title", "")
        abstract = doc.get("abstract", "")
        source = doc.get("source", "")

        if not (title.strip() or abstract.strip()):
            continue

        # embedding_mode에 따라 텍스트 생성
        if embedding_mode == "3*title+abstract":
            embedding_text = f"{title} {title} {title} {abstract}"
        elif embedding_mode == "title+abstract":
            embedding_text = f"{title} {abstract}"
        elif embedding_mode == "title":
            embedding_text = title
        elif embedding_mode == "abstract":
            embedding_text = abstract
        else:
            raise ValueError(f"Unknown embedding_mode: {embedding_mode}")

        documents_data.append(
            {
                "cn": cn,
                "title": title,
                "abstract": abstract,
                "source": source,
                "embedding_text": embedding_text,
                "embedding_mode": embedding_mode,
            }
        )

    return documents_data


def load_jsonl_and_make_text_for_embedding(jsonl_path, embedding_mode="title+abstract"):
    """
    JSONL 파일에서 모든 문서를 로드하고 임베딩할 텍스트 생성

    Args:
        jsonl_path: JSONL 파일 경로

    Returns:
        documents_data: 문서 정보가 담긴 리스트
    """
    docs = load_jsonl(jsonl_path)
    docs_with_emb = make_text_for_embedding(docs, embedding_mode="3*title+abstract")
    return docs_with_emb
