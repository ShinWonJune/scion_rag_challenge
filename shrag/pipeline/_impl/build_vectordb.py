# main.py
import os
import json
import argparse
from typing import List, Optional
from shrag.utils.load_json import load_config
from shrag.utils.load_jsonl_and_make_text_for_embedding import (
    load_jsonl_and_make_text_for_embedding as load_jsonl_docs,
)
from shrag.utils.create_class_from_schema import create_class_from_schema

# Import the newly created modules
from shrag.features.embedding_processor import generate_batch_embeddings
from shrag.data_handler.for_embedding import prepare_documents, save_results


def build_vectordb_search(
    config_path="../configs/query_encoder/config_gte-multilingual-base.json",
    data_schema="../configs/csv_schema/test_2.json",
    docs_jsonl_path: Optional[str] = None,
    auto_data_load=False,
    gpu_id: Optional[int] = None,
):
    # 1. Load configurations and create dynamic document class
    config = load_config(config_path)
    try:
        DynamicDocument = create_class_from_schema("Document", data_schema)
    except Exception as e:
        print(f"Failed to load schema and create class: {e}")
        return

    # 2. Load source documents
    if auto_data_load:
        jsonl_path = config.get("jsonl_path", config_path)
    else:
        if not docs_jsonl_path:
            print("Error: --docs_jsonl_path is required when --auto_data_load is not set.")
            return
        jsonl_path = docs_jsonl_path
    print(f"Loading documents from {jsonl_path}: Auto Loading Data", auto_data_load)
    documents_data = load_jsonl_docs(jsonl_path)
    print(f"Loaded {len(documents_data)} documents")

    if not documents_data:
        print("No documents found. Exiting.")
        return

    # 3. Generate Embeddings (Separated Logic)
    # This function is now focused only on the ML model and vector generation.
    model_name = config["model_name"]
    embeddings = generate_batch_embeddings(
        documents_data, model_name, config["embedding_dim"], gpu_id=gpu_id
    )

    if embeddings is None:
        print("Embedding generation failed. Exiting.")
        return
    expected_dim = int(config["embedding_dim"])
    actual_dim = int(embeddings.shape[1])
    if actual_dim != expected_dim:
        raise ValueError(
            f"embedding_dim mismatch for {config['model_name']}: "
            f"config={expected_dim} actual={actual_dim}"
        )

    # 4. Prepare Document Objects for Saving (Separated Logic)
    # This function handles the data structuring, combining raw data with embeddings.
    documents_to_save = prepare_documents(documents_data, embeddings, DynamicDocument)

    if not documents_to_save:
        print("No documents were successfully prepared for saving. Exiting.")
        return

    # 5. Save Results and Update Config (Separated Logic)
    # This function is responsible for all file I/O and finalization.
    save_results(
        config=config,
        documents_to_save=documents_to_save,
        embedding_shape=embeddings.shape,
        document_class=DynamicDocument,
        config_path=config_path,
        model_name=model_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="벡터 데이터베이스 구축을 위한 문서 임베딩 생성"
    )
    
    parser.add_argument(
        "--config_path",
        default="../configs/query_encoder/config_gte-multilingual-base.json",
        help="임베딩 모델 설정 파일 경로"
    )
    parser.add_argument(
        "--data_schema",
        default="../configs/csv_schema/test_2.json",
        help="데이터 스키마 파일 경로"
    )
    parser.add_argument(
        "--docs_jsonl_path",
        default=None,
        help="문서 JSONL 파일 경로"
    )
    parser.add_argument(
        "--auto_data_load",
        action="store_true",
        help="설정 파일에서 데이터 경로를 자동으로 로드"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="사용할 GPU ID (예: 0, 1, 2, 3). 지정하지 않으면 자동 선택"
    )
    
    args = parser.parse_args()
    
    build_vectordb_search(
        config_path=args.config_path,
        data_schema=args.data_schema,
        docs_jsonl_path=args.docs_jsonl_path,
        auto_data_load=args.auto_data_load,
        gpu_id=args.gpu_id
    )
