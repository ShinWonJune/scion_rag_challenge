# main.py
import os
import json
from typing import List
from utils.load_json import load_config
from utils.load_jsonl_and_make_text_for_embedding import (
    load_jsonl_and_make_text_for_embedding as load_jsonl_docs,
)
from utils.create_class_from_schema import create_class_from_schema

# Import the newly created modules
from features.embedding_processor import generate_batch_embeddings
from data_handler.for_embedding import prepare_documents, save_results


def build_vectordb_search(
    config_path="../configs/query_encoder/config_gte-multilingual-base.json",
    data_schema="/workspace/configs/csv_schema/test_2.json",
    docs_jsonl_path="/workspace/data/expr/search_documents_20250912_013206.jsonl",
    auto_data_load=False,
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
        documents_data, model_name, config["embedding_dim"]
    )

    if embeddings is None:
        print("Embedding generation failed. Exiting.")
        return

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
    build_vectordb_search()
