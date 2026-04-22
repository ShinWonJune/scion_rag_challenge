# data_handler.py
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Type
from dataclasses import fields

# Assuming save_documents_batch is in this utility file
from src.utils.save_as_csv_with_metadata import save_documents_batch


def prepare_documents(
    documents_data: List[Dict], embeddings: Any, document_class: Type
) -> List[Any]:
    """
    Combines original document data with their embeddings into a list of document objects.

    Args:
        documents_data (List[Dict]): The original list of document data.
        embeddings (Any): The numpy array of embeddings.
        document_class (Type): The dynamic dataclass to use for creating document instances.

    Returns:
        A list of document class instances ready to be saved.
    """
    documents_to_save = []
    schema_fields = {field.name for field in fields(document_class)}

    for doc_data, emb in zip(documents_data, embeddings):
        filtered_data = {k: v for k, v in doc_data.items() if k in schema_fields}

        # Ensure the embedding field is present in the schema before adding
        if "embedding" in schema_fields:
            filtered_data["embedding"] = emb.tolist()
        else:
            print(
                "Warning: 'embedding' field not in schema. Skipping embedding data for this record."
            )

        try:
            doc_instance = document_class(**filtered_data)
            documents_to_save.append(doc_instance)
        except TypeError as e:
            print(f"Data object creation error: {e}. Check schema vs. data keys.")
            print(f"  - Missing keys in data: {schema_fields - filtered_data.keys()}")
            print(
                f"  - Extra keys in data (filtered): {filtered_data.keys() - schema_fields}"
            )
            continue

    return documents_to_save


def save_results(
    config: Dict,
    documents_to_save: List[Any],
    embedding_shape: tuple,
    document_class: Type,
    config_path: str,
    model_name: str,
):
    """
    Saves the prepared documents to a CSV file and updates the configuration.

    Args:
        config (Dict): The configuration dictionary.
        documents_to_save (List[Any]): The list of document objects to save.
        embedding_shape (tuple): The shape of the embeddings array.
        document_class (Type): The dynamic dataclass used.
        config_path (str): The path to the configuration file for updating.
        model_name (str): The name of the model used for embeddings.
    """
    # 1. Create save path and filename
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_")
    output_dir = os.path.join(config.get("output_dir", "../output"), timestamp)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"vector_db_{config.get('nickname', 'docs')}_{safe_model_name}.csv"
    )

    # 2. Save documents to CSV
    # This function needs to be imported from your utils
    save_documents_batch(
        documents=documents_to_save,
        output_file=output_file,
        document_class=document_class,
        mode="w",
    )
    print(
        f"--- SIMULATING SAVE to {output_file} ---"
    )  # Placeholder for your save function

    # 3. Print results and update config
    print(f"✅ VectorDB saved to {output_file}")
    print(f"✅ Embedding mode: 3*title+abstract")
    print(f"✅ Total documents embedded: {len(documents_to_save)}")
    print(f"✅ Embedding dimension: {embedding_shape[1]}")

    config.update(
        {
            "output_file": output_file,
            "jsonl_path": config.get("jsonl_path"),
            "embedding_mode": "3*title+abstract",
            "last_run": timestamp,
        }
    )
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"✅ Config updated at {config_path}")
