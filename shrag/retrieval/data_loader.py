# -*- coding: utf-8 -*-
"""Functions for loading VectorDB and questions."""

from __future__ import annotations

import ast
import csv
import dataclasses
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from shrag.retrieval import utils


@dataclasses.dataclass
class VectorDB:
    """In-memory vector database loaded from a CSV."""

    doc_ids: List[str]
    embeddings: np.ndarray  # (N, D), float32, L2-normalized
    metadata: List[Dict[str, Any]]


@dataclasses.dataclass
class QuestionItem:
    """Represents a single question with its decomposed parts."""

    qid: str
    question: str
    single_hop_questions: List[str]
    meta: Dict[str, Any]


def load_schema(schema_path: str) -> Dict[str, str]:
    """Loads a simple column:type schema from a JSON file."""
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema JSON not found: {schema_path}")
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Schema file '{schema_path}' is not valid JSON.")


def load_vectordb_from_csv(csv_path: str, schema_path: str) -> VectorDB:
    """
    Loads a vector database from a CSV file using Python's built-in csv module.
    This approach avoids pandas' automatic type inference, providing more robust parsing.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"VectorDB CSV not found: {csv_path}")

    schema = load_schema(schema_path)
    schema_columns = set(schema.keys())

    # --- Identify key columns based on convention ---
    embedding_col = "embedding"
    doc_id_candidates = ["cn", "CN", "doc_id"]
    doc_id_col = next((cand for cand in doc_id_candidates if cand in schema), None)

    if not doc_id_col:
        raise ValueError(
            f"Schema must define a document ID column from: {doc_id_candidates}"
        )
    if embedding_col not in schema:
        raise ValueError(f"Schema must define an '{embedding_col}' column.")
    # --- End of key column identification ---

    metadata_cols = [
        col for col in schema.keys() if col not in [doc_id_col, embedding_col]
    ]

    # --- Read data using the csv module ---
    doc_ids_list: List[str] = []
    embeddings_list: List[List[float]] = []
    metadata_list: List[Dict[str, Any]] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Validate CSV header against the schema
        csv_columns = set(reader.fieldnames or [])
        if not schema_columns.issubset(csv_columns):
            missing_cols = schema_columns - csv_columns
            raise ValueError(
                f"CSV file is missing required columns from schema: {missing_cols}"
            )

        for row in reader:
            # Document ID (empty string if not present)
            doc_ids_list.append(row.get(doc_id_col, ""))

            # Embedding (safely parse string representation)
            embedding_str = row.get(embedding_col) or "[]"
            try:
                embeddings_list.append(ast.literal_eval(embedding_str))
            except (ValueError, SyntaxError):
                embeddings_list.append([])  # Add empty list on parsing error

            # Metadata (all other schema columns)
            # An empty field in the CSV is read as an empty string '', which is what we want.
            metadata_item = {key: row.get(key, "") for key in metadata_cols}
            metadata_list.append(metadata_item)
    # --- End of data reading ---

    # --- Validate and Convert Embeddings ---
    embs_arr = np.array([])
    if embeddings_list:
        # Check for consistent embedding dimensions to prevent downstream errors.
        first_dim = len(embeddings_list[0])
        for i, emb in enumerate(embeddings_list[1:], 1):
            if len(emb) != first_dim:
                raise ValueError(
                    f"Inconsistent embedding dimension in '{csv_path}'. "
                    f"Row {i + 2} (header is row 1) has dimension {len(emb)}, "
                    f"but the first data row has dimension {first_dim}. "
                    "Please ensure all embeddings are generated with the same model."
                )

        # Convert lists to final, high-performance data structures
        embs_arr = np.array(embeddings_list, dtype=np.float32)
        if embs_arr.size > 0:
            embs_arr = utils.l2_normalize(embs_arr)
    # --- End of Validation ---

    return VectorDB(doc_ids=doc_ids_list, embeddings=embs_arr, metadata=metadata_list)


def load_questions_jsonl(path: str) -> List[QuestionItem]:
    """Loads questions from a JSONL file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Questions JSONL not found: {path}")
    items: List[QuestionItem] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append(
                QuestionItem(
                    qid=str(obj.get("id")),
                    question=str(obj.get("question", "")),
                    single_hop_questions=obj.get("single_hop_questions", []) or [],
                    meta=obj.get("meta", {}) or {},
                )
            )
    return items


def select_questions(
    all_items: List[QuestionItem],
    ids: Optional[List[str]] = None,
    idxs: Optional[List[int]] = None,
    range_spec: Optional[str] = None,
) -> List[QuestionItem]:
    """Selects a subset of questions based on IDs, indices, or a range."""
    if not (ids or idxs or range_spec):
        return all_items

    selected: List[QuestionItem] = []
    id_map = {it.qid: it for it in all_items}

    if ids:
        selected.extend(id_map[qid] for qid in ids if qid in id_map)

    if idxs:
        selected.extend(all_items[i - 1] for i in idxs if 0 < i <= len(all_items))

    if range_spec:
        try:
            start_str, end_str = range_spec.split("-", 1)
            start, end = int(start_str), int(end_str)
            if start < 1 or end > len(all_items) or start > end:
                raise ValueError
            selected.extend(all_items[i - 1] for i in range(start, end + 1))
        except (ValueError, IndexError):
            raise ValueError(
                f"Invalid range '{range_spec}'. Valid range is 1-{len(all_items)}"
            )

    # Remove duplicates while preserving the original order
    seen_qids = set()
    unique_selected = []
    for item in selected:
        if item.qid not in seen_qids:
            unique_selected.append(item)
            seen_qids.add(item.qid)
    return unique_selected
