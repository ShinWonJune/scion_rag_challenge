# -*- coding: utf-8 -*-
"""Main script to run the dense retrieval process."""

from __future__ import annotations

import argparse
import json
import os
import sys

# retrieval_system 패키지의 모듈들을 절대 경로로 임포트
from shrag.retrieval import data_loader, query_encoder, result_saver, utils
from shrag.retrieval.retrievers import get_retriever


def run_retrieval_for_question(
    q_item: data_loader.QuestionItem,
    encoder: query_encoder.QueryEncoder,
    index: "retriever_base.Retriever",
    vectordb: data_loader.VectorDB,
    top_k: int,
    model_name: str,
    query_instruction: str | None = None,
) -> dict[str, any]:
    """Runs retrieval for a single question item (original + single-hop)."""
    queries = [(q_item.question, {"type": "original"})]
    queries.extend(
        (q, {"type": "single_hop", "index": i})
        for i, q in enumerate(q_item.single_hop_questions)
    )

    q_texts = [q for q, _ in queries]
    q_vecs = encoder.encode_queries(q_texts, instruction=query_instruction)
    scores, indices = index.search(q_vecs, top_k=top_k)

    results = []
    for i, (q_text, q_meta) in enumerate(queries):
        hits = []
        for k in range(scores.shape[1]):
            doc_idx = int(indices[i, k])
            # 동적 메타데이터를 결과에 포함
            hit_data = {
                "rank": k + 1,
                "score": float(scores[i, k]),
                "doc_id": vectordb.doc_ids[doc_idx],
                **vectordb.metadata[doc_idx],
            }
            hits.append(hit_data)
        results.append({"query": q_text, "query_meta": q_meta, "hits": hits})

    return {
        "id": q_item.qid,
        "model_name": model_name,
        "retrieval_results": results,
        "meta": q_item.meta,
    }


def main():
    """Main function to parse arguments and run the retrieval pipeline."""
    p = argparse.ArgumentParser(description="Dense Retrieval with a configured model.")
    p.add_argument(
        "--vectordb_csv", required=False, help="Path to the CSV vector database."
    )
    p.add_argument(
        "--schema_json",
        required=True,
        help="Path to the JSON schema for the VectorDB CSV.",
    )
    p.add_argument(
        "--config_json", required=True, help="Path to the query encoder config JSON."
    )
    p.add_argument(
        "--questions_jsonl", required=True, help="Path to the questions JSONL file."
    )
    p.add_argument(
        "--top_k", type=int, default=10, help="Number of documents to retrieve."
    )
    p.add_argument(
        "--output_root",
        default="../results/retrieval_docs",
        help="Root directory for output files.",
    )
    p.add_argument(
        "--output_subdir",
        default=None,
        help="Optional output subdirectory. If omitted, timestamp subdirectory is used.",
    )
    # --- Arguments for question selection ---
    p.add_argument("--ids", help="Comma-separated question IDs to process.")
    p.add_argument("--idx", help="Comma-separated 1-based indices to process.")
    p.add_argument(
        "--range", dest="range_spec", help="1-based inclusive range like '1-10'."
    )
    # --- Arguments for encoder model ---
    p.add_argument(
        "--device",
        default="auto",
        help="Device for encoding ('auto', 'cpu', 'cuda:0').",
    )
    p.add_argument(
        "--query_instruction",
        default="",
        help="Custom instruction for the query encoder.",
    )
    args = p.parse_args()

    # --- Load Data and Models ---
    print("[INFO] Loading resources...", file=sys.stderr)
    with open(args.config_json, "r", encoding="utf-8") as f:
        config = json.load(f)
    model_name = config.get("model_name")
    if not model_name:
        raise ValueError("Config JSON must contain a 'model_name'.")
    if args.vectordb_csv is None:
        args.vectordb_csv = config["output_file"]
    print(args.vectordb_csv)
    vectordb = data_loader.load_vectordb_from_csv(args.vectordb_csv, args.schema_json)
    all_questions = data_loader.load_questions_jsonl(args.questions_jsonl)
    encoder = query_encoder.QueryEncoder(model_name=model_name, device=args.device)
    retriever = get_retriever(vectordb.embeddings)
    print(f"[INFO] Resources loaded successfully.", file=sys.stderr)

    # --- Select Questions ---
    selected_questions = data_loader.select_questions(
        all_questions,
        ids=args.ids.split(",") if args.ids else None,
        idxs=[int(i) for i in args.idx.split(",")] if args.idx else None,
        range_spec=args.range_spec,
    )
    print(
        f"[INFO] Selected {len(selected_questions)} question(s) to process.",
        file=sys.stderr,
    )

    # --- Run Retrieval and Save Results ---
    if not selected_questions:
        print("[WARN] No questions to process. Exiting.", file=sys.stderr)
        return

    saver = result_saver.ResultSaver(args.output_root, output_subdir=args.output_subdir)
    saved_files = []

    for q_item in selected_questions:
        payload = run_retrieval_for_question(
            q_item,
            encoder,
            retriever,
            vectordb,
            args.top_k,
            model_name,
            args.query_instruction,
        )
        saved_path = saver.save_result(payload)
        saved_files.append(saved_path)
        print(
            f"[OK] Saved result for QID {q_item.qid} to {saved_path}", file=sys.stderr
        )

    print("\n--- Retrieval Complete ---")
    print(f"Output folder: {saver.get_output_folder()}")
    print("Saved files:")
    for path in saved_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
