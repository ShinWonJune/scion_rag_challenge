"""Compute hit@k metrics for ScienceON platform search outputs.

Two modes:

1. raw  — hit@k uses ScienceON's returned order (platform native ranking).
2. gte  — hit@k uses gte-multilingual-base dense reranking of the platform set.

Inputs:
  --search-meta  search_meta_results.json from `pipeline.step1_search`
  --gold         gold JSON `{qid: [doc_id, ...]}`
  --ks           k values to compute (default: 1 3 5 10)

For each gold qid, hit@k = 1 if any gold doc appears in top-k of that qid's
ranked candidate list, else 0. Reported metrics average over qids.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_search_meta(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[dict[str, Any]]] = {}
    for row in payload.get("results", []):
        qid = str(row.get("question_id"))
        docs = row.get("documents") or []
        out[qid] = docs
    return out


def load_gold(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): {str(v).strip() for v in (vs if isinstance(vs, list) else [vs])}
            for k, vs in payload.items()}


def doc_id_of(doc: dict[str, Any]) -> str:
    return str(doc.get("doc_id") or doc.get("CN") or doc.get("cn") or "").strip()


def hit_at_k(ranked_doc_ids: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return 1.0 if any(d in gold for d in ranked_doc_ids[:k]) else 0.0


def rerank_with_gte(
    docs_by_qid: dict[str, list[dict[str, Any]]],
    queries_by_qid: dict[str, str],
    encoder_config_path: Path,
) -> dict[str, list[str]]:
    """Encode question + each candidate doc with gte-multilingual-base, return doc_id list ranked by cosine similarity."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    cfg = json.loads(encoder_config_path.read_text(encoding="utf-8"))
    model_name = cfg["model_name"]
    truncate_dim = cfg.get("embedding_dim")
    embedding_mode = cfg.get("embedding_mode", "title+abstract")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading encoder %s on %s (truncate_dim=%s, mode=%s)",
                model_name, device, truncate_dim, embedding_mode)
    model = SentenceTransformer(model_name, trust_remote_code=True,
                                truncate_dim=truncate_dim, device=device)

    def doc_text(doc: dict[str, Any]) -> str:
        title = (doc.get("title") or "").strip()
        abstract = (doc.get("abstract") or doc.get("text") or "").strip()
        if embedding_mode == "3*title+abstract":
            return f"{title} {title} {title} {abstract}".strip()
        return f"{title} {abstract}".strip()

    reranked: dict[str, list[str]] = {}
    for qid, docs in docs_by_qid.items():
        if not docs:
            reranked[qid] = []
            continue
        query = queries_by_qid.get(qid, "")
        q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        d_texts = [doc_text(d) for d in docs]
        d_embs = model.encode(d_texts, normalize_embeddings=True, convert_to_numpy=True,
                              batch_size=32, show_progress_bar=False)
        sims = (q_emb @ d_embs.T).flatten()  # (n_docs,)
        order = np.argsort(-sims)
        reranked[qid] = [doc_id_of(docs[i]) for i in order]
    return reranked


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Compute hit@k for ScienceON platform search output")
    parser.add_argument("--search-meta", required=True, help="search_meta_results.json from step1_search")
    parser.add_argument("--gold", required=True, help="gold JSON {qid:[doc_id]}")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--rerank", choices=["none", "gte"], default="none",
                        help="Optional dense rerank before computing hit@k")
    parser.add_argument("--encoder-config", default="configs/query_encoder/config_gte-multilingual-base.json",
                        help="Encoder config used when --rerank gte")
    parser.add_argument("--output", required=True, help="Output JSON metrics path")
    parser.add_argument("--label", default="", help="Free-form label written into output (e.g. 'target5_raw')")
    args = parser.parse_args()

    docs_by_qid = load_search_meta(Path(args.search_meta))
    gold = load_gold(Path(args.gold))
    common_qids = sorted(set(docs_by_qid) & set(gold), key=lambda x: int(x) if x.isdigit() else x)
    logger.info("docs_by_qid=%d qids, gold=%d qids, common=%d qids",
                len(docs_by_qid), len(gold), len(common_qids))

    queries_by_qid: dict[str, str] = {}
    payload = json.loads(Path(args.search_meta).read_text(encoding="utf-8"))
    for row in payload.get("results", []):
        queries_by_qid[str(row.get("question_id"))] = row.get("query", "")

    if args.rerank == "gte":
        only_common = {qid: docs_by_qid[qid] for qid in common_qids}
        ranked = rerank_with_gte(only_common, queries_by_qid, Path(args.encoder_config))
    else:
        ranked = {qid: [doc_id_of(d) for d in docs_by_qid[qid]] for qid in common_qids}

    metrics: dict[str, Any] = {
        "label": args.label,
        "rerank": args.rerank,
        "n_qids_common": len(common_qids),
        "candidate_count_stats": {
            "min": min(len(docs_by_qid[q]) for q in common_qids) if common_qids else 0,
            "max": max(len(docs_by_qid[q]) for q in common_qids) if common_qids else 0,
            "mean": (sum(len(docs_by_qid[q]) for q in common_qids) / len(common_qids))
                    if common_qids else 0,
        },
    }
    per_qid: dict[str, dict[str, int]] = {}
    for qid in common_qids:
        per_qid[qid] = {}
    for k in args.ks:
        hits = []
        for qid in common_qids:
            h = hit_at_k(ranked[qid], gold[qid], k)
            per_qid[qid][f"hit_at_{k}"] = int(h)
            hits.append(h)
        metrics[f"hit_at_{k}"] = round(sum(hits) / len(hits), 6) if hits else 0.0
        metrics[f"hit_at_{k}_count"] = int(sum(hits))

    metrics["per_qid"] = per_qid

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== hit@k report ===")
    print(f"label    : {args.label}")
    print(f"rerank   : {args.rerank}")
    print(f"n_qids   : {len(common_qids)}")
    print(f"cand     : min={metrics['candidate_count_stats']['min']} "
          f"max={metrics['candidate_count_stats']['max']} "
          f"mean={metrics['candidate_count_stats']['mean']:.2f}")
    for k in args.ks:
        print(f"hit@{k:<3}: {metrics[f'hit_at_{k}']:.4f}  ({metrics[f'hit_at_{k}_count']}/{len(common_qids)})")
    print(f"wrote    : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
