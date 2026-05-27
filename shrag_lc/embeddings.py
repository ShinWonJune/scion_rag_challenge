"""Embeddings: the 3*title+abstract text strategy + a HuggingFaceEmbeddings factory.

Ported from:
  * ``utils/load_jsonl_and_make_text_for_embedding.py`` (embedding-text modes)
  * ``features/embedding_processor.py`` (gte fp16, batch=32, normalize, max_len=512)
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import PipelineConfig


def build_embedding_text(doc: dict, embedding_mode: str = "3*title+abstract") -> str:
    """Construct the text to embed from a document dict.

    Mirrors ``make_text_for_embedding`` — the default 3T+A repeats the title
    three times so the (dominant) title signal is front-loaded.
    """
    title = (doc.get("title") or "").strip()
    abstract = (doc.get("abstract") or "").strip()

    if embedding_mode in ("T", "title"):
        return title
    if embedding_mode in ("A", "abstract"):
        return abstract
    if embedding_mode in ("T+A", "title+abstract"):
        return f"{title} {abstract}"
    if embedding_mode == "2T+A":
        return f"{title} {title} {abstract}"
    if embedding_mode in ("3T+A", "3*title+abstract"):
        return f"{title} {title} {title} {abstract}"
    if embedding_mode == "5T+A":
        return f"{title} {title} {title} {title} {title} {abstract}"
    raise ValueError(f"Unknown embedding_mode: {embedding_mode}")


def doc_to_document(doc: dict, embedding_mode: str = "3*title+abstract") -> Document:
    """Convert a common-schema search dict into a LangChain ``Document``.

    ``page_content`` is the embedding text (3T+A); raw fields are kept in
    metadata so retrieval/citation can reference doc_id, title, abstract.
    """
    doc_id = str(doc.get("doc_id") or doc.get("CN") or doc.get("cn") or "")
    return Document(
        page_content=build_embedding_text(doc, embedding_mode),
        metadata={
            "doc_id": doc_id,
            "title": doc.get("title", ""),
            "abstract": doc.get("abstract", ""),
            "source": doc.get("source", ""),
            "url": doc.get("url", ""),
        },
    )


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_embeddings(cfg: PipelineConfig) -> HuggingFaceEmbeddings:
    """SentenceTransformer-backed embeddings (gte fp16 b32 normalize len=512)."""
    device = _resolve_device(cfg.device)
    model_kwargs: dict = {"device": device, "trust_remote_code": True}
    if cfg.use_fp16 and device == "cuda":
        # Nested kwargs forwarded to the underlying HF AutoModel.
        model_kwargs["model_kwargs"] = {"torch_dtype": "float16"}

    emb = HuggingFaceEmbeddings(
        model_name=cfg.embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs={
            "batch_size": cfg.embedding_batch_size,
            "normalize_embeddings": True,
        },
    )
    # Truncate long inputs (BEST_CONFIG: max_seq_length=512, no retrieval impact).
    client = getattr(emb, "_client", None) or getattr(emb, "client", None)
    if client is not None and cfg.embedding_max_seq_length:
        try:
            client.max_seq_length = cfg.embedding_max_seq_length
        except Exception:
            pass
    return emb
