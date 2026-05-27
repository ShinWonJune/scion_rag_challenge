"""shrag_lc — LangChain reimplementation of the SHRAG pipeline.

Pipeline: question -> keyword extraction (KO/EN) -> platform search
(ScienceON/PubMed/Wikipedia) -> 3T+A embedding -> FAISS dense retrieval
-> cross-encoder rerank -> grounded LLM answer.

Mirrors the original ``shrag`` package using LangChain idioms (LCEL,
Runnable, VectorStore, Retriever).
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
