# embedding_processor.py
import time
from typing import List, Dict, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from shrag.utils.embedding_cache import EmbeddingCache, make_encoder_id


def generate_batch_embeddings(
    documents_data: List[Dict],
    model_name: str,
    truncate_dimension: Optional[int] = None,
    gpu_id: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    SentenceTransformer 모델을 로드하고 주어진 문서들의 임베딩을 생성합니다.

    Args:
        documents_data (List[Dict]): 각 딕셔너리가 문서를 나타내는 리스트.
        model_name (str): 사용할 SentenceTransformer 모델의 이름 또는 경로.
        truncate_dimension (Optional[int]): 임베딩을 잘라낼 차원.
                                            None일 경우 모델의 기본 출력 차원을 사용합니다.
        gpu_id (Optional[int]): 사용할 GPU ID (예: 0, 1, 2, 3).
                               None일 경우 자동으로 디바이스를 선택합니다.

    Returns:
        Optional[np.ndarray]: 문서 임베딩의 numpy 배열. 실패 시 None을 반환합니다.
    """
    if not documents_data:
        print("⚠️ 임베딩을 생성할 문서가 없습니다.")
        return None

    # 트러케이션(truncation) 적용 여부 안내
    if truncate_dimension:
        print(
            f"모델 로딩: {model_name} (임베딩 차원을 {truncate_dimension}으로 조절합니다)"
        )
    else:
        print(f"모델 로딩: {model_name}")

    try:
        # GPU 디바이스 설정
        if gpu_id is not None:
            # 특정 GPU ID가 지정된 경우
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                device = f"cuda:{gpu_id}"
                print(f"지정된 GPU 사용: {device}")
                
                # PyTorch CUDA 메모리 설정 최적화
                import os
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                
                # 지정된 GPU로 컨텍스트 설정
                torch.cuda.set_device(device)
                torch.cuda.empty_cache()  # 기존 캐시 정리
            else:
                print(f"경고: GPU {gpu_id}를 사용할 수 없습니다. CPU를 사용합니다.")
                device = "cpu"
        else:
            # GPU ID가 지정되지 않은 경우 자동 선택
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"자동 선택된 디바이스: {device}")
        
        # 모델 로드 시 truncate_dim 파라미터 전달
        model = SentenceTransformer(
            model_name, trust_remote_code=True, truncate_dim=truncate_dimension, device=device
        )
    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생 ({model_name}): {e}")
        return None

    # 임베딩할 텍스트 추출
    texts_to_embed = [doc.get("embedding_text", "") for doc in documents_data]
    if not any(texts_to_embed):
        print(
            "⚠️ 경고: 모든 문서에서 'embedding_text' 키를 찾을 수 없거나 값이 비어있습니다."
        )
        return None

    print("문서 인코딩 중...")
    embeddings = model.encode(
        texts_to_embed, convert_to_numpy=True, show_progress_bar=True
    )

    if embeddings is not None:
        print(f"✅ 임베딩 생성 완료! (최종 차원: {embeddings.shape[1]})")

    return embeddings


def generate_batch_embeddings_cached(
    documents_data: List[Dict],
    model_name: str,
    embedding_mode: str,
    truncate_dimension: Optional[int] = None,
    gpu_id: Optional[int] = None,
    cache: Optional[EmbeddingCache] = None,
) -> Optional[np.ndarray]:
    """Cached variant of generate_batch_embeddings.

    Looks up (encoder_id, mode, doc_id, text_sha1) in cache before encoding.
    Encodes only missing entries, stores them, returns full vector matrix in input order.

    Falls back to full encoding when cache is None.

    Args:
        documents_data: each dict needs 'cn' (doc_id) and 'embedding_text'
        model_name: sentence-transformers model name
        embedding_mode: e.g. "3T+A" — for cache key separation across modes
        truncate_dimension: model output dim (Matryoshka)
        gpu_id: optional CUDA device id
        cache: EmbeddingCache instance (or None for no caching)
    """
    if not documents_data:
        print("⚠️ 임베딩을 생성할 문서가 없습니다.")
        return None

    if cache is None:
        return generate_batch_embeddings(
            documents_data, model_name, truncate_dimension, gpu_id=gpu_id
        )

    encoder_id = make_encoder_id(model_name, truncate_dimension)
    doc_ids = [str(d.get("cn") or d.get("doc_id") or "") for d in documents_data]
    texts = [str(d.get("embedding_text", "")) for d in documents_data]

    cached_by_idx, missing_idx = cache.get_batch(encoder_id, embedding_mode, doc_ids, texts)

    n_total = len(documents_data)
    n_cached = len(cached_by_idx)
    n_missing = len(missing_idx)
    print(
        f"[cache] hits={n_cached}/{n_total} miss={n_missing} encoder={encoder_id} mode={embedding_mode}"
    )

    # Determine dim from cached vector if any, else infer after encoding
    dim_from_cache: Optional[int] = None
    if cached_by_idx:
        any_idx = next(iter(cached_by_idx))
        dim_from_cache = int(cached_by_idx[any_idx].shape[0])

    # Encode missing
    new_vectors: Optional[np.ndarray] = None
    if missing_idx:
        miss_docs = [documents_data[i] for i in missing_idx]
        t0 = time.perf_counter()
        new_vectors = generate_batch_embeddings(
            miss_docs, model_name, truncate_dimension, gpu_id=gpu_id
        )
        elapsed = time.perf_counter() - t0
        if new_vectors is None:
            print("❌ 미스 문서 인코딩 실패")
            return None
        print(f"[cache] encoded {n_missing} missing in {elapsed:.1f}s")
        # Store new
        miss_doc_ids = [doc_ids[i] for i in missing_idx]
        miss_texts = [texts[i] for i in missing_idx]
        n_written = cache.put_batch(encoder_id, embedding_mode, miss_doc_ids, miss_texts, new_vectors)
        print(f"[cache] wrote {n_written} new entries")

    # Determine final dim
    dim = dim_from_cache if dim_from_cache else int(new_vectors.shape[1])

    # Assemble in input order
    out = np.zeros((n_total, dim), dtype=np.float32)
    for idx, vec in cached_by_idx.items():
        out[idx] = vec
    if new_vectors is not None:
        for j, idx in enumerate(missing_idx):
            out[idx] = new_vectors[j]
    return out
