# embedding_processor.py
from typing import List, Dict, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


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
