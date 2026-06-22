"""bge-m3 임베딩 모듈.

청크 텍스트를 벡터로 변환한다.
"""

from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer

_DEFAULT_MODEL = "BAAI/bge-m3"


def _resolve_device() -> str:
    """실행 환경에 맞는 device를 고른다.

    CUDA GPU가 있으면 "cuda"(박스/Colab), 없으면 "cpu"(맥북)로 떨어진다.
    덕분에 맥북에서는 무거운 모델이 GPU를 쓰지 않는다.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embedding_model(model_name: str = _DEFAULT_MODEL) -> SentenceTransformer:
    """임베딩 모델을 로드한다.

    Args:
        model_name: HuggingFace 모델 이름

    Returns:
        SentenceTransformer 객체
    """
    return SentenceTransformer(model_name, device=_resolve_device())


def embed_texts(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> list[list[float]]:
    """텍스트 리스트를 임베딩 벡터로 변환한다.

    Args:
        texts: 임베딩할 텍스트 리스트
        model: SentenceTransformer 모델
        batch_size: 배치 크기

    Returns:
        임베딩 벡터 리스트
    """
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings.tolist()
