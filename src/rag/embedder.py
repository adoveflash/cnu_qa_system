"""bge-m3 임베딩 모듈.

청크 텍스트를 벡터로 변환한다.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

_DEFAULT_MODEL = "BAAI/bge-m3"


def load_embedding_model(model_name: str = _DEFAULT_MODEL) -> SentenceTransformer:
    """임베딩 모델을 로드한다.

    Args:
        model_name: HuggingFace 모델 이름

    Returns:
        SentenceTransformer 객체
    """
    return SentenceTransformer(model_name, device="cpu")


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
    embeddings = model.encode(texts, batch_size=batch_size)
    return embeddings.tolist()
