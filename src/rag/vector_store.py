"""ChromaDB 벡터 스토어 모듈.

청크 임베딩을 ChromaDB에 저장하고 검색한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb

from src.rag.embedder import embed_texts, load_embedding_model

_DEFAULT_COLLECTION = "cnu_chunks"


def build_index(
    chunks_path: Path,
    db_path: Path,
    collection_name: str = _DEFAULT_COLLECTION,
    embedding_model_name: str = "BAAI/bge-m3",
    batch_size: int = 32,
) -> int:
    """chunks.jsonl을 읽어 ChromaDB 인덱스를 구축한다.

    Args:
        chunks_path: data/corpus/chunks.jsonl 경로
        db_path: data/vector_db/ 경로
        collection_name: ChromaDB 컬렉션 이름
        embedding_model_name: 임베딩 모델명
        batch_size: 임베딩 배치 크기

    Returns:
        인덱싱된 청크 수
    """
    chunks: list[dict[str, Any]] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    if not chunks:
        print("  청크가 없습니다.")
        return 0

    print(f"  임베딩 모델 로드 중: {embedding_model_name}")
    model = load_embedding_model(embedding_model_name)

    texts = [c["text"] for c in chunks]
    print(f"  {len(texts)}개 청크 임베딩 중...")
    embeddings = embed_texts(texts, model, batch_size=batch_size)

    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))

    # 기존 컬렉션 삭제 후 재생성 (전체 재인덱싱)
    try:
        client.delete_collection(collection_name)
    except (ValueError, chromadb.errors.NotFoundError):
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB는 배치 크기 제한이 있으므로 나눠서 추가
    add_batch = 500
    for i in range(0, len(chunks), add_batch):
        batch_chunks = chunks[i : i + add_batch]
        batch_embeddings = embeddings[i : i + add_batch]
        collection.add(
            ids=[c["chunk_id"] for c in batch_chunks],
            embeddings=batch_embeddings,
            documents=[c["text"] for c in batch_chunks],
            metadatas=[
                {"url": c["url"], "title": c["title"], "source": c["source"]} for c in batch_chunks
            ],
        )

    print(f"  인덱싱 완료: {collection.count()}개")
    return collection.count()


def search(
    query: str,
    db_path: Path,
    top_k: int = 5,
    collection_name: str = _DEFAULT_COLLECTION,
    embedding_model_name: str = "BAAI/bge-m3",
) -> list[dict[str, Any]]:
    """쿼리와 유사한 청크를 검색한다.

    Args:
        query: 검색 질문
        db_path: data/vector_db/ 경로
        top_k: 반환할 최대 결과 수
        collection_name: ChromaDB 컬렉션 이름
        embedding_model_name: 임베딩 모델명

    Returns:
        검색 결과 리스트 [{text, url, title, source, score}, ...]
    """
    model = load_embedding_model(embedding_model_name)
    query_embedding = embed_texts([query], model)[0]

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i in range(len(results["ids"][0])):
        output.append(
            {
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "url": results["metadatas"][0][i]["url"],
                "title": results["metadatas"][0][i]["title"],
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i],
            }
        )
    return output
