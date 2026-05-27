"""RAG 검색기 모듈.

질문을 받아 관련 컨텍스트를 반환한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

import chromadb

from src.rag.embedder import embed_texts, load_embedding_model

_DEFAULT_DB_PATH = Path("data/vector_db")
_DEFAULT_COLLECTION = "cnu_chunks"


class Retriever:
    """벡터 DB 검색기. 모델과 DB를 한 번만 로드하여 재사용한다."""

    def __init__(
        self,
        db_path: Path = _DEFAULT_DB_PATH,
        collection_name: str = _DEFAULT_COLLECTION,
        embedding_model_name: str = "BAAI/bge-m3",
        top_k: int = 5,
    ) -> None:
        """검색기를 초기화한다.

        Args:
            db_path: ChromaDB 경로
            collection_name: 컬렉션 이름
            embedding_model_name: 임베딩 모델명
            top_k: 기본 검색 결과 수
        """
        self.model: SentenceTransformer = load_embedding_model(embedding_model_name)
        client = chromadb.PersistentClient(path=str(db_path))
        self.collection = client.get_collection(collection_name)
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """질문과 유사한 청크를 검색한다.

        Args:
            query: 검색 질문
            top_k: 반환할 결과 수 (None이면 기본값 사용)

        Returns:
            [{text, url, title, source, distance}, ...]
        """
        k = top_k or self.top_k
        query_embedding = embed_texts([query], self.model)[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
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

    def build_context(self, query: str, top_k: int | None = None) -> tuple[str, list[str]]:
        """질문에 대한 컨텍스트 문자열과 출처 URL 목록을 생성한다.

        Args:
            query: 검색 질문
            top_k: 검색 결과 수

        Returns:
            (컨텍스트 문자열, 출처 URL 리스트) 튜플
        """
        results = self.retrieve(query, top_k)
        if not results:
            return "", []

        context_parts = []
        urls = []
        for i, r in enumerate(results, 1):
            context_parts.append(f"[참고{i}] {r['title']}\n{r['text']}")
            if r["url"] not in urls:
                urls.append(r["url"])

        return "\n\n".join(context_parts), urls
