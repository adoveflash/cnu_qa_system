"""RAG 검색기 모듈.

질문을 받아 관련 컨텍스트를 반환한다.
static(고정 정보) + live(실시간 갱신 정보) 두 컬렉션을 동시에 검색한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

import chromadb

from src.rag.embedder import embed_texts, load_embedding_model

_DEFAULT_DB_PATH = Path("data/vector_db")
_STATIC_COLLECTION = "cnu_chunks"
_LIVE_COLLECTION = "cnu_live"


class Retriever:
    """벡터 DB 검색기. static + live 컬렉션을 동시에 검색한다."""

    def __init__(
        self,
        db_path: Path = _DEFAULT_DB_PATH,
        static_collection: str = _STATIC_COLLECTION,
        live_collection: str = _LIVE_COLLECTION,
        embedding_model_name: str = "BAAI/bge-m3",
        top_k: int = 5,
    ) -> None:
        """검색기를 초기화한다.

        Args:
            db_path: ChromaDB 경로
            static_collection: 고정 정보 컬렉션 이름
            live_collection: 실시간 갱신 컬렉션 이름
            embedding_model_name: 임베딩 모델명
            top_k: 기본 검색 결과 수
        """
        self.model: SentenceTransformer = load_embedding_model(embedding_model_name)
        client = chromadb.PersistentClient(path=str(db_path))

        self.static = client.get_collection(static_collection)

        # live 컬렉션은 없을 수 있음 (갱신 전)
        try:
            self.live = client.get_collection(live_collection)
        except Exception:
            self.live = None

        self.top_k = top_k

    def _query_collection(
        self, collection: Any, query_embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        """단일 컬렉션에서 검색한다."""
        count = collection.count()
        if count == 0:
            return []

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            output.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "url": meta.get("url", ""),
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "distance": results["distances"][0][i],
                }
            )
        return output

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """static + live 컬렉션에서 질문과 유사한 청크를 검색한다.

        두 컬렉션 결과를 합쳐 거리순으로 정렬 후 top_k개 반환한다.

        Args:
            query: 검색 질문
            top_k: 반환할 결과 수 (None이면 기본값 사용)

        Returns:
            [{text, url, title, source, distance}, ...]
        """
        k = top_k or self.top_k
        query_embedding = embed_texts([query], self.model)[0]

        # static 컬렉션 검색
        results = self._query_collection(self.static, query_embedding, k)

        # live 컬렉션 검색 (있으면)
        if self.live is not None:
            live_results = self._query_collection(self.live, query_embedding, k)
            results.extend(live_results)

        # 거리순 정렬 후 top_k
        results.sort(key=lambda x: x["distance"])
        return results[:k]

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
            if r["url"] and r["url"] not in urls:
                urls.append(r["url"])

        return "\n\n".join(context_parts), urls
