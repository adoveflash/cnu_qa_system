"""RAG 검색기 모듈.

질문을 받아 관련 컨텍스트를 반환한다.
static(고정 정보) + live(실시간 갱신 정보) 두 컬렉션을 동시에 검색한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

import chromadb

from src.rag.embedder import load_embedding_model

_DEFAULT_DB_PATH = Path("data/vector_db")
_STATIC_COLLECTION = "cnu_chunks"
_LIVE_COLLECTION = "cnu_live"

# 질문 키워드 → source 매핑 (학과 부스팅용)
# 값은 부스팅할 source 리스트 (manual은 수동 추가 데이터이므로 항상 포함)
_DEPARTMENT_KEYWORDS: dict[str, list[str]] = {
    "computer": [
        "컴퓨터",
        "컴융",
        "컴공",
        "소프트웨어",
        "인공지능학부",
        "컴퓨터융합",
        "컴퓨터인공지능",
        "SW",
        "AI학부",
    ],
    "plus_kr": [
        "학교",
        "충남대",
        "대학교",
        "캠퍼스",
    ],
    "job": [
        "취업",
        "인재개발원",
        "채용",
        "진로",
    ],
    # 장학금: 학과 페이지(약학대·경영 등)가 아니라 공식 안내(plus_kr)를 우선한다.
    "scholarship": [
        "장학",
        "장학금",
        "장학생",
        "등록금 지원",
    ],
}

# source 그룹: 특정 source 감지 시 함께 부스팅할 source 목록
_SOURCE_GROUP: dict[str, list[str]] = {
    "computer": ["computer", "manual"],
    "plus_kr": ["plus_kr", "manual"],
    "job": ["job"],
    "scholarship": ["plus_kr", "job", "manual"],
}

# source 부스팅 가중치 (distance에서 이만큼 차감)
# bge-m3 cosine distance(0~2) 기준. 0.08은 약해 학과 청크 오염을 못 막아 0.15로 상향.
_BOOST_WEIGHT = float(os.environ.get("RAG_BOOST_WEIGHT", "0.15"))

# 제출 파이프라인 기본 검색 모드. naive는 컷오프·재정렬이 없어 무관 청크가 출처로 샌다.
# rerank(cross-encoder)를 기본으로 두되, T4 메모리 부족 시 RAG_MODE=naive로 폴백 가능.
_DEFAULT_MODE = os.environ.get("RAG_MODE", "rerank")

# cross-encoder rerank_score 컷오프. 이 값 미만 청크는 컨텍스트·출처에서 제외한다.
# bge-reranker-v2-m3은 raw logit을 내며 양수≈관련/음수≈무관(sigmoid 0.5 기준 0.0).
# 무관 청크가 출처 배지로 새는 것을 막는 핵심 레버. RAG_RERANK_MIN으로 튜닝.
_RERANK_SCORE_MIN = float(os.environ.get("RAG_RERANK_MIN", "0.0"))


def _detect_sources(query: str) -> list[str]:
    """질문에서 키워드를 감지하여 부스팅할 source 리스트를 반환한다."""
    for source, keywords in _DEPARTMENT_KEYWORDS.items():
        for kw in keywords:
            if kw in query:
                return _SOURCE_GROUP.get(source, [source])
    return []


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

        # advanced RAG 구성요소 (lazy 로딩 캐시)
        self._bm25 = None
        self._reranker = None
        self._all_docs = None

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

    def _dense(self, query: str, fetch_k: int) -> list[dict[str, Any]]:
        """dense(bge-m3) 검색 — static+live 병합 + 학과 부스팅 후 fetch_k개."""
        query_embedding = self.model.encode(query).tolist()

        results = self._query_collection(self.static, query_embedding, fetch_k)
        if self.live is not None:
            results.extend(self._query_collection(self.live, query_embedding, fetch_k))

        preferred_sources = _detect_sources(query)
        if preferred_sources:
            for r in results:
                if r["source"] in preferred_sources:
                    r["distance"] = max(0, r["distance"] - _BOOST_WEIGHT)

        results.sort(key=lambda x: x["distance"])
        return results[:fetch_k]

    def _all_documents(self) -> list[dict[str, Any]]:
        """BM25용 — 전체 코퍼스 문서를 1회 적재해 캐싱한다."""
        if self._all_docs is None:
            docs: list[dict[str, Any]] = []
            for col in [self.static, self.live]:
                if col is None:
                    continue
                got = col.get(include=["documents", "metadatas"])
                for cid, text, meta in zip(got["ids"], got["documents"], got["metadatas"]):
                    docs.append(
                        {
                            "chunk_id": cid,
                            "text": text,
                            "url": meta.get("url", ""),
                            "title": meta.get("title", ""),
                            "source": meta.get("source", ""),
                            "distance": 1.0,
                        }
                    )
            self._all_docs = docs
        return self._all_docs

    def _bm25_search(self, query: str, n: int) -> list[dict[str, Any]]:
        """BM25 희소검색 (lazy 인덱스 구축)."""
        if self._bm25 is None:
            from src.rag.advanced import BM25Index

            self._bm25 = BM25Index(self._all_documents())
        return self._bm25.search(query, n)

    def _rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """cross-encoder 재정렬 (lazy 모델 로딩)."""
        if self._reranker is None:
            from src.rag.advanced import Reranker

            self._reranker = Reranker()
        return self._reranker.rerank(query, candidates, top_k)

    @staticmethod
    def _apply_cutoff(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """rerank_score가 컷오프 미만인 무관 청크를 제거한다.

        무관 청크가 컨텍스트뿐 아니라 출처 배지(약학대·경영 등)로 새는 것을 막는다.
        전부 컷오프되면 빈 리스트를 반환해 "확인되지 않은 정보" 정직 응답으로 유도한다
        (과거 RAG 데이터로 잘못된 정보를 단정하는 것보다 낫다).
        """
        return [r for r in results if r.get("rerank_score", 1.0) >= _RERANK_SCORE_MIN]

    def retrieve(
        self, query: str, top_k: int | None = None, mode: str | None = None
    ) -> list[dict[str, Any]]:
        """질문과 유사한 청크를 검색한다 (RAG 모드 선택 가능).

        모드:
            - "naive": dense(bge-m3) top_k
            - "hybrid": dense + BM25 RRF 융합
            - "rerank": dense 후보를 cross-encoder로 재정렬 + score 컷오프 (기본)
            - "hybrid_rerank": hybrid 후보를 재정렬 + score 컷오프

        rerank 계열 모드는 _RERANK_SCORE_MIN 미만 청크를 탈락시켜 출처 오염을 막는다.

        Args:
            query: 검색 질문
            top_k: 반환할 결과 수 (None이면 기본값)
            mode: 검색 모드 (None이면 _DEFAULT_MODE — 환경변수 RAG_MODE, 기본 rerank)

        Returns:
            [{text, url, title, source, distance, ...}, ...]
        """
        k = top_k or self.top_k
        mode = mode or _DEFAULT_MODE

        if mode == "naive":
            return self._dense(query, k * 2)[:k]

        from src.rag.advanced import rrf_fuse

        cand_n = max(k * 4, 20)  # 재정렬/융합용 넓은 후보
        dense = self._dense(query, cand_n)

        if mode == "rerank":
            return self._apply_cutoff(self._rerank(query, dense, k))

        bm25 = self._bm25_search(query, cand_n)
        fused = rrf_fuse([dense, bm25], top_k=cand_n)

        if mode == "hybrid":
            return fused[:k]
        if mode == "hybrid_rerank":
            return self._apply_cutoff(self._rerank(query, fused, k))

        # 알 수 없는 모드는 naive로 폴백
        return dense[:k]

    def build_context(
        self, query: str, top_k: int | None = None, mode: str | None = None
    ) -> tuple[str, list[str]]:
        """질문에 대한 컨텍스트 문자열과 출처 URL 목록을 생성한다.

        Args:
            query: 검색 질문
            top_k: 검색 결과 수
            mode: RAG 검색 모드 (None이면 _DEFAULT_MODE — naive/hybrid/rerank/hybrid_rerank)

        Returns:
            (컨텍스트 문자열, 출처 URL 리스트) 튜플. 관련 청크가 없으면 ("", []).
        """
        results = self.retrieve(query, top_k, mode=mode)
        if not results:
            return "", []

        context_parts = []
        urls = []
        for i, r in enumerate(results, 1):
            context_parts.append(f"[참고{i}] {r['title']}\n{r['text']}")
            if r["url"] and r["url"] not in urls:
                urls.append(r["url"])

        return "\n\n".join(context_parts), urls
