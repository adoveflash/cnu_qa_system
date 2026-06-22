"""Advanced RAG 구성요소 — BM25 희소검색, Reranker, RRF 융합.

모드별로 무거운 모델/인덱스를 lazy 로딩한다 (naive 모드에선 아무것도 안 띄움).

- BM25Index: 코퍼스 문서에 대한 희소(키워드) 검색 — 정확 용어(메뉴명·정류장명) 강함
- Reranker: bge-reranker-v2-m3 cross-encoder로 후보 재정렬 — 정밀도 향상
- rrf_fuse: dense + sparse 결과를 Reciprocal Rank Fusion으로 융합
"""

from __future__ import annotations

import re
from typing import Any

_CJK = re.compile(r"[가-힣一-鿿]")
_WORD = re.compile(r"[A-Za-z0-9]+")


def tokenize_ko(text: str) -> list[str]:
    """한국어 친화 간이 토크나이저.

    영문/숫자는 단어 단위, 한글/한자는 문자 바이그램으로 쪼갠다.
    (형태소 분석기 없이도 BM25가 한국어에서 동작하도록)
    """
    text = text.lower()
    tokens = _WORD.findall(text)
    cjk = "".join(_CJK.findall(text))
    tokens += [cjk[i : i + 2] for i in range(len(cjk) - 1)]  # 문자 바이그램
    return tokens


class BM25Index:
    """코퍼스 문서에 대한 BM25 희소검색 인덱스."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        """인덱스를 구축한다.

        Args:
            docs: [{chunk_id, text, url, title, source}, ...]
        """
        from rank_bm25 import BM25Okapi

        self.docs = docs
        self._bm25 = BM25Okapi([tokenize_ko(d["text"]) for d in docs])

    def search(self, query: str, n: int) -> list[dict[str, Any]]:
        """질문과 BM25 점수가 높은 상위 n개 문서를 반환한다."""
        scores = self._bm25.get_scores(tokenize_ko(query))
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked[:n]:
            d = dict(self.docs[i])
            d["bm25_score"] = float(scores[i])
            out.append(d)
        return out


class Reranker:
    """bge-reranker-v2-m3 cross-encoder 재정렬기 (lazy 로딩)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        self._model = None

    def _ensure(self) -> None:
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = CrossEncoder(self.model_name, device=device)

    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """후보를 (질문, 문서) 쌍으로 점수화해 상위 top_k를 반환한다."""
        if not candidates:
            return []
        self._ensure()
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


def rrf_fuse(
    result_lists: list[list[dict[str, Any]]], top_k: int, k: int = 60
) -> list[dict[str, Any]]:
    """여러 검색 결과를 Reciprocal Rank Fusion으로 융합한다.

    각 문서의 점수 = Σ 1/(k + rank). chunk_id로 동일 문서를 합친다.

    Args:
        result_lists: 융합할 결과 리스트들 (각각 순위대로 정렬됨)
        top_k: 반환할 문서 수
        k: RRF 상수 (기본 60)

    Returns:
        융합 점수 내림차순 상위 top_k 문서
    """
    fused: dict[str, dict[str, Any]] = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc.get("chunk_id") or doc.get("url", "") + doc["text"][:40]
            if key not in fused:
                fused[key] = dict(doc)
                fused[key]["rrf_score"] = 0.0
            fused[key]["rrf_score"] += 1.0 / (k + rank)
    ranked = sorted(fused.values(), key=lambda d: d["rrf_score"], reverse=True)
    return ranked[:top_k]
