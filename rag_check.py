"""RAG 점검 스크립트.

사용법 (레포 루트에서):
    python rag_check.py              # 1단계: 벡터DB 통계 (모델 로드 없음, 빠름)
    python rag_check.py --retrieve   # 2단계: bge-m3 로드 후 실제 검색 품질 확인

1단계: 컬렉션 청크 수 / source 분포 / 길이 통계 / URL 결측 / 중복
2단계: 5개 도메인 샘플 질문에 대해 top-k 검색 결과(거리·source·제목) 출력
"""

from __future__ import annotations

import collections
import sys

import chromadb

DB_PATH = "data/vector_db"


def db_stats() -> None:
    """벡터DB 컬렉션별 통계를 출력한다 (모델 로드 없음)."""
    client = chromadb.PersistentClient(path=DB_PATH)
    for c in client.list_collections():
        col = client.get_collection(c.name)
        n = col.count()
        print(f"\n{'='*50}\n[{c.name}] {n} chunks\n{'='*50}")
        if n == 0:
            continue
        got = col.get(include=["documents", "metadatas"])
        docs, metas = got["documents"], got["metadatas"]

        src = collections.Counter(m.get("source", "?") for m in metas)
        print("source 분포:")
        for k, v in src.most_common():
            print(f"  {k:12s} {v}")

        lens = sorted(len(d) for d in docs)
        print(f"청크 길이(자) min/med/max: {lens[0]} / {lens[len(lens)//2]} / {lens[-1]}")
        print(f"20자 미만 청크 : {sum(1 for d in docs if len(d.strip()) < 20)}")
        print(f"URL 없는 청크  : {sum(1 for m in metas if not m.get('url'))}")
        print(f"본문 완전중복  : {n - len(set(docs))}")

        # title 없는 청크
        no_title = sum(1 for m in metas if not m.get("title"))
        print(f"title 없는 청크: {no_title}")


# 5개 도메인 + 보강 영역 샘플 질문
SAMPLE_QUERIES = [
    ("졸업요건", "컴퓨터융합학부 졸업하려면 몇 학점 들어야 해?"),
    ("학교 공지사항", "장학금 신청 공지 알려줘"),
    ("학사일정", "이번 학기 수강신청 정정 기간 언제야?"),
    ("식단 안내", "오늘 학생회관 식당 점심 메뉴 뭐야?"),
    ("통학/셔틀버스", "셔틀버스 시간표 알려줘"),
    ("취업/진로", "인재개발원 취업 프로그램 뭐 있어?"),
]


def retrieve_test(top_k: int = 5) -> None:
    """실제 Retriever로 샘플 질문 검색 품질을 확인한다 (bge-m3 로드)."""
    print("bge-m3 로드 중... (수십 초 걸릴 수 있음)")
    from src.rag.retriever import Retriever

    retriever = Retriever(top_k=top_k)
    print(f"로드 완료. top_k={top_k}\n")

    for category, query in SAMPLE_QUERIES:
        print(f"\n{'='*60}\n[{category}] Q: {query}\n{'='*60}")
        results = retriever.retrieve(query, top_k=top_k)
        if not results:
            print("  ⚠️  검색 결과 없음")
            continue
        for i, r in enumerate(results, 1):
            title = (r.get("title") or "")[:30]
            text = r["text"][:70].replace("\n", " ")
            print(f"  {i}. dist={r['distance']:.3f} [{r['source']:10s}] {title}")
            print(f"     {text}...")


if __name__ == "__main__":
    if "--retrieve" in sys.argv:
        retrieve_test()
    else:
        db_stats()
        print("\n→ 검색 품질도 보려면: python rag_check.py --retrieve")
