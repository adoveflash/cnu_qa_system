"""검증된 졸업학점 요약을 manual 청크로 벡터DB에 추가한다.

코퍼스에 흩어져 있어 학부 부스팅에 묻히는 '총 졸업학점 130' 사실을
manual source(부스팅 대상)로 합쳐 넣어, 졸업 질문에 안정적으로 검색되게 한다.

모든 수치는 코퍼스에서 확인된 사실:
- 130학점 졸업 원칙: plus_kr (충남대 학칙)
- 교양 인정 48학점(2025~)/42학점(2013~2024): computer 졸업요건표
- 프로젝트 교과목 최소 3개 이수: computer 비교과 졸업요건

사용법 (레포 루트): python add_manual.py
"""

from __future__ import annotations

import chromadb

MANUAL_CHUNKS = [
    {
        "id": "manual_grad_credit_2026",
        "title": "컴퓨터융합학부 졸업 학점 안내 (총 130학점)",
        "url": "https://computer.cnu.ac.kr/computer/edu/requirements.do",
        "source": "manual",
        "text": (
            "충남대학교 컴퓨터융합학부 및 인공지능학과 졸업 학점 안내입니다. "
            "학사학위과정 졸업에 필요한 총 학점은 130학점을 원칙으로 합니다. "
            "교양과목은 졸업학점으로 최대 48학점까지 인정됩니다 "
            "(2025학년도 교육과정 적용자 기준이며, 2013~2024학년도 입학자는 42학점까지 인정). "
            "전공·트랙 요건으로는 프로젝트 교과목을 졸업 시까지 최소 3개 이상 이수해야 합니다. "
            "정확한 세부 졸업요건은 컴퓨터융합학부 졸업요건표를 확인하세요."
        ),
    },
]


def main() -> None:
    from src.rag.embedder import load_embedding_model

    print("bge-m3 로드 중...")
    model = load_embedding_model("BAAI/bge-m3")
    col = chromadb.PersistentClient(path="data/vector_db").get_collection("cnu_chunks")

    for c in MANUAL_CHUNKS:
        emb = model.encode(c["text"]).tolist()
        col.upsert(
            ids=[c["id"]],
            embeddings=[emb],
            documents=[c["text"]],
            metadatas=[{"url": c["url"], "title": c["title"], "source": c["source"]}],
        )
        print(f"✅ upsert: {c['id']} (source={c['source']})")

    print(f"\n현재 cnu_chunks 총 {col.count()}개")


if __name__ == "__main__":
    main()
