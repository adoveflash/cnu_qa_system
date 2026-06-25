"""벡터DB 재구축: chunks.jsonl → cnu_chunks 컬렉션 재인덱싱.

cnu_chunks를 삭제 후 재생성한다 (cnu_live는 건드리지 않음).
재빌드 후에는 add_manual.py로 큐레이션 청크를 다시 넣어야 한다.

사용법: CUDA_VISIBLE_DEVICES=0 python build_db.py
"""

from __future__ import annotations

from pathlib import Path

from src.rag.vector_store import build_index

if __name__ == "__main__":
    n = build_index(Path("data/corpus/chunks.jsonl"), Path("data/vector_db"))
    print(f"인덱싱 완료: {n}개")
