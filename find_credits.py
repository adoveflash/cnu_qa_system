"""코퍼스에서 '졸업 총 학점' 관련 문장을 찾는다 (모델 로드 없음).

N학점이 졸업/총/이수/전공/교양/최소 키워드와 함께 나오는 줄을 출력.
총 졸업학점 정보가 코퍼스에 실제로 있는지 확인용.

사용법: python find_credits.py
"""

from __future__ import annotations

import re

import chromadb

NUM_CREDIT = re.compile(r"\d+\s*학점")
CONTEXT_KW = ("졸업", "총", "이수", "전공", "교양", "최소", "기준학점")


def main() -> None:
    col = chromadb.PersistentClient(path="data/vector_db").get_collection("cnu_chunks")
    g = col.get(include=["documents", "metadatas"])
    hits = 0
    seen: set[str] = set()
    for d, m in zip(g["documents"], g["metadatas"]):
        for line in d.splitlines():
            line = line.strip()
            if not line or line in seen:
                continue
            if NUM_CREDIT.search(line) and any(k in line for k in CONTEXT_KW):
                seen.add(line)
                print(f"[{m.get('source','?'):9s}] {line[:140]}")
                hits += 1
    print(f"\n--- 매칭된 '학점' 문장: {hits}개")


if __name__ == "__main__":
    main()
