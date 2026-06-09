"""답변 생성(grounding) 점검 스크립트.

실제 파이프라인(Retriever + Gemma 4 + generate_answer)으로 샘플 질문에
답변을 생성하고, 검색된 출처와 답변·소요시간을 출력한다.

사용법 (레포 루트에서, 48GB GPU 1장 고정 권장):
    CUDA_VISIBLE_DEVICES=0 python answer_test.py

확인 포인트:
- 답변이 검색 컨텍스트에 근거하는가 (환각 없는가)
- thinking 쓰레기 토큰이 섞이는가
- 출처 배지가 붙는가
- 1건당 생성 시간
"""

from __future__ import annotations

import time

# RAG만으로 답하는지 보려고 tool(실시간 조회)은 끈다.
USE_TOOLS = False
TOP_K = 5

SAMPLE_QUERIES = [
    ("졸업요건", "컴퓨터융합학부 졸업하려면 몇 학점 들어야 해?"),
    ("학교 공지사항", "장학금 신청 공지 알려줘"),
    ("학사일정", "이번 학기 수강신청 정정 기간 언제야?"),
    ("식단 안내", "오늘 학생회관 식당 점심 메뉴 뭐야?"),
    ("통학/셔틀버스", "셔틀버스 시간표 알려줘"),
]


def main() -> None:
    from src.rag.retriever import Retriever

    print("bge-m3 로드 중...")
    retriever = Retriever(top_k=TOP_K)

    from src.model.inference import generate_answer, load_inference_model

    print("Gemma 4 로드 중... (4bit, 수십 초~수 분)")
    t0 = time.time()
    model, tokenizer = load_inference_model()
    print(f"모델 로드 완료 ({time.time()-t0:.0f}s)\n")

    for category, query in SAMPLE_QUERIES:
        print(f"\n{'='*70}\n[{category}] Q: {query}\n{'='*70}")
        context, urls = retriever.build_context(query, top_k=TOP_K)

        # 검색된 출처 먼저 표시
        print(f"검색 출처 {len(urls)}개:")
        for u in urls:
            print(f"  - {u}")

        t1 = time.time()
        answer = generate_answer(
            query, context, urls, model, tokenizer, use_tools=USE_TOOLS
        )
        dt = time.time() - t1

        print(f"\n답변 ({dt:.1f}s, {len(answer)}자):")
        print(answer)


if __name__ == "__main__":
    main()
