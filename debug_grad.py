"""졸업요건 답변 회귀 진단.

졸업 질문을 top_k=3, 5로 각각 검색·생성하여
- 검색된 청크 전문 (총 졸업학점 텍스트가 잡히는지)
- 각 설정의 최종 답변
을 비교 출력한다.

사용법: CUDA_VISIBLE_DEVICES=0 python debug_grad.py
"""

from __future__ import annotations

QUERY = "컴퓨터융합학부 졸업 요건이 어떻게 되나요?"


def main() -> None:
    from src.rag.retriever import Retriever

    print("bge-m3 로드 중...")
    retriever = Retriever()

    from src.model.inference import generate_answer, load_inference_model

    print("Gemma 4 로드 중...")
    model, tokenizer = load_inference_model()
    print("로드 완료\n")

    # 1) 검색 청크 전문 확인 (top_k=8까지 넓게)
    print("=" * 70)
    print(f"질문: {QUERY}")
    print("=" * 70)
    results = retriever.retrieve(QUERY, top_k=8)
    print(f"\n[검색된 청크 {len(results)}개 — 총 졸업학점 텍스트가 있는지 확인]\n")
    for i, r in enumerate(results, 1):
        print(f"--- {i}. dist={r['distance']:.3f} [{r['source']}] {r.get('title','')[:40]}")
        print(r["text"][:500])
        print()

    # 2) top_k=3 vs 5 답변 비교
    for k in (3, 5):
        print("\n" + "=" * 70)
        print(f"[top_k={k}] 답변")
        print("=" * 70)
        context, urls = retriever.build_context(QUERY, top_k=k)
        answer = generate_answer(QUERY, context, urls, model, tokenizer, use_tools=False)
        print(answer)


if __name__ == "__main__":
    main()
