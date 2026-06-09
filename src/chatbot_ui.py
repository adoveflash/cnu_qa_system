"""Task 2 — 챗봇 배치 추론 + UI 진입점.

배치 추론(chat_output.json 생성) + Gradio 챗봇 UI 실행.

사용법:
    python -m src.chatbot_ui                  # 배치 + UI
    python -m src.chatbot_ui --batch-only     # 배치만
    python -m src.chatbot_ui --ui-only        # UI만
"""

from __future__ import annotations

import json
from pathlib import Path


def _load_pipeline() -> tuple:
    """Retriever + (model, tokenizer)를 로드한다.

    Returns:
        (retriever, model_or_None, tokenizer_or_None) 튜플
    """
    from src.rag.retriever import Retriever

    print("[chat] Retriever 로드 중...")
    retriever = Retriever()

    model = None
    tokenizer = None
    try:
        from src.model.inference import load_inference_model

        print("[chat] 모델 로드 중...")
        model, tokenizer = load_inference_model()
    except Exception as e:
        print(f"[chat] 모델 로드 실패 (RAG만 사용): {e}")

    return retriever, model, tokenizer


def run_chat_batch(
    test_path: str = "data/test_chat.json",
    output_path: str = "outputs/chat_output.json",
) -> None:
    """test_chat.json에 대한 배치 추론을 수행한다.

    Args:
        test_path: 테스트 입력 JSON 경로
        output_path: 결과 출력 JSON 경로
    """
    test_file = Path(test_path)
    if not test_file.exists():
        print(f"[chat] {test_path} 없음 — 건너뜀")
        return

    with open(test_file, encoding="utf-8") as f:
        test_data = json.load(f)

    if not test_data:
        print("[chat] 테스트 데이터 비어있음 — 건너뜀")
        return

    retriever, model, tokenizer = _load_pipeline()

    from src.model.inference import fallback_answer

    if model is not None and tokenizer is not None:
        from src.model.inference import generate_answer

    print(f"[chat] {len(test_data)}건 추론 중...")
    results = []
    for idx, item in enumerate(test_data):
        question = item["user"]
        context, urls = retriever.build_context(question)

        if model is not None and tokenizer is not None:
            answer = generate_answer(question, context, urls, model, tokenizer)
        else:
            answer = fallback_answer(question, context, urls)

        # 조교 양식: id, user, model
        results.append({"id": item.get("id", idx), "user": question, "model": answer})
        print(f"  Q: {question[:40]}... → {len(answer)}자")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[chat] 결과 저장: {output_path} ({len(results)}건)")


def launch_ui() -> None:
    """Gradio 챗봇 UI를 실행한다."""
    retriever, model, tokenizer = _load_pipeline()

    from src.app.ui import launch

    launch(retriever, model, tokenizer, share=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task 2 챗봇 배치 추론 + UI")
    parser.add_argument("--batch-only", action="store_true", help="배치 추론만 실행")
    parser.add_argument("--ui-only", action="store_true", help="UI만 실행")
    parser.add_argument("--test", default="data/test_chat.json", help="테스트 파일 경로")
    parser.add_argument("--output", default="outputs/chat_output.json", help="출력 파일 경로")
    args = parser.parse_args()

    if args.ui_only:
        launch_ui()
    elif args.batch_only:
        run_chat_batch(test_path=args.test, output_path=args.output)
    else:
        run_chat_batch(test_path=args.test, output_path=args.output)
        launch_ui()
