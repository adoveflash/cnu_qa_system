"""Task 3 — 실시간 정보 반영 모듈.

RAG live 컬렉션을 갱신한 뒤, 실시간 질문에 대한 배치 추론을 수행한다.
결과: outputs/realtime_output.json

사용법:
    python -m src.realtime_model                    # live 갱신 + 추론
    python -m src.realtime_model --skip-refresh     # 갱신 건너뛰고 추론만
"""

from __future__ import annotations

import json
from pathlib import Path


def run_realtime_batch(
    test_path: str = "data/test_realtime.json",
    output_path: str = "outputs/realtime_output.json",
    adapter_path: str = "models/lora_adapter",
    skip_refresh: bool = False,
) -> None:
    """실시간 정보 질문에 대한 배치 추론을 수행한다.

    Args:
        test_path: 테스트 입력 JSON 경로
        output_path: 결과 출력 JSON 경로
        adapter_path: LoRA 어댑터 경로
        skip_refresh: True면 live 컬렉션 갱신을 건너뜀
    """
    test_file = Path(test_path)
    if not test_file.exists():
        print(f"[realtime] {test_path} 없음 — 건너뜀")
        return

    with open(test_file, encoding="utf-8") as f:
        test_data = json.load(f)

    if not test_data:
        print("[realtime] 테스트 데이터 비어있음 — 건너뜀")
        return

    # 1) live 컬렉션 갱신
    if not skip_refresh:
        print("[realtime] live 컬렉션 갱신 중...")
        from src.rag.live_updater import refresh_live_collection

        refresh_live_collection()

    # 2) Retriever 로드
    print("[realtime] Retriever 로드 중...")
    from src.rag.retriever import Retriever

    retriever = Retriever()

    # 3) 모델 로드 시도
    model = None
    tokenizer = None
    adapter = Path(adapter_path)
    if adapter.exists():
        try:
            from src.model.inference import load_model_with_lora

            print("[realtime] LoRA 모델 로드 중...")
            model, tokenizer = load_model_with_lora(adapter_path=str(adapter))
        except Exception as e:
            print(f"[realtime] 모델 로드 실패 (RAG만 사용): {e}")

    from src.model.inference import fallback_answer

    if model is not None and tokenizer is not None:
        from src.model.inference import generate_answer

    # 4) 배치 추론
    print(f"[realtime] {len(test_data)}건 추론 중...")
    results = []
    for item in test_data:
        question = item["user"]
        context, urls = retriever.build_context(question)

        if model is not None and tokenizer is not None:
            answer = generate_answer(question, context, urls, model, tokenizer)
        else:
            answer = fallback_answer(question, context, urls)

        results.append({"user": question, "model": answer})
        print(f"  Q: {question[:40]}... → {len(answer)}자")

    # 5) 저장
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[realtime] 결과 저장: {output_path} ({len(results)}건)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task 3 실시간 정보 배치 추론")
    parser.add_argument("--skip-refresh", action="store_true", help="live 컬렉션 갱신 건너뛰기")
    parser.add_argument("--test", default="data/test_realtime.json", help="테스트 파일 경로")
    parser.add_argument("--output", default="outputs/realtime_output.json", help="출력 파일 경로")
    parser.add_argument("--adapter", default="models/lora_adapter", help="LoRA 어댑터 경로")
    args = parser.parse_args()

    run_realtime_batch(
        test_path=args.test,
        output_path=args.output,
        adapter_path=args.adapter,
        skip_refresh=args.skip_refresh,
    )
