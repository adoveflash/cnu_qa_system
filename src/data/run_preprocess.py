"""데이터 전처리 파이프라인 실행.

사용법:
  python -m src.data.run_preprocess              # 클리닝 + 마스킹 + 청킹만
  python -m src.data.run_preprocess --generate-qa # 위 + Q&A 생성 + eval 분리
  python -m src.data.run_preprocess --generate-qa --max-chunks 10  # Q&A 테스트
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.cleaner import clean_all
from src.data.chunker import chunk_all
from src.data.pii_masker import mask_file

RAW_DIR = Path("data/corpus/raw")
CLEANED_PATH = Path("data/corpus/cleaned.jsonl")
CHUNKS_PATH = Path("data/corpus/chunks.jsonl")
TRAIN_PATH = Path("data/qa/train.jsonl")
EVAL_PATH = Path("data/qa/eval.jsonl")


def main() -> None:
    """전처리 파이프라인: clean → mask PII → chunk (→ Q&A 생성 → eval 분리)."""
    parser = argparse.ArgumentParser(description="데이터 전처리 파이프라인")
    parser.add_argument("--generate-qa", action="store_true", help="Q&A 생성까지 수행")
    parser.add_argument("--max-chunks", type=int, default=None, help="Q&A 생성 최대 청크 수")
    parser.add_argument("--model", type=str, default="qwen2.5:7b", help="Ollama 모델명")
    args = parser.parse_args()

    # 1단계: 클리닝
    print("[1/3] 클리닝 시작")
    cleaned_count = clean_all(RAW_DIR, CLEANED_PATH)
    print(f"[1/3] 클리닝 완료: {cleaned_count}건\n")

    # 2단계: PII 마스킹 (in-place)
    print("[2/3] PII 마스킹 시작")
    doc_count, mask_count = mask_file(CLEANED_PATH, CLEANED_PATH)
    print(f"[2/3] PII 마스킹 완료: {doc_count}건 처리, {mask_count}건 치환\n")

    # 3단계: 청킹
    print("[3/3] 청킹 시작")
    chunk_count = chunk_all(CLEANED_PATH, CHUNKS_PATH)
    print(f"[3/3] 청킹 완료: {chunk_count}개 청크 생성\n")

    print("전처리 파이프라인 완료!")
    print(f"  cleaned: {CLEANED_PATH}")
    print(f"  chunks:  {CHUNKS_PATH}")

    # 4단계 (옵션): Q&A 생성
    if args.generate_qa:
        from src.data.qa_generator import generate_all, split_eval_set

        print(f"\n[4/5] Q&A 생성 시작 (모델: {args.model}, 증분 처리)")
        new_count = generate_all(
            CHUNKS_PATH, TRAIN_PATH, model=args.model, max_chunks=args.max_chunks
        )
        print(f"[4/5] Q&A 생성 완료: {new_count}쌍 새로 생성\n")

        # 5단계: eval 분리
        print("[5/5] eval 세트 분리")
        split_eval_set(TRAIN_PATH, EVAL_PATH)
        print()


if __name__ == "__main__":
    main()
