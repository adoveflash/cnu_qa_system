"""로컬 서버 학습 실행 스크립트.

사용법:
    # GPU 0번(RTX 8000) 사용 (기본)
    python scripts/run_train.py

    # GPU 지정
    CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py

    # 배치 사이즈 변경
    python scripts/run_train.py --batch_size 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA 학습 실행")
    parser.add_argument("--train_path", type=str, default="data/qa/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="models/lora_adapter")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    train(
        train_path=Path(args.train_path),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
