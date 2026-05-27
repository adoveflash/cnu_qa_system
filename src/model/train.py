"""QLoRA 파인튜닝 모듈.

train.jsonl을 사용하여 LoRA 어댑터를 학습한다.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer

from src.model.base import load_model, load_tokenizer

_SEED = 42
_SYSTEM_PROMPT = "당신은 충남대학교 학내 정보 안내 도우미입니다. 주어진 참고 자료를 바탕으로 정확하게 답변하세요. 참고 자료에 없는 내용은 '확인되지 않은 정보입니다'라고 답하세요."


def get_lora_config() -> LoraConfig:
    """QLoRA 설정을 반환한다. r=16, alpha=32, dropout=0.05.

    Returns:
        LoraConfig 객체
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )


def load_train_data(train_path: Path) -> list[dict[str, Any]]:
    """train.jsonl을 로드한다.

    Args:
        train_path: data/qa/train.jsonl 경로

    Returns:
        Q&A 레코드 리스트
    """
    with open(train_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_chat_messages(qa: dict[str, Any]) -> list[dict[str, str]]:
    """Q&A 레코드를 chat 메시지 형식으로 변환한다.

    Args:
        qa: question, answer, url 키를 포함하는 딕셔너리

    Returns:
        [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]
    """
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": qa["question"]},
        {"role": "assistant", "content": qa["answer"]},
    ]


def prepare_dataset(
    train_data: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int = 512,
) -> Dataset:
    """학습 데이터를 토크나이즈된 Dataset으로 변환한다.

    Args:
        train_data: Q&A 레코드 리스트
        tokenizer: 토크나이저
        max_length: 최대 시퀀스 길이

    Returns:
        HuggingFace Dataset
    """
    random.seed(_SEED)
    random.shuffle(train_data)

    texts = []
    for qa in train_data:
        messages = format_chat_messages(qa)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    def tokenize_fn(examples: dict) -> dict:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = Dataset.from_dict({"text": texts})
    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


def train(
    train_path: Path = Path("data/qa/train.jsonl"),
    output_dir: Path = Path("models/lora_adapter"),
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 512,
) -> None:
    """QLoRA 파인튜닝을 실행한다.

    Args:
        train_path: 학습 데이터 경로
        output_dir: LoRA 어댑터 저장 경로
        model_name: 베이스 모델명
        num_epochs: 학습 에폭 수
        batch_size: 배치 크기
        learning_rate: 학습률
        max_length: 최대 시퀀스 길이
    """
    torch.manual_seed(_SEED)

    print(f"모델 로드: {model_name}")
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)

    print("LoRA 적용")
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"학습 데이터 로드: {train_path}")
    train_data = load_train_data(train_path)
    dataset = prepare_dataset(train_data, tokenizer, max_length)
    print(f"학습 샘플 수: {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        seed=_SEED,
        bf16=True,
        report_to="none",
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("학습 시작")
    trainer.train()

    print(f"어댑터 저장: {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("학습 완료!")
