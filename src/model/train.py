"""QLoRA 파인튜닝 모듈.

train.jsonl(Q&A 쌍)과 chunks.jsonl(RAG 컨텍스트)을 사용하여 LoRA 어댑터를 학습한다.
추론 시와 동일한 형식(시스템 프롬프트 + 참고 자료 + 질문 → 답변)으로 학습하며,
답변 토큰에만 loss를 적용한다.
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

from src.model.base import load_model, load_tokenizer, _DEFAULT_MODEL

_SEED = 42
_SYSTEM_PROMPT = (
    "당신은 충남대학교 학내 정보 안내 도우미입니다.\n"
    "규칙:\n"
    "1. 반드시 주어진 참고 자료에 있는 정보만 사용하여 답변하세요.\n"
    "2. 참고 자료에 없는 내용은 추측하지 말고 '확인되지 않은 정보입니다'라고 답하세요.\n"
    "3. 답변은 간결하고 정확하게 작성하세요.\n"
    "4. 답변 끝에 출처 URL을 포함하세요."
)


def get_lora_config() -> LoraConfig:
    """QLoRA 설정을 반환한다. r=32, alpha=64, dropout=0.1.

    Returns:
        LoraConfig 객체
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
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


def load_chunks_map(chunks_path: Path) -> dict[str, str]:
    """chunks.jsonl에서 chunk_id → text 매핑을 로드한다.

    Args:
        chunks_path: data/corpus/chunks.jsonl 경로

    Returns:
        {chunk_id: text} 딕셔너리
    """
    chunks_map: dict[str, str] = {}
    if not chunks_path.exists():
        return chunks_map
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunk = json.loads(line)
                chunks_map[chunk["chunk_id"]] = chunk["text"]
    return chunks_map


def format_chat_messages(qa: dict[str, Any], chunk_text: str = "") -> list[dict[str, str]]:
    """Q&A 레코드를 chat 메시지 형식으로 변환한다.

    추론 시와 동일한 형식: 시스템 프롬프트 + "참고 자료:\n{context}\n\n질문:" + 답변

    Args:
        qa: question, answer 키를 포함하는 딕셔너리
        chunk_text: RAG 컨텍스트 텍스트 (빈 문자열이면 질문만)

    Returns:
        [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]
    """
    if chunk_text:
        user_content = f"참고 자료:\n{chunk_text}\n\n질문: {qa['question']}"
    else:
        user_content = qa["question"]

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": qa["answer"]},
    ]


def _find_assistant_start(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    """assistant 응답이 시작되는 토큰 위치를 찾는다.

    system + user 메시지만 토크나이즈한 길이로 assistant 시작점을 결정한다.

    Args:
        tokenizer: 토크나이저
        messages: 전체 chat 메시지 (system + user + assistant)

    Returns:
        assistant 응답 시작 토큰 인덱스
    """
    prompt_only = messages[:2]  # system + user
    tpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if "qwen" in _DEFAULT_MODEL.lower():
        tpl_kwargs["enable_thinking"] = False
    prompt_text = tokenizer.apply_chat_template(prompt_only, **tpl_kwargs)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    return len(prompt_ids)


def prepare_dataset(
    train_data: list[dict[str, Any]],
    tokenizer: Any,
    chunks_map: dict[str, str],
    max_length: int = 768,
) -> Dataset:
    """학습 데이터를 토크나이즈하고 답변 토큰에만 loss를 적용하는 Dataset을 생성한다.

    Args:
        train_data: Q&A 레코드 리스트
        tokenizer: 토크나이저
        chunks_map: {chunk_id: text} 매핑
        max_length: 최대 시퀀스 길이

    Returns:
        HuggingFace Dataset (input_ids, attention_mask, labels)
    """
    random.seed(_SEED)
    random.shuffle(train_data)

    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for qa in train_data:
        chunk_text = chunks_map.get(qa.get("chunk_id", ""), "")
        messages = format_chat_messages(qa, chunk_text)

        # 전체 시퀀스 토크나이즈
        tpl_kwargs2: dict = {"tokenize": False, "add_generation_prompt": False}
        if "qwen" in _DEFAULT_MODEL.lower():
            tpl_kwargs2["enable_thinking"] = False
        full_text = tokenizer.apply_chat_template(messages, **tpl_kwargs2)
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            add_special_tokens=False,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # assistant 시작점 찾기 → 그 이전은 labels=-100
        assistant_start = _find_assistant_start(tokenizer, messages)
        labels = [-100] * min(assistant_start, len(input_ids))
        labels += input_ids[len(labels) :]

        # padding 부분도 -100
        labels = [lb if am == 1 else -100 for lb, am in zip(labels, attention_mask)]

        # max_length 맞추기
        if len(labels) < max_length:
            labels += [-100] * (max_length - len(labels))
        labels = labels[:max_length]

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)

    return Dataset.from_dict(
        {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }
    )


def train(
    train_path: Path = Path("data/qa/train.jsonl"),
    eval_path: Path = Path("data/qa/eval.jsonl"),
    chunks_path: Path = Path("data/corpus/chunks.jsonl"),
    output_dir: Path = Path("models/lora_adapter"),
    model_name: str = _DEFAULT_MODEL,
    num_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 768,
) -> None:
    """QLoRA 파인튜닝을 실행한다.

    Args:
        train_path: 학습 데이터 경로
        eval_path: 검증 데이터 경로
        chunks_path: 청크 데이터 경로
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

    # 학습 시에만 KV cache 비활성화
    model.config.use_cache = False

    print("LoRA 적용")
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 청크 매핑 로드
    print(f"청크 매핑 로드: {chunks_path}")
    chunks_map = load_chunks_map(chunks_path)
    print(f"  청크 수: {len(chunks_map)}")

    # 학습 데이터
    print(f"학습 데이터 로드: {train_path}")
    train_data = load_train_data(train_path)
    train_dataset = prepare_dataset(train_data, tokenizer, chunks_map, max_length)
    print(f"학습 샘플 수: {len(train_dataset)}")

    # 검증 데이터
    eval_dataset = None
    if eval_path.exists():
        print(f"검증 데이터 로드: {eval_path}")
        eval_data = load_train_data(eval_path)
        eval_dataset = prepare_dataset(eval_data, tokenizer, chunks_map, max_length)
        print(f"검증 샘플 수: {len(eval_dataset)}")

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=50 if eval_dataset else None,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        seed=_SEED,
        bf16=True,
        report_to="none",
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("학습 시작")
    trainer.train()

    print(f"어댑터 저장: {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("학습 완료!")
