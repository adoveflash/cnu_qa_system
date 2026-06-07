"""베이스 모델 로드 모듈.

4bit NF4 양자화로 모델을 로드한다.
환경변수 BASE_MODEL로 모델 선택 가능 (기본: Qwen/Qwen3-8B).
예: BASE_MODEL=google/gemma-4-12b-it python ...
"""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_DEFAULT_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")
_SEED = 42


def get_bnb_config() -> BitsAndBytesConfig:
    """4bit NF4 양자화 설정을 반환한다.

    Returns:
        BitsAndBytesConfig 객체
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(name: str = _DEFAULT_MODEL) -> AutoTokenizer:
    """토크나이저를 로드한다.

    Args:
        name: HuggingFace 모델 이름

    Returns:
        토크나이저 객체
    """
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(name: str = _DEFAULT_MODEL) -> AutoModelForCausalLM:
    """4bit 양자화 모델을 로드한다.

    Args:
        name: HuggingFace 모델 이름

    Returns:
        양자화된 CausalLM 모델
    """
    torch.manual_seed(_SEED)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    return model
