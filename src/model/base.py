"""베이스 모델 로드 모듈.

Qwen2.5-7B-Instruct를 4bit NF4 양자화로 로드한다.
모델 교체는 이 파일의 `load_model(name)` 한 줄 변경으로 가능하다.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
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
    model.config.use_cache = False
    return model
