"""베이스 모델 로드 모듈 (Gemma 4 전용).

google/gemma-4-12b-it를 4bit NF4 양자화로 로드한다.
환경변수 BASE_MODEL로 변경 가능하나 로딩 클래스/세팅은 Gemma 4 기준이다.
"""

from __future__ import annotations

import os

# CUDA 메모리 단편화로 인한 OOM 방지 (torch CUDA 초기화 전에 설정해야 적용됨)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

# Gemma 4는 텍스트 전용 Gemma4ForCausalLM으로 로드한다.
# AutoModelForImageTextToText(멀티모달 conditional-generation)로 로드하면 config 파싱에서
# `'list' object has no attribute 'keys'`로 터진다(검증된 submission 노트북·메모리 기준).
from transformers import Gemma4ForCausalLM as _ModelClass

_DEFAULT_MODEL = os.environ.get("BASE_MODEL", "google/gemma-4-12b-it")
_SEED = 42

# Gemma 4 연산 dtype. Turing(T4·RTX8000)은 bf16 텐서코어가 없어 float16을 쓴다
# (가속은 안 되지만 정확도 보장). Ampere+에서도 float16으로 정상 동작.
_COMPUTE_DTYPE = torch.float16


def get_bnb_config() -> BitsAndBytesConfig:
    """4bit NF4 양자화 설정을 반환한다."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_COMPUTE_DTYPE,
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


def load_model(name: str = _DEFAULT_MODEL):
    """Gemma 4를 4bit 양자화로 로드한다.

    Args:
        name: HuggingFace 모델 이름

    Returns:
        양자화된 모델
    """
    torch.manual_seed(_SEED)
    model = _ModelClass.from_pretrained(
        name,
        quantization_config=get_bnb_config(),
        device_map={"": 0},  # 통째 GPU0 적재 ("auto"는 4bit에서 CPU offload 오류)
        dtype=_COMPUTE_DTYPE,
    )
    return model
