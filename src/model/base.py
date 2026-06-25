"""베이스 모델 로드 모듈.

4bit NF4 양자화로 모델을 로드한다.
환경변수 BASE_MODEL로 모델 선택 가능 (기본: google/gemma-4-12b-it).
예: BASE_MODEL=Qwen/Qwen3-8B python ...
"""

from __future__ import annotations

import os

# CUDA 메모리 단편화로 인한 OOM 방지 (torch CUDA 초기화 전에 설정해야 적용됨)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

# AutoModelForImageTextToText는 Gemma 4/3 체크포인트를 올바른 클래스로 자동 매핑한다.
# (Gemma 4 → Gemma4ForConditionalGeneration, Gemma 3 → Gemma3ForConditionalGeneration)
# 덕분에 BASE_MODEL 환경변수만 바꾸면 모델 교체 가능 (폴백 Tier B).
from transformers import AutoModelForImageTextToText as _ModelClass

# 기본은 Gemma 4. Gemma 3은 폴백(Tier B) — BASE_MODEL=google/gemma-3-12b-it 로 전환.
_DEFAULT_MODEL = os.environ.get("BASE_MODEL", "google/gemma-4-12b-it")
_SEED = 42


def _dtype_for(name: str) -> torch.dtype:
    """모델별 연산 dtype을 고른다.

    Gemma 3는 float16이면 NaN 오버플로 → 생성이 전부 <pad>가 되므로 반드시 bfloat16.
    Gemma 4는 float16에서 정상. (Turing은 bf16 텐서코어가 없어 가속은 안 되지만 정확도는 보장.)

    Args:
        name: HuggingFace 모델 이름

    Returns:
        torch.bfloat16 (Gemma 3) 또는 torch.float16 (그 외)
    """
    return torch.bfloat16 if "gemma-3" in name.lower() else torch.float16


def get_bnb_config(name: str = _DEFAULT_MODEL) -> BitsAndBytesConfig:
    """4bit NF4 양자화 설정을 반환한다.

    Args:
        name: HuggingFace 모델 이름 (compute dtype 선택용)

    Returns:
        BitsAndBytesConfig 객체
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_dtype_for(name),
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
    """4bit 양자화 모델을 로드한다.

    Args:
        name: HuggingFace 모델 이름

    Returns:
        양자화된 모델
    """
    torch.manual_seed(_SEED)
    model = _ModelClass.from_pretrained(
        name,
        quantization_config=get_bnb_config(name),
        device_map={"": 0},  # 통째 GPU0 적재 ("auto"는 Gemma 3 4bit에서 CPU offload 오류)
        torch_dtype=_dtype_for(name),
    )
    return model
