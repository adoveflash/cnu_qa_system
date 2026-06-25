"""베이스 모델 로드 모듈 (Gemma 4 전용).

google/gemma-4-12b-it를 4bit NF4 양자화로 로드한다.
"""

from __future__ import annotations

import os

# CUDA 메모리 단편화로 인한 OOM 방지 (torch CUDA 초기화 전에 설정해야 적용됨)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# transformers 5.10.x는 import 시점에 torch.float8_e8m0fnu(torch 2.7+ 전용 dtype)를 참조한다.
# 드라이버가 CUDA 12.4면 torch 2.6(cu124)이 상한이라 그 dtype이 없어 import가 깨진다.
# fp8 경로는 안 타고 4bit NF4만 쓰므로, 존재하는 float8 dtype으로 alias만 깔아 import를
# 통과시킨다(실제 fp8 연산 없음). torch 2.7+면 이 분기는 건너뛴다.
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.float8_e4m3fn

from transformers import AutoTokenizer, BitsAndBytesConfig

# gemma-4-12b-it는 멀티모달 unified 체크포인트(text decoder + vision/audio embedder)라
# AutoModelForImageTextToText로 로드해야 가중치(model.language_model.*)가 올바로 매핑된다.
# Gemma4ForCausalLM(텍스트 전용)으로 로드하면 키가 안 맞아 전부 랜덤 초기화된다.
from transformers import AutoModelForImageTextToText as _ModelClass

# transformers 회귀버그(#42374): gemma-4 unified 체크포인트를 텍스트로 로드할 때
# GenerationConfig.from_model_config가 dict 형태 config에 to_dict()를 호출해
# `'dict' object has no attribute 'to_dict'`로 깨진다. upstream fix와 동일하게,
# dict면 PretrainedConfig로 감싼 뒤 진행하도록 from_model_config를 패치한다.
from transformers import GenerationConfig as _GenerationConfig, PretrainedConfig as _PretrainedConfig

_orig_from_model_config = _GenerationConfig.from_model_config.__func__


def _safe_from_model_config(cls, model_config):
    try:
        return _orig_from_model_config(cls, model_config)
    except AttributeError:
        if isinstance(model_config, dict):
            return _orig_from_model_config(cls, _PretrainedConfig.from_dict(model_config))
        return cls()


_GenerationConfig.from_model_config = classmethod(_safe_from_model_config)

_DEFAULT_MODEL = os.environ.get("BASE_MODEL", "google/gemma-4-12b-it")
_SEED = 42

# Turing(T4·RTX8000)은 bf16 텐서코어가 없어 float16.
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
    """토크나이저를 로드한다."""
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(name: str = _DEFAULT_MODEL):
    """Gemma 4를 4bit 양자화로 로드한다."""
    torch.manual_seed(_SEED)
    model = _ModelClass.from_pretrained(
        name,
        quantization_config=get_bnb_config(),
        device_map={"": 0},  # 통째 GPU0 적재
        dtype=_COMPUTE_DTYPE,
    )
    return model
