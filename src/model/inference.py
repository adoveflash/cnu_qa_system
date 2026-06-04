"""추론 모듈.

RAG 컨텍스트와 질문을 받아 답변을 생성한다.
"""

from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.base import load_model, load_tokenizer

_SYSTEM_PROMPT = "당신은 충남대학교 학내 정보 안내 도우미입니다. 주어진 참고 자료를 바탕으로 정확하게 답변하세요. 참고 자료에 없는 내용은 '확인되지 않은 정보입니다'라고 답하세요. 답변 끝에 출처 URL을 포함하세요."
_SEED = 42


def load_model_with_lora(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    adapter_path: str = "models/lora_adapter",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """베이스 모델에 LoRA 어댑터를 합쳐서 로드한다.

    Args:
        model_name: 베이스 모델명
        adapter_path: LoRA 어댑터 경로 (로컬 또는 HF Hub)

    Returns:
        (모델, 토크나이저) 튜플
    """
    tokenizer = load_tokenizer(model_name)
    base_model = load_model(model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_answer(
    question: str,
    context: str,
    urls: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 256,
) -> str:
    """질문과 컨텍스트를 기반으로 답변을 생성한다.

    Args:
        question: 사용자 질문
        context: RAG 검색으로 얻은 컨텍스트
        urls: 출처 URL 리스트
        model: LLM 모델
        tokenizer: 토크나이저
        max_new_tokens: 최대 생성 토큰 수

    Returns:
        생성된 답변 문자열
    """
    user_message = f"참고 자료:\n{context}\n\n질문: {question}"

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    torch.manual_seed(_SEED)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    # 입력 부분 제거하고 생성된 부분만 디코딩
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 출처 URL 추가 (답변에 없으면)
    if urls and not any(url in answer for url in urls):
        url_text = "\n\n출처:\n" + "\n".join(f"- {url}" for url in urls)
        answer += url_text

    return answer


def fallback_answer(question: str, context: str, urls: list[str]) -> str:
    """모델 없이 RAG 컨텍스트만으로 응답을 구성한다.

    Args:
        question: 사용자 질문
        context: RAG 검색 컨텍스트
        urls: 출처 URL 리스트

    Returns:
        RAG 기반 응답 문자열
    """
    if not context:
        return "관련 정보를 찾을 수 없습니다."

    excerpt = context[:500]
    if len(context) > 500:
        excerpt += "..."

    answer = f"검색된 정보를 바탕으로 답변드립니다.\n\n{excerpt}"

    if urls:
        answer += "\n\n출처:\n" + "\n".join(f"- {url}" for url in urls)

    return answer
