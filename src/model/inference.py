"""추론 모듈.

RAG 컨텍스트와 질문을 받아 답변을 생성한다.
키워드 기반 tool 감지로 실시간 정보(식단, 셔틀, 학사일정, 공지)를 조회할 수 있다.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from src.model.base import load_model, load_tokenizer
from src.tools.definitions import execute_tool
from src.tools.detector import detect_tool

_SEED = 42
_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def _build_system_prompt() -> str:
    """오늘 날짜를 포함한 시스템 프롬프트를 생성한다."""
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d (%A)")
    return (
        f"너는 충남대학교 학내 정보를 안내하는 친절한 AI 챗봇이야. 오늘 날짜: {today}\n\n"
        "대화 스타일:\n"
        "- 친근하고 자연스러운 말투로 대답해. 딱딱하지 않게, 친구에게 설명하듯이.\n"
        "- '~해요', '~이에요' 같은 존댓말을 사용하되 부드럽게.\n"
        "- 질문에 맞는 핵심 정보를 먼저 알려주고, 필요하면 추가 설명을 덧붙여.\n\n"
        "규칙:\n"
        "1. 주어진 참고 자료에 있는 정보를 기반으로 답변해.\n"
        "2. 참고 자료에 없는 내용은 절대 지어내지 마. '해당 정보를 찾지 못했어요'라고 솔직히 답해.\n"
        "3. 기숙사 식단과 학생회관 식단을 혼동하지 마.\n"
        "4. 사용자가 점심만 물어보면 점심만 답해.\n"
        "5. 참고 자료의 원본 데이터를 정확히 전달해. 없는 메뉴나 일정을 지어내지 마.\n"
        "6. 반드시 한국어로 답변해."
    )


_SYSTEM_PROMPT = _build_system_prompt()

# URL → 출처 라벨 매핑
_SOURCE_LABELS: dict[str, str] = {
    "computer.cnu.ac.kr": "컴퓨터융합학부",
    "plus.cnu.ac.kr": "충남대 공식",
    "job.cnu.ac.kr": "인재개발원",
    "sugang.cnu.ac.kr": "수강신청",
    "www.cnucoop.co.kr": "생활협동조합",
    "mobileadmin.cnu.ac.kr": "충남대 식단",
    "portal.cnu.ac.kr": "학사포털",
}


def _strip_think_tags(text: str) -> str:
    """Qwen3 thinking mode 태그를 제거한다."""
    return _THINK_TAG_RE.sub("", text).strip()


def _format_sources(urls: list[str]) -> str:
    """URL 리스트를 출처 텍스트로 변환한다."""
    if not urls:
        return ""
    seen: list[str] = []
    for url in urls:
        for domain, label in _SOURCE_LABELS.items():
            if domain in url:
                if label not in seen:
                    seen.append(label)
                break
        else:
            from urllib.parse import urlparse

            host = urlparse(url).hostname or url
            short = host.replace("www.", "").split(".")[0]
            if short not in seen:
                seen.append(short)
    return "\n\n📌 " + ", ".join(seen) + " 정보를 참고했습니다."


def _build_messages(question: str, context: str) -> list[dict]:
    """시스템 프롬프트 + 참고자료 + 질문으로 메시지를 구성한다."""
    user_content = f"참고 자료:\n{context}\n\n질문: {question}" if context else question
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _resolve_context(
    question: str, rag_context: str, urls: list[str], use_tools: bool
) -> tuple[str, list[str], bool]:
    """질문에 대해 tool 또는 RAG 컨텍스트를 결정한다.

    Returns:
        (final_context, final_urls, used_tool)
    """
    if use_tools:
        tool_name, tool_args = detect_tool(question)
        if tool_name:
            print(f"  [tool] {tool_name}({tool_args})")
            result = execute_tool(tool_name, tool_args)
            tool_context = f"[{tool_name} 결과]\n{result}"
            return tool_context, [], True
    return rag_context, urls, False


def load_model_with_lora(
    model_name: str = "Qwen/Qwen3-8B",
    adapter_path: str = "models/lora_adapter",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """베이스 모델에 LoRA 어댑터를 합쳐서 로드한다."""
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
    use_tools: bool = True,
) -> str:
    """답변을 생성한다 (배치 추론용).

    Args:
        question: 사용자 질문
        context: RAG 검색 컨텍스트
        urls: 출처 URL 리스트
        model: LLM 모델
        tokenizer: 토크나이저
        max_new_tokens: 최대 생성 토큰 수
        use_tools: tool 사용 여부

    Returns:
        생성된 답변 문자열
    """
    final_context, final_urls, used_tool = _resolve_context(
        question, context, urls, use_tools
    )
    messages = _build_messages(question, final_context)

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    torch.manual_seed(_SEED)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    answer = _strip_think_tags(answer)

    if not used_tool:
        answer += _format_sources(final_urls)
    return answer


def generate_answer_stream(
    question: str,
    context: str,
    urls: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 192,
    use_tools: bool = True,
) -> Iterator[str]:
    """답변을 스트리밍 생성한다 (UI용).

    Args:
        question: 사용자 질문
        context: RAG 검색 컨텍스트
        urls: 출처 URL 리스트
        model: LLM 모델
        tokenizer: 토크나이저
        max_new_tokens: 최대 생성 토큰 수
        use_tools: tool 사용 여부

    Yields:
        누적된 답변 문자열
    """
    # tool 감지 + 실행 (스트리밍 전에 완료)
    used_tool = False
    if use_tools:
        tool_name, tool_args = detect_tool(question)
        if tool_name:
            yield f"🔍 실시간 정보 조회 중... ({tool_name})"
            print(f"  [tool] {tool_name}({tool_args})")
            result = execute_tool(tool_name, tool_args)
            final_context = f"[{tool_name} 결과]\n{result}"
            final_urls: list[str] = []
            used_tool = True
        else:
            final_context = context
            final_urls = urls
    else:
        final_context = context
        final_urls = urls

    messages = _build_messages(question, final_context)

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": 1.2,
        "streamer": streamer,
    }

    torch.manual_seed(_SEED)
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    accumulated = ""
    for chunk in streamer:
        accumulated += chunk
        yield accumulated

    thread.join()

    # 최종 정리: think 태그 제거 + 출처 추가
    final = _strip_think_tags(accumulated)
    if not used_tool:
        final += _format_sources(final_urls)
    yield final


def fallback_answer(question: str, context: str, urls: list[str]) -> str:
    """모델 없이 RAG 컨텍스트만으로 응답을 구성한다."""
    if not context:
        return "관련 정보를 찾을 수 없습니다."

    excerpt = context[:500]
    if len(context) > 500:
        excerpt += "..."

    answer = f"검색된 정보를 바탕으로 답변드립니다.\n\n{excerpt}"
    answer += _format_sources(urls)
    return answer
