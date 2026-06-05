"""추론 모듈.

RAG 컨텍스트와 질문을 받아 답변을 생성한다.
Tool calling을 지원하여 실시간 정보(식단, 셔틀, 학사일정, 공지)를 조회할 수 있다.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from src.model.base import load_model, load_tokenizer
from src.tools.definitions import TOOLS, execute_tool

_SYSTEM_PROMPT = (
    "당신은 충남대학교 학내 정보 안내 도우미입니다.\n"
    "규칙:\n"
    "1. 반드시 주어진 참고 자료에 있는 정보만 사용하여 답변하세요.\n"
    "2. 참고 자료에 없는 내용은 추측하지 말고 '확인되지 않은 정보입니다'라고 답하세요.\n"
    "3. 답변은 간결하고 정확하게 작성하세요.\n"
    "4. 답변 끝에 출처 URL을 포함하세요.\n"
    "5. 실시간 정보(식단, 셔틀버스, 학사일정, 공지사항)가 필요하면 도구를 호출하세요."
)
_SEED = 42
_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", flags=re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Qwen3 thinking mode 태그를 제거한다."""
    return _THINK_TAG_RE.sub("", text).strip()


def _parse_tool_calls(text: str) -> list[dict]:
    """생성된 텍스트에서 tool_call을 파싱한다.

    Args:
        text: 모델 생성 텍스트

    Returns:
        [{"name": ..., "arguments": {...}}, ...] 리스트
    """
    calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(match.group(1))
            name = obj.get("name", "")
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            calls.append({"name": name, "arguments": args})
        except (json.JSONDecodeError, KeyError):
            continue
    return calls


# URL → 사람이 읽기 쉬운 출처 라벨 매핑
_SOURCE_LABELS: dict[str, str] = {
    "computer.cnu.ac.kr": "컴퓨터융합학부",
    "plus.cnu.ac.kr": "충남대 공식",
    "job.cnu.ac.kr": "인재개발원",
    "sugang.cnu.ac.kr": "수강신청",
    "www.cnucoop.co.kr": "생활협동조합",
    "mobileadmin.cnu.ac.kr": "충남대 식단",
    "portal.cnu.ac.kr": "학사포털",
}


def _format_sources(urls: list[str]) -> str:
    """URL 리스트를 자연스러운 출처 텍스트로 변환한다."""
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


def load_model_with_lora(
    model_name: str = "Qwen/Qwen3-8B",
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


def _generate_once(
    messages: list[dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 256,
    tools: list[dict] | None = None,
) -> str:
    """단일 생성 호출. tool calling 여부와 무관하게 사용한다.

    Args:
        messages: 대화 메시지 리스트
        model: LLM 모델
        tokenizer: 토크나이저
        max_new_tokens: 최대 생성 토큰 수
        tools: tool 스키마 리스트 (None이면 tool 없이 생성)

    Returns:
        생성된 텍스트
    """
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    if tools:
        template_kwargs["tools"] = tools

    text = tokenizer.apply_chat_template(messages, **template_kwargs)
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
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_answer(
    question: str,
    context: str,
    urls: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 256,
    use_tools: bool = True,
) -> str:
    """질문과 컨텍스트를 기반으로 답변을 생성한다. Tool calling을 지원한다.

    Args:
        question: 사용자 질문
        context: RAG 검색으로 얻은 컨텍스트
        urls: 출처 URL 리스트
        model: LLM 모델
        tokenizer: 토크나이저
        max_new_tokens: 최대 생성 토큰 수
        use_tools: tool calling 활성화 여부

    Returns:
        생성된 답변 문자열
    """
    user_message = f"참고 자료:\n{context}\n\n질문: {question}" if context else question

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    tools = TOOLS if use_tools else None
    raw = _generate_once(messages, model, tokenizer, max_new_tokens, tools=tools)
    raw = _strip_think_tags(raw)

    # Tool call 파싱 및 실행 (최대 1회)
    tool_calls = _parse_tool_calls(raw) if use_tools else []
    if tool_calls:
        # 모델이 tool_call을 생성한 경우
        messages.append({"role": "assistant", "content": raw})

        for tc in tool_calls:
            print(f"  [tool] {tc['name']}({tc['arguments']})")
            result = execute_tool(tc["name"], tc["arguments"])
            messages.append({"role": "tool", "name": tc["name"], "content": result})

        # tool 결과를 포함하여 최종 답변 생성 (tool 없이)
        final_raw = _generate_once(messages, model, tokenizer, max_new_tokens, tools=None)
        answer = _strip_think_tags(final_raw)
    else:
        answer = raw

    answer += _format_sources(urls)
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
    """질문과 컨텍스트를 기반으로 답변을 스트리밍 생성한다. Tool calling을 지원한다.

    Tool call이 발생하면: tool 실행 후 최종 답변을 스트리밍한다.
    Tool call이 없으면: 직접 스트리밍한다.

    Args:
        question: 사용자 질문
        context: RAG 검색으로 얻은 컨텍스트
        urls: 출처 URL 리스트
        model: LLM 모델
        tokenizer: 토크나이저
        max_new_tokens: 최대 생성 토큰 수
        use_tools: tool calling 활성화 여부

    Yields:
        누적된 답변 문자열 (Gradio chatbot 호환)
    """
    user_message = f"참고 자료:\n{context}\n\n질문: {question}" if context else question
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    tools = TOOLS if use_tools else None

    # 1단계: 첫 번째 생성 (tool call 여부 확인)
    raw = _generate_once(messages, model, tokenizer, max_new_tokens, tools=tools)
    raw = _strip_think_tags(raw)

    tool_calls = _parse_tool_calls(raw) if use_tools else []

    if tool_calls:
        # Tool 실행 중 상태 표시
        tool_names = ", ".join(tc["name"] for tc in tool_calls)
        yield f"🔍 실시간 정보 조회 중... ({tool_names})"

        messages.append({"role": "assistant", "content": raw})
        for tc in tool_calls:
            print(f"  [tool] {tc['name']}({tc['arguments']})")
            result = execute_tool(tc["name"], tc["arguments"])
            messages.append({"role": "tool", "name": tc["name"], "content": result})

        # 2단계: tool 결과로 최종 답변 스트리밍
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        text = tokenizer.apply_chat_template(messages, **template_kwargs)
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
            yield _strip_think_tags(accumulated)

        thread.join()

        final = _strip_think_tags(accumulated)
        source_text = _format_sources(urls)
        if source_text:
            yield final + source_text
    else:
        # Tool call 없음 — 이미 생성된 결과를 그대로 반환
        answer = raw
        source_text = _format_sources(urls)
        if source_text:
            answer += source_text
        yield answer


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
    answer += _format_sources(urls)

    return answer
