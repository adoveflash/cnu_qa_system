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
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, LogitsProcessor

from src.model.base import load_model, load_tokenizer, _DEFAULT_MODEL
from src.tools.definitions import execute_tool
from src.tools.detector import detect_tool

_SEED = 42
_IS_QWEN = "qwen" in _DEFAULT_MODEL.lower()
_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")


def _remove_chinese(text: str) -> str:
    """중국어 문자를 제거한다."""
    return _CHINESE_RE.sub("", text).strip()


class SuppressChineseLogitsProcessor(LogitsProcessor):
    """중국어 토큰의 생성 확률을 -inf로 설정하여 중국어 출력을 차단한다."""

    def __init__(self, tokenizer: AutoTokenizer):
        self._bad_ids: list[int] = []
        chinese_re = re.compile(r"[\u4e00-\u9fff]")
        for token_id in range(tokenizer.vocab_size):
            token = tokenizer.decode([token_id])
            if chinese_re.search(token):
                self._bad_ids.append(token_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self._bad_ids] = float("-inf")
        return scores


def _build_chinese_suppressor(tokenizer: AutoTokenizer) -> list:
    """중국어 억제 LogitsProcessor 리스트를 생성한다."""
    return [SuppressChineseLogitsProcessor(tokenizer)]


def _build_system_prompt() -> str:
    """오늘 날짜와 현재 학기를 포함한 시스템 프롬프트를 생성한다."""
    from datetime import datetime

    now = datetime.now()
    today = now.strftime("%Y-%m-%d (%A)")
    month = now.month
    if 3 <= month <= 8:
        semester = f"{now.year}학년도 1학기"
    else:
        year = now.year if month >= 9 else now.year - 1
        semester = f"{year}학년도 2학기"
    return (
        f"너는 충남대학교 학내 정보를 안내하는 친절한 AI 챗봇이야.\n"
        f"오늘 날짜: {today} | 현재 학기: {semester}\n\n"
        "대화 스타일:\n"
        "- 친근하고 자연스러운 말투로 대답해. 딱딱하지 않게, 친구에게 설명하듯이.\n"
        "- '~해요', '~이에요' 같은 존댓말을 사용하되 부드럽게.\n"
        "- 질문에 맞는 핵심 정보를 먼저 알려주고, 필요하면 추가 설명을 덧붙여.\n\n"
        "규칙:\n"
        "1. 주어진 참고 자료에 있는 정보를 기반으로 답변해. 참고 자료에 날짜, 학점, 일정 등 구체적 수치가 있으면 반드시 포함해서 답해.\n"
        "2. 참고 자료에 없는 내용은 절대 지어내지 마. '해당 정보를 찾지 못했어요'라고 솔직히 답해.\n"
        "3. 기숙사 식단과 학생회관 식단을 혼동하지 마.\n"
        "4. 사용자가 점심만 물어보면 점심만 답해.\n"
        "5. 참고 자료의 원본 데이터를 그대로 전달해. 메뉴명, 일정명 등 고유명사에 형용사나 수식어를 절대 추가하지 마. 없는 메뉴나 일정을 지어내지 마.\n"
        "6. 반드시 문법적으로 자연스러운 한국어로만 답변해. 중국어, 영어, 일본어 등 다른 언어를 절대 사용하지 마.\n"
        "7. '이번 학기'는 현재 학기를 의미하고, '다음 학기'는 그 다음 학기를 의미해.\n"
        "8. 공지사항이나 목록을 보여줄 때는 참고 자료에 나온 순서대로(위에서부터) 답변해. 임의로 순서를 바꾸지 마."
    )


_SYSTEM_PROMPT = _build_system_prompt()

# URL → (라벨, 아이콘) 매핑
_SOURCE_LABELS: dict[str, tuple[str, str]] = {
    "computer.cnu.ac.kr": ("컴퓨터융합학부", "💻"),
    "plus.cnu.ac.kr": ("충남대 공식", "🏫"),
    "job.cnu.ac.kr": ("인재개발원", "💼"),
    "sugang.cnu.ac.kr": ("수강신청", "📝"),
    "www.cnucoop.co.kr": ("생활협동조합", "🍽️"),
    "mobileadmin.cnu.ac.kr": ("충남대 식단", "🍱"),
    "portal.cnu.ac.kr": ("학사포털", "🎓"),
    "biz.cnu.ac.kr": ("경영학부", "📊"),
    "economics.cnu.ac.kr": ("경제학과", "📈"),
    "chem.cnu.ac.kr": ("화학과", "🧪"),
    "math.cnu.ac.kr": ("수학과", "🔢"),
    "pharm.cnu.ac.kr": ("약학대학", "💊"),
    "archi.cnu.ac.kr": ("건축학과", "🏛️"),
}


def _strip_think_tags(text: str) -> str:
    """Qwen3 thinking mode 태그를 제거한다."""
    return _THINK_TAG_RE.sub("", text).strip()


def _format_sources(urls: list[str]) -> str:
    """URL 리스트를 HTML 배지 형태로 변환한다."""
    if not urls:
        return ""
    seen: list[tuple[str, str]] = []  # (label, icon)
    for url in urls:
        for domain, (label, icon) in _SOURCE_LABELS.items():
            if domain in url:
                if label not in [s[0] for s in seen]:
                    seen.append((label, icon))
                break
        else:
            from urllib.parse import urlparse

            host = urlparse(url).hostname or url
            short = host.replace("www.", "").split(".")[0]
            if short not in [s[0] for s in seen]:
                seen.append((short, "🔗"))

    badges = " ".join(
        f'<span class="source-chip">{icon} {label}</span>'
        for label, icon in seen
    )
    return f'\n\n<div class="source-row"><span class="source-label">참고</span>{badges}</div>'


def _build_messages(
    question: str, context: str, history: list[dict] | None = None
) -> list[dict]:
    """시스템 프롬프트 + 대화 이력 + 참고자료 + 질문으로 메시지를 구성한다."""
    system_prompt = _build_system_prompt()
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]

    # 최근 3턴(6메시지)의 대화 이력 포함
    if history:
        for turn in history[-6:]:
            msgs.append(turn)

    user_content = (
        f"아래 참고 자료를 반드시 읽고, 참고 자료에 있는 내용만으로 답변해.\n\n"
        f"참고 자료:\n{context}\n\n질문: {question}"
    ) if context else question
    msgs.append({"role": "user", "content": user_content})
    return msgs


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


_chinese_suppressor: list | None = None


def load_model_with_lora(
    model_name: str | None = None,
    adapter_path: str = "models/lora_adapter",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """베이스 모델에 LoRA 어댑터를 합쳐서 로드한다."""
    global _chinese_suppressor
    if model_name is None:
        model_name = _DEFAULT_MODEL
    is_qwen = "qwen" in model_name.lower()
    tokenizer = load_tokenizer(model_name)
    base_model = load_model(model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    if is_qwen:
        print("[inference] 중국어 토큰 억제 필터 구축 중...")
        _chinese_suppressor = _build_chinese_suppressor(tokenizer)
        print(f"[inference] 중국어 토큰 {len(_chinese_suppressor[0]._bad_ids)}개 차단 설정 완료")
    return model, tokenizer


def generate_answer(
    question: str,
    context: str,
    urls: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 512,
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
    messages = _build_messages(question, final_context, history=None)

    tpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if _IS_QWEN:
        tpl_kwargs["enable_thinking"] = False
    text = tokenizer.apply_chat_template(messages, **tpl_kwargs)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": 1.3,
    }
    if _chinese_suppressor:
        gen_kwargs["logits_processor"] = _chinese_suppressor

    torch.manual_seed(_SEED)
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

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
    max_new_tokens: int = 512,
    use_tools: bool = True,
    history: list[dict] | None = None,
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
        history: 이전 대화 이력 (멀티턴 지원)

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

    messages = _build_messages(question, final_context, history=history)

    tpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if _IS_QWEN:
        tpl_kwargs["enable_thinking"] = False
    text = tokenizer.apply_chat_template(messages, **tpl_kwargs)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": 1.3,
        "streamer": streamer,
    }
    if _chinese_suppressor:
        gen_kwargs["logits_processor"] = _chinese_suppressor

    torch.manual_seed(_SEED)
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    accumulated = ""
    for chunk in streamer:
        accumulated += chunk
        cleaned = _strip_think_tags(accumulated)
        if _IS_QWEN:
            cleaned = _remove_chinese(cleaned)
        yield cleaned

    thread.join()

    # 최종 정리 + 출처 추가
    final = _strip_think_tags(accumulated)
    if _IS_QWEN:
        final = _remove_chinese(final)
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
