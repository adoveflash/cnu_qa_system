"""추론 모듈.

RAG 컨텍스트와 질문을 받아 답변을 생성한다.
키워드 기반 tool 감지로 실시간 정보(식단, 셔틀, 학사일정, 공지)를 조회할 수 있다.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator

import torch
from transformers import TextIteratorStreamer

from src.model.base import load_model, load_tokenizer, _DEFAULT_MODEL
from src.tools.definitions import execute_tool
from src.tools.detector import detect_tool

_SEED = 42


def _clean_thinking(text: str) -> str:
    """Gemma 4 thinking 토큰/쓰레기 제거."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?(?=안녕하세요)", "", text, flags=re.DOTALL)
    if text and not re.match(r"[가-힣*#\-]", text):
        match = re.search(r"[가-힣]", text)
        if match:
            text = text[match.start():]
    return text.strip()


def _strip_context_markers(text: str) -> str:
    """답변에 새어나온 내부 참고자료 마커를 제거한다.

    `build_context`가 청크를 '[참고N]'으로 묶는데 모델이 이를 그대로 인용하는 경우가 있어
    사용자에게 무의미하게 노출된다. 안전하게 제거 가능한 '[참고N]' 패턴만 지운다.
    ('제공된 자료에 따르면' 같은 문장형 메타 표현은 문법을 깰 수 있어 프롬프트로 예방한다.)

    Args:
        text: 생성된 답변

    Returns:
        마커가 제거된 답변
    """
    text = re.sub(r"\[\s*참고\s*\d+\s*\]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fold_system_into_user(messages: list[dict]) -> list[dict]:
    """system 메시지를 다음 user 메시지 앞에 합친다 (Gemma 3 등 system role 미지원 모델용)."""
    system_text = ""
    out: list[dict] = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        text = content if isinstance(content, str) else ""
        if role == "system":
            system_text += text + "\n\n"
        elif role == "user" and system_text:
            out.append({"role": "user", "content": system_text + text})
            system_text = ""
        else:
            out.append(m)
    return out


def _render_prompt(tokenizer, messages: list[dict]) -> str:
    """chat 템플릿을 적용한다.

    일부 모델(Gemma 3 등)은 system role / enable_thinking 인자를 지원하지 않는다.
    실패하면 system을 user에 합치고 enable_thinking 없이 재시도해 모델 호환성을 넓힌다.
    Gemma 4는 첫 시도에서 그대로 통과한다.
    """
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        folded = _fold_system_into_user(messages)
        return tokenizer.apply_chat_template(
            folded, tokenize=False, add_generation_prompt=True
        )


def _build_system_prompt() -> str:
    """오늘 날짜와 현재 학기를 포함한 시스템 프롬프트를 생성한다."""
    from datetime import datetime, timezone, timedelta

    KST = timezone(timedelta(hours=9))
    now = datetime.now(KST)
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
        "8. 공지사항이나 목록을 보여줄 때는 참고 자료에 나온 순서대로(위에서부터) 답변해. 임의로 순서를 바꾸지 마.\n"
        "9. 졸업요건·졸업학점은 학과마다 다르다(예: 건축학과 등 일부 학과는 130학점이 아님). "
        "참고 자료가 특정 학과(예: 컴퓨터융합학부·인공지능학과)에 관한 것이면 그 학과에 한정해서 답하고, "
        "사용자가 묻는 학과의 자료가 없으면 다른 학과의 수치를 끌어다 쓰지 말고 "
        "'해당 학과의 정확한 졸업요건은 확인되지 않았어요'라고 답해.\n"
        "10. 답변에 '[참고1]', '[참고2]' 같은 자료 번호나 '제공된 자료/입력하신 데이터에 따르면' "
        "같은 내부 참고자료 언급을 절대 쓰지 마. 참고자료의 존재를 드러내지 말고 사용자에게 "
        "바로 자연스럽게 정보를 전달해."
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

    badges = " ".join(f'<span class="source-chip">{icon} {label}</span>' for label, icon in seen)
    return f'\n\n<div class="source-row"><span class="source-label">참고</span>{badges}</div>'


def _build_messages(question: str, context: str, history: list[dict] | None = None) -> list[dict]:
    """시스템 프롬프트 + 대화 이력 + 참고자료 + 질문으로 메시지를 구성한다."""
    system_prompt = _build_system_prompt()
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]

    # 최근 3턴(6메시지)의 대화 이력 포함
    if history:
        for turn in history[-6:]:
            msgs.append(turn)

    user_content = (
        (
            f"아래 참고 자료를 반드시 읽고, 참고 자료에 있는 내용만으로 답변해.\n\n"
            f"참고 자료:\n{context}\n\n질문: {question}"
        )
        if context
        else question
    )
    msgs.append({"role": "user", "content": user_content})
    return msgs


_TOOL_FAIL_KEYWORDS = ["찾을 수 없습니다", "가져올 수 없습니다", "조회 실패"]


def _resolve_context(
    question: str, rag_context: str, urls: list[str], use_tools: bool
) -> tuple[str, list[str], bool]:
    """질문에 대해 tool 또는 RAG 컨텍스트를 결정한다.

    tool 결과가 실패 메시지면 RAG fallback으로 전환한다.

    Returns:
        (final_context, final_urls, used_tool)
    """
    if use_tools:
        tool_name, tool_args = detect_tool(question)
        if tool_name:
            print(f"  [tool] {tool_name}({tool_args})")
            result = execute_tool(tool_name, tool_args)
            # tool 실패 시 RAG fallback
            if any(kw in result for kw in _TOOL_FAIL_KEYWORDS):
                print("  [tool] 실패 → RAG fallback")
                return rag_context, urls, False
            tool_context = f"[{tool_name} 결과]\n{result}"
            return tool_context, [], True
    return rag_context, urls, False


def load_inference_model(
    model_name: str | None = None,
) -> tuple:
    """추론용 모델과 토크나이저를 로드한다."""
    if model_name is None:
        model_name = _DEFAULT_MODEL
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)
    print(f"[inference] 모델 로드 완료: {model_name}")
    return model, tokenizer


def generate_answer(
    question: str,
    context: str,
    urls: list[str],
    model,
    tokenizer,
    max_new_tokens: int = 1024,
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
    final_context, final_urls, used_tool = _resolve_context(question, context, urls, use_tools)
    messages = _build_messages(question, final_context, history=None)

    text = _render_prompt(tokenizer, messages)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": 1.3,
    }

    torch.manual_seed(_SEED)
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    answer = _clean_thinking(answer)
    answer = _strip_context_markers(answer)
    answer = answer.replace("\r", "").replace("~", r"\~")

    if not used_tool:
        answer += _format_sources(final_urls)
    return answer


def generate_answer_stream(
    question: str,
    context: str,
    urls: list[str],
    model,
    tokenizer,
    max_new_tokens: int = 1024,
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

    text = _render_prompt(tokenizer, messages)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": 1.3,
        "streamer": streamer,
    }

    torch.manual_seed(_SEED)
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    accumulated = ""
    for chunk in streamer:
        accumulated += chunk
        yield accumulated.replace("~", r"\~")

    thread.join()

    # 최종 정리 + 출처 추가
    final = _clean_thinking(accumulated)
    final = _strip_context_markers(final)
    final = final.replace("\r", "").replace("~", r"\~")
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
