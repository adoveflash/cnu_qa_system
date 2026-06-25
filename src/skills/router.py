"""LLM 기반 스킬 라우터 — 키워드 매칭을 대체한다.

기존 키워드 라우팅(registry.detect)은 "1학"(제1학생회관 줄임말) 같은 변형을 인식 못 하고,
의미가 아니라 사람이 하드코딩한 키워드로만 발동했다. 여기서는 LLM(Gemma)이 질문을 읽고
등록된 스킬 중 호출할 도구와 인자를 직접 고른다.

반환 규약 (tri-state):
    - ("get_xxx", {...})  : 도구 선택 (인자는 parameters 스키마로 검증/보정)
    - (None, {})          : 도구 불필요 — LLM이 명시적으로 판단 (RAG로)
    - None                : 파싱 실패 — 호출부가 키워드 detect()로 폴백
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import torch

from src.skills.registry import all_skills

_SEED = 42
_KST = timezone(timedelta(hours=9))
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def build_tool_specs() -> list[dict]:
    """등록된 스킬을 LLM 라우터용 도구 스펙 리스트로 변환한다."""
    specs = []
    for s in all_skills():
        specs.append(
            {"name": s.name, "description": s.description, "parameters": s.parameters or {}}
        )
    return specs


def _today_str() -> str:
    """오늘 날짜 문자열 (상대 날짜 해석용)."""
    days = ["월", "화", "수", "목", "금", "토", "일"]
    now = datetime.now(_KST)
    return f"{now.strftime('%Y-%m-%d')} ({days[now.weekday()]}요일)"


def _routing_messages(question: str) -> list[dict]:
    """라우팅 판단용 메시지를 구성한다 (도구 목록 + 오늘 날짜 + 질문)."""
    tools_json = json.dumps(build_tool_specs(), ensure_ascii=False, indent=2)
    system = (
        "너는 충남대 챗봇의 도구 라우터다. 사용자 질문에 답하려면 아래 도구 중 하나를 "
        "호출해야 하는지 판단한다. 도구는 '실시간·구조화된' 정보(식단·셔틀·학사일정·공지) "
        "전용이다. 졸업요건·장학금·일반 상식처럼 도구가 필요 없으면 tool을 null로 둔다.\n\n"
        f"오늘 날짜: {_today_str()}. 날짜 인자는 이 값 기준으로 YYYY-MM-DD로 채운다.\n\n"
        f"사용 가능한 도구:\n{tools_json}\n\n"
        "반드시 JSON 객체 하나만 출력한다. 설명·마크다운·코드펜스 금지.\n"
        '형식: {"tool": "도구이름" 또는 null, "args": {인자: 값}}\n'
        '예: {"tool": "get_meal_menu", "args": {"location": "dormitory"}}\n'
        '예: {"tool": null, "args": {}}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"질문: {question}"},
    ]


def _render(tokenizer, messages: list[dict]) -> str:
    """chat 템플릿을 적용한다 (system role 미지원 모델은 user에 합쳐 재시도)."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        sys_text = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user_text = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        folded = [{"role": "user", "content": f"{sys_text}\n\n{user_text}"}]
        return tokenizer.apply_chat_template(folded, tokenize=False, add_generation_prompt=True)


def _generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 128) -> str:
    """라우팅 판단을 짧게 그리디 생성해 디코드 텍스트를 반환한다."""
    text = _render(tokenizer, messages)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    torch.manual_seed(_SEED)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True)


def _extract_json(text: str) -> dict | None:
    """생성 텍스트에서 첫 JSON 객체를 추출해 파싱한다 (think 토큰·잡설 무시)."""
    match = _JSON_OBJ.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def coerce_args(parameters: dict, raw_args: dict) -> dict:
    """LLM이 준 인자를 parameters 스키마로 검증·타입 보정한다.

    스키마에 없는 키는 버리고, integer 타입은 int로 캐스팅하며, None/빈값은 제외한다.
    """
    out: dict = {}
    for key, spec in (parameters or {}).items():
        if key not in raw_args:
            continue
        val = raw_args[key]
        if val is None or val == "":
            continue
        if spec.get("type") == "integer":
            try:
                val = int(val)
            except (TypeError, ValueError):
                continue
        out[key] = val
    return out


def route_llm(question: str, model, tokenizer) -> tuple[str | None, dict] | None:
    """LLM으로 질문에 맞는 스킬과 인자를 고른다.

    Args:
        question: 사용자 질문
        model: LLM (Gemma 등)
        tokenizer: 토크나이저

    Returns:
        ("get_xxx", args)  도구 선택 / (None, {}) 도구 불필요 / None 파싱 실패(키워드 폴백).
    """
    try:
        decoded = _generate(model, tokenizer, _routing_messages(question))
    except Exception:  # noqa: BLE001 - 생성 실패는 키워드 폴백으로 흡수
        return None

    obj = _extract_json(decoded)
    if obj is None or "tool" not in obj:
        return None  # 파싱 실패 → 키워드 폴백

    tool = obj.get("tool")
    if tool is None or tool == "null":
        return None, {}  # LLM이 명시적으로 도구 불필요라고 판단

    valid = {s.name: s for s in all_skills()}
    skill = valid.get(tool)
    if skill is None:
        return None  # 모르는 도구명 → 폴백

    raw_args = obj.get("args") or {}
    args = coerce_args(skill.parameters, raw_args) if isinstance(raw_args, dict) else {}
    return tool, args
