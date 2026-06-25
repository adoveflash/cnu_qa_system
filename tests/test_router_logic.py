"""LLM 라우터 파싱·tri-state 로직 검증 (Gemma 불필요).

router._generate(모델 생성)를 가짜 출력으로 몽키패치해, JSON 추출·인자 보정·
tri-state 반환(도구/none/파싱실패)을 모델 없이 검증한다.

실행: python tests/test_router_logic.py
"""

import sys

sys.path.insert(0, ".")  # noqa: E402

from src.skills import router  # noqa: E402
from src.skills.router import build_tool_specs, coerce_args, _extract_json  # noqa: E402


def _patch_output(text: str) -> None:
    """라우터의 모델 생성을 고정 문자열로 대체한다."""
    router._generate = lambda model, tokenizer, messages, max_new_tokens=128: text


def test_build_tool_specs() -> None:
    """등록 스킬이 name/description/parameters 스펙으로 노출된다."""
    specs = {s["name"]: s for s in build_tool_specs()}
    assert {"get_meal_menu", "get_shuttle_schedule", "get_academic_calendar", "get_notices"} <= set(
        specs
    )
    assert "location" in specs["get_meal_menu"]["parameters"]
    assert specs["get_shuttle_schedule"]["parameters"] == {}  # 무인자 도구
    print("OK build_tool_specs: 4개 도구 + parameters 노출")


def test_extract_json_ignores_noise() -> None:
    """think 토큰·잡설이 앞뒤에 붙어도 JSON 객체만 추출한다."""
    txt = '<think>고민</think> 결과: {"tool": "get_notices", "args": {"count": 3}} 끝'
    assert _extract_json(txt) == {"tool": "get_notices", "args": {"count": 3}}
    assert _extract_json("도구 없음") is None
    print("OK extract_json: 잡음 무시 + 실패시 None")


def test_coerce_args_types() -> None:
    """integer 인자는 캐스팅, 스키마 밖/빈값은 제거한다."""
    params = {"count": {"type": "integer"}}
    assert coerce_args(params, {"count": "3", "bogus": "x"}) == {"count": 3}
    assert coerce_args(params, {"count": ""}) == {}
    assert coerce_args(params, {"count": "abc"}) == {}  # 캐스팅 실패는 드롭
    print("OK coerce_args: 타입 보정 + 스키마 밖 제거")


def test_route_tool_selected() -> None:
    """LLM이 도구를 고르면 (이름, 보정된 인자)를 반환한다."""
    _patch_output('{"tool": "get_meal_menu", "args": {"location": "dormitory"}}')
    assert router.route_llm("기숙사 밥 뭐야", None, None) == (
        "get_meal_menu",
        {"location": "dormitory"},
    )
    print("OK route_llm: 도구 선택 → (이름, 인자)")


def test_route_explicit_none() -> None:
    """LLM이 도구 불필요라고 판단하면 (None, {}) — 키워드 폴백 안 함."""
    _patch_output('{"tool": null, "args": {}}')
    assert router.route_llm("졸업학점 몇 점이야", None, None) == (None, {})
    print("OK route_llm: 명시적 none → (None, {})")


def test_route_parse_fail_returns_none() -> None:
    """JSON 파싱 실패/모르는 도구는 None 반환 → 호출부가 키워드 폴백."""
    _patch_output("도구를 못 고르겠어요")
    assert router.route_llm("셔틀 언제 와", None, None) is None
    _patch_output('{"tool": "get_unknown", "args": {}}')
    assert router.route_llm("뭐든", None, None) is None
    print("OK route_llm: 파싱실패/미지의 도구 → None(폴백 신호)")


if __name__ == "__main__":
    test_build_tool_specs()
    test_extract_json_ignores_noise()
    test_coerce_args_types()
    test_route_tool_selected()
    test_route_explicit_none()
    test_route_parse_fail_returns_none()
    print("\n=== 라우터 로직 검증 통과 ===")
