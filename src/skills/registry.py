"""스킬 레지스트리 — 스킬 자동 수집 + 질문 라우팅 + 실행.

스킬 모듈(meal/shuttle/calendar/notice...)을 import 하면 각 모듈이 register()로
자기 자신을 여기 등록한다. 라우팅/실행은 등록된 스킬만 보고 동작하므로,
새 스킬은 모듈 파일 하나 추가 → _SKILL_MODULES 에 이름 한 줄 추가로 끝난다.
"""

from __future__ import annotations

import importlib

from src.skills.base import Skill

# 패키지 로드시 import 할 스킬 모듈 목록 (각 모듈이 register() 호출)
_SKILL_MODULES = ["meal", "shuttle", "calendar", "notice"]

_REGISTRY: dict[str, Skill] = {}
_LOADED = False


def register(skill: Skill) -> None:
    """스킬을 레지스트리에 등록한다 (중복 이름은 덮어쓰기)."""
    _REGISTRY[skill.name] = skill


def _ensure_loaded() -> None:
    """스킬 모듈들을 1회 import 해 자동 등록을 트리거한다."""
    global _LOADED
    if _LOADED:
        return
    for mod in _SKILL_MODULES:
        importlib.import_module(f"src.skills.{mod}")
    _LOADED = True


def all_skills() -> list[Skill]:
    """등록된 모든 스킬을 반환한다."""
    _ensure_loaded()
    return list(_REGISTRY.values())


def detect(question: str) -> tuple[Skill | None, dict]:
    """질문에 가장 잘 맞는 스킬과 추론된 인자를 반환한다.

    여러 스킬이 매칭되면 가장 구체적인(긴 키워드) 스킬을 고른다.

    Returns:
        (스킬, 인자). 매칭이 없으면 (None, {}).
    """
    _ensure_loaded()
    best: tuple[Skill, int] | None = None
    for skill in _REGISTRY.values():
        length = skill.match_length(question)
        if length is None:
            continue
        if best is None or length > best[1]:
            best = (skill, length)
    if best is None:
        return None, {}
    skill = best[0]
    return skill, skill.infer_args(question)


def run_skill(name: str, arguments: dict | None = None) -> str:
    """이름으로 스킬을 찾아 실행한다.

    Args:
        name: 스킬 이름
        arguments: 실행 인자 (None이면 빈 dict)

    Returns:
        실행 결과 텍스트. 알 수 없는 스킬이면 안내 문자열.
    """
    _ensure_loaded()
    skill = _REGISTRY.get(name)
    if skill is None:
        return f"알 수 없는 스킬: {name}"
    try:
        return skill.run(**(arguments or {}))
    except Exception as e:  # noqa: BLE001 - 실행 실패는 메시지로 흡수
        return f"스킬 실행 오류 ({name}): {e}"
