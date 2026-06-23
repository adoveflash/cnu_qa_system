"""[호환 shim] 키워드 기반 tool 감지 → 스킬 레지스트리로 위임.

라우팅 로직은 `src/skills/` 의 각 스킬 모듈로 이동했다.
기존 호출부(inference 등)가 깨지지 않도록 detect_tool() 인터페이스만 유지한다.
새 코드는 `from src.skills import detect` 를 직접 사용한다.
"""

from __future__ import annotations

from src.skills import detect


def detect_tool(question: str) -> tuple[str | None, dict]:
    """질문에서 스킬을 감지한다 (스킬 레지스트리 위임).

    Args:
        question: 사용자 질문

    Returns:
        (스킬 이름, 인자) 튜플. 매칭이 없으면 (None, {}).
    """
    skill, args = detect(question)
    return (skill.name if skill else None), args
