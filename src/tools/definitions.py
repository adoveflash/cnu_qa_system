"""[호환 shim] tool 실행 → 스킬 레지스트리로 위임.

실제 구현(식단·셔틀·학사일정·공지 크롤러)은 `src/skills/` 의 각 스킬 모듈로
이동했다. 기존 호출부가 깨지지 않도록 execute_tool() 인터페이스만 유지한다.
새 코드는 `from src.skills import run_skill` 을 직접 사용한다.
"""

from __future__ import annotations

from src.skills import run_skill


def execute_tool(name: str, arguments: dict) -> str:
    """이름으로 스킬을 실행한다 (스킬 레지스트리 위임).

    Args:
        name: 스킬 이름
        arguments: 실행 인자 딕셔너리

    Returns:
        실행 결과 텍스트
    """
    return run_skill(name, arguments)
