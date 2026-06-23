"""캠퍼스 스킬 패키지.

각 캠퍼스 기능을 자기완결적 Skill 모듈로 두고, 레지스트리가 자동 수집한다.

공개 API:
    detect(question) -> (Skill | None, args)   질문 라우팅
    run_skill(name, args) -> str               이름으로 실행
    all_skills() -> list[Skill]                등록된 스킬 목록
"""

from src.skills.base import Skill
from src.skills.registry import all_skills, detect, register, run_skill

__all__ = ["Skill", "all_skills", "detect", "register", "run_skill"]
