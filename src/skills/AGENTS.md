# src/skills — 캠퍼스 스킬 패키지

각 캠퍼스 기능(식단·셔틀·학사일정·공지...)을 **자기완결적 스킬 모듈** 하나로 둔다.
레지스트리가 자동 수집하므로, 라우팅/실행은 등록된 스킬만 보고 동작한다.

## 구조

- `base.py` — `Skill` 데이터클래스 + 공용 HTTP `SESSION`(재시도·학술 UA) + `now_kst()`
- `registry.py` — 스킬 자동 import/등록, `detect(질문)→스킬`, `run_skill(이름, 인자)`
- `meal.py` / `shuttle.py` / `calendar.py` / `notice.py` — 실제 스킬

호환용 shim: `src/tools/detector.py(detect_tool)`, `src/tools/definitions.py(execute_tool)`
→ 둘 다 이 패키지로 위임한다. 신규 코드는 `from src.skills import detect, run_skill` 사용.

## 새 스킬 추가 (파일 하나 + 한 줄)

1. `src/skills/<이름>.py` 작성:

```python
from src.skills.base import Skill, SESSION, now_kst
from src.skills.registry import register

def get_xxx(arg: str | None = None) -> str:
    ...  # 라이브 크롤링 또는 정적 텍스트 반환

def _infer_args(question: str) -> dict:   # 인자가 필요할 때만
    return {}

register(Skill(
    name="get_xxx",
    description="...",
    keywords=["키워드1", "키워드2"],
    negative_keywords=["오발동 방지어"],   # 선택
    run=get_xxx,
    infer_args=_infer_args,               # 선택 (기본: 인자 없음)
))
```

2. `registry.py` 의 `_SKILL_MODULES` 에 `"<이름>"` 한 줄 추가.

끝. `detect()`/`run_skill()` 이 자동으로 새 스킬을 라우팅에 포함한다.

## 라우팅 규칙

- 질문에 키워드가 포함되면 매칭. 여러 스킬이 걸리면 **가장 긴 키워드**의 스킬 우선.
- `negative_keywords` 가 하나라도 걸리면 그 스킬은 매칭 제외(오발동 방지).
- 매칭이 없으면 `(None, {})` → 호출부에서 RAG 검색으로 폴백.

## 주의

- 새 라이브 크롤링 스킬은 반드시 `robots.txt` 확인 + `SESSION`(공용 UA) 사용 +
  요청 간 `CRAWL_DELAY`(3초) 이상. 도메인 추가는 CLAUDE.md 수집 범위 정책을 따른다.
- 장학금·커리큘럼처럼 **정형 라이브 소스가 없는 정보는 스킬로 만들지 말고 RAG** 에 맡긴다
  (검색 컨텍스트로 답변). 스킬은 "실시간·구조화된" 정보 전용.
