# src/skills — 캠퍼스 스킬 패키지

각 캠퍼스 기능(식단·셔틀·학사일정·공지...)을 **자기완결적 스킬 모듈** 하나로 둔다.
레지스트리가 자동 수집하므로, 라우팅/실행은 등록된 스킬만 보고 동작한다.

## 구조

- `base.py` — `Skill` 데이터클래스 + 공용 HTTP `SESSION`(재시도·학술 UA) + `now_kst()`
- `router.py` — **LLM tool-calling 라우터**(`route_llm`). 기본 라우팅 경로.
- `registry.py` — 스킬 자동 import/등록, `detect(질문)→스킬`(키워드 폴백), `run_skill(이름, 인자)`
- `meal.py` / `shuttle.py` / `calendar.py` / `notice.py` — 실제 스킬

호환용 shim: `src/tools/detector.py(detect_tool)`, `src/tools/definitions.py(execute_tool)`
→ 둘 다 이 패키지의 **키워드 detect**로 위임한다(레거시·노트북 인라인 사본용).
신규 코드는 `from src.skills import detect, run_skill` + `from src.skills.router import route_llm` 사용.

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
    description="...",                      # LLM 라우터가 읽는 도구 설명 — 명확하게
    keywords=["키워드1", "키워드2"],         # 키워드 폴백용
    negative_keywords=["오발동 방지어"],   # 선택 (키워드 폴백 전용)
    parameters={                           # LLM tool-calling 인자 스키마 (인자 있을 때만)
        "arg": {"type": "string", "description": "...", "enum": [...]},  # enum 선택
    },
    run=get_xxx,
    infer_args=_infer_args,               # 선택, 키워드 폴백 전용 (기본: 인자 없음)
))
```

2. `registry.py` 의 `_SKILL_MODULES` 에 `"<이름>"` 한 줄 추가.

끝. `route_llm()`(LLM)과 `detect()`(키워드 폴백)이 자동으로 새 스킬을 포함한다.

## 라우팅 규칙

기본은 **LLM tool-calling**(`router.route_llm`): Gemma가 `description`+`parameters`를 읽고
호출할 도구와 인자를 직접 고른다. "1학"(제1학생회관) 같은 변형·줄임말을 의미로 인식한다.

- `route_llm` 반환 tri-state: `("get_xxx", args)` 도구선택 / `(None, {})` 도구불필요(LLM 판단) /
  `None` 파싱실패 → 호출부가 **키워드 `detect()`로 폴백**.
- 키워드 `detect()`(폴백): 질문에 키워드 포함 시 매칭, 여러 개면 **가장 긴 키워드** 우선,
  `negative_keywords` 걸리면 제외. 매칭 없으면 `(None, {})` → RAG 폴백.
- 즉 키워드(`keywords`/`negative_keywords`/`infer_args`)는 이제 **안전망**이지 1차 경로가 아니다.

## 주의

- 새 라이브 크롤링 스킬은 반드시 `robots.txt` 확인 + `SESSION`(공용 UA) 사용 +
  요청 간 `CRAWL_DELAY`(3초) 이상. 도메인 추가는 CLAUDE.md 수집 범위 정책을 따른다.
- 장학금·커리큘럼처럼 **정형 라이브 소스가 없는 정보는 스킬로 만들지 말고 RAG** 에 맡긴다
  (검색 컨텍스트로 답변). 스킬은 "실시간·구조화된" 정보 전용.
