"""스킬(skill) 공통 토대 — Skill 정의 + 공용 HTTP 세션/시간 헬퍼.

각 캠퍼스 기능(식단·셔틀·학사일정·공지...)을 하나의 Skill 로 캡슐화한다.
새 기능은 `src/skills/<이름>.py` 파일 하나에 Skill 을 만들고 register()만 하면
레지스트리가 자동으로 라우팅에 포함시킨다 (definitions/detector를 안 건드림).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

import requests
from bs4 import BeautifulSoup  # noqa: F401 - 스킬 모듈들이 base에서 가져다 쓰도록 재노출

KST = timezone(timedelta(hours=9))

USER_AGENT = "CNU-NLP-StudentProject/1.0 (academic-coursework)"
CRAWL_DELAY = 3.0


def now_kst() -> datetime:
    """KST 기준 현재 시각."""
    return datetime.now(KST)


def _build_session() -> requests.Session:
    """학교 서버용 공용 세션 (재시도 + 학술용 User-Agent)."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.verify = False  # 학교 사이트 인증서 체인 이슈 회피
    retry_cfg = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=1.0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_cfg)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# 모든 스킬이 공유하는 단일 세션 (커넥션 풀 재사용)
SESSION = _build_session()


def _no_args(_question: str) -> dict:
    """인자가 필요 없는 스킬용 기본 추론기."""
    return {}


@dataclass
class Skill:
    """캠퍼스 기능 하나를 표현하는 스킬.

    Attributes:
        name: 스킬 식별자 (기존 tool 이름과 호환 — get_meal_menu 등)
        description: 스킬 설명 (발표/디버그용)
        keywords: 이 스킬을 발동시키는 질문 키워드
        negative_keywords: 오발동을 막는 제외 키워드
        infer_args: 질문에서 실행 인자(dict)를 추론하는 함수
        run: 실제 실행 함수 (infer_args 결과를 **kwargs 로 받음)
    """

    name: str
    description: str
    keywords: list[str]
    run: Callable[..., str]
    negative_keywords: list[str] = field(default_factory=list)
    infer_args: Callable[[str], dict] = _no_args

    def match_length(self, question: str) -> int | None:
        """질문에 매칭되면 가장 긴 매칭 키워드 길이를, 아니면 None 을 반환한다.

        negative_keyword 가 하나라도 걸리면 매칭 실패로 처리한다.
        길이를 반환하는 이유: 여러 스킬이 동시에 걸릴 때 더 구체적(긴) 키워드 우선.
        """
        if any(neg in question for neg in self.negative_keywords):
            return None
        best: int | None = None
        for kw in self.keywords:
            if kw in question and (best is None or len(kw) > best):
                best = len(kw)
        return best

    def execute(self, question: str) -> str:
        """질문에서 인자를 추론해 스킬을 실행한다."""
        return self.run(**self.infer_args(question))
