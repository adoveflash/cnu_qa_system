"""robots.txt 허용 여부 확인 유틸리티.

학교 서버가 자체 서명 인증서를 사용하므로 urllib.robotparser의 내장 fetch 대신
requests(verify=False)로 robots.txt를 내려받은 뒤 parse()에 직접 feed한다.
"""

import urllib.robotparser
from functools import lru_cache
from urllib.parse import urlparse

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

USER_AGENT = "CNU-NLP-StudentProject/1.0 (academic-coursework)"


@lru_cache(maxsize=32)
def _get_parser(robots_url: str) -> urllib.robotparser.RobotFileParser:
    """robots.txt를 requests로 내려받아 파싱한 파서 객체를 반환한다 (도메인당 1회 캐시)."""
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        resp = requests.get(
            robots_url,
            timeout=15,
            verify=False,
            headers={"User-Agent": USER_AGENT},
        )
        if resp.status_code == 200:
            # modified()로 last_checked 설정 필수 — 없으면 can_fetch()가 항상 False
            rp.modified()
            rp.parse(resp.text.splitlines())
        elif resp.status_code in (401, 403):
            rp.disallow_all = True
        else:
            # 404 등 → robots.txt 없음 → 전체 허용
            rp.allow_all = True
    except Exception:
        # 연결 실패 → 전체 허용으로 간주
        rp.allow_all = True
    return rp


# 학술 목적 수집이 승인된 경로 (CLAUDE.md 참조)
_ALLOWED_OVERRIDES: list[str] = [
    "plus.cnu.ac.kr/html/kr/",
    "plus.cnu.ac.kr/html/hub/",
]


def is_allowed(url: str) -> bool:
    """주어진 URL이 robots.txt 기준 크롤링 허용 대상인지 반환한다.

    Args:
        url: 확인할 페이지 URL

    Returns:
        크롤링 허용이면 True, Disallow이면 False
    """
    # 학술 목적 승인된 경로는 허용
    for override in _ALLOWED_OVERRIDES:
        if override in url:
            return True

    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = _get_parser(robots_url)
    return rp.can_fetch(USER_AGENT, url)
