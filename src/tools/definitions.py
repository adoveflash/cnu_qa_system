"""Tool calling 정의 및 실행 모듈.

Qwen3 네이티브 tool calling을 위한 tool 스키마와 실행 함수를 정의한다.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

USER_AGENT = "CNU-NLP-StudentProject/1.0 (academic-coursework)"
CRAWL_DELAY = 3.0

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": USER_AGENT})
_SESSION.verify = False

# ── Tool JSON Schema (Qwen3 format) ────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_meal_menu",
            "description": "충남대학교 교내 식당의 오늘 또는 이번 주 식단 메뉴를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "조회할 날짜 (YYYY-MM-DD). 생략 시 오늘 날짜.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_shuttle_schedule",
            "description": "충남대학교 셔틀버스 시간표와 노선 정보를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_academic_calendar",
            "description": (
                "충남대학교 학사일정을 조회합니다. "
                "수강신청, 개강, 중간고사, 기말고사, 방학 등의 일정을 확인할 수 있습니다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "month": {
                        "type": "integer",
                        "description": "조회할 월 (1-12). 생략 시 전체 학사일정.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_notices",
            "description": "충남대학교 컴퓨터융합학부 최신 공지사항을 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "가져올 공지 수 (기본 5, 최대 10).",
                    }
                },
                "required": [],
            },
        },
    },
]


# ── Tool 실행 함수 ──────────────────────────────────────────────────────────


def _fetch_dorm_meal(date: str) -> str:
    """기숙사 식단을 크롤링한다 (dorm.cnu.ac.kr).

    Args:
        date: YYYY-MM-DD 형식 날짜

    Returns:
        정리된 기숙사 식단 텍스트
    """
    try:
        resp = _SESSION.get(
            "https://dorm.cnu.ac.kr/html/kr/sub03/sub03_0304.html",
            timeout=30,
        )
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # 날짜에서 일(day) 추출 — 페이지는 "6(금)" 같은 형식 사용
        dt = datetime.strptime(date, "%Y-%m-%d")
        day_num = dt.day
        weekday_kr = ["월", "화", "수", "목", "금", "토", "일"][dt.weekday()]
        target_label = f"{day_num}({weekday_kr})"

        body = soup.find("body")
        if not body:
            return ""

        text = body.get_text(separator="\n", strip=True)
        lines = text.split("\n")

        # 해당 날짜 섹션 찾기
        capture = False
        day_lines: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 해당 날짜 시작
            if target_label in line:
                capture = True
                day_lines.append(f"--- {date} ({weekday_kr}) ---")
                continue
            # 다음 날짜가 시작되면 중단
            if capture and re.match(r"^\d{1,2}\([월화수목금토일]\)", line):
                break
            if capture:
                day_lines.append(line)

        if day_lines:
            return "[기숙사 식단]\n" + "\n".join(day_lines[:40])
    except Exception as e:
        return f"[기숙사] 조회 실패: {e}"
    return ""


def _fetch_student_hall_meal(date: str) -> str:
    """학생회관 식단을 playwright로 크롤링한다 (mobileadmin.cnu.ac.kr).

    JS 렌더링이 필요하므로 playwright를 사용한다.

    Args:
        date: YYYY-MM-DD 형식 날짜

    Returns:
        정리된 학생회관 식단 텍스트
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [tool] playwright 미설치 — pip install playwright && playwright install chromium")
        return "[학생회관] playwright 미설치. 학생회관 식단 조회 불가."

    date_str = date.replace("-", "")
    url = f"https://mobileadmin.cnu.ac.kr/food/index.jsp?searchYmd={date_str}"
    print(f"  [tool] 학생회관 식단 크롤링: {url}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(url, timeout=30000)
            # JS 렌더링 대기 — 식단 테이블이 로드될 때까지
            page.wait_for_timeout(5000)

            # 전체 페이지 텍스트 추출 (테이블 포함)
            content = page.content()
            browser.close()

        soup = BeautifulSoup(content, "html.parser")
        for tag in soup.find_all(["script", "style"]):
            tag.decompose()

        tables = soup.find_all("table")
        parts: list[str] = []
        for table in tables:
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                line = " | ".join(c for c in cells if c)
                if line and "운영안함" not in line:
                    parts.append(line)

        print(f"  [tool] 학생회관 식단: {len(parts)}행 추출")

        if parts:
            return f"[학생회관 식단 {date}]\n" + "\n".join(parts[:50])
        else:
            return f"[학생회관] {date} 식단 데이터 없음 (주말 또는 미운영)"
    except Exception as e:
        print(f"  [tool] 학생회관 조회 실패: {e}")
        return f"[학생회관] 조회 실패: {e}"


def get_meal_menu(date: str | None = None) -> str:
    """교내 식당 식단을 크롤링하여 반환한다.

    기숙사 식단(dorm.cnu.ac.kr)과 학생회관 식단(mobileadmin.cnu.ac.kr)을 조회한다.

    Args:
        date: 조회 날짜 (YYYY-MM-DD). None이면 오늘.

    Returns:
        식단 정보 텍스트
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    results: list[str] = []

    # 1) 기숙사 식단
    dorm = _fetch_dorm_meal(date)
    if dorm:
        results.append(dorm)

    time.sleep(CRAWL_DELAY)

    # 2) 학생회관 식단
    hall = _fetch_student_hall_meal(date)
    if hall:
        results.append(hall)

    if not results:
        return f"{date} 식단 정보를 가져올 수 없습니다. 주말이거나 운영하지 않는 날일 수 있습니다."
    return "\n\n".join(results)


def get_shuttle_schedule() -> str:
    """셔틀버스 시간표를 반환한다.

    Returns:
        셔틀버스 시간표 텍스트
    """
    # 크롤링 데이터에서 로드 (가장 최신 파일)
    raw_dir = Path("data/corpus/raw")
    shuttle_files = sorted(raw_dir.glob("shuttle_*.jsonl"), reverse=True)

    if shuttle_files:
        results: list[str] = []
        with open(shuttle_files[0], encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                results.append(f"[{doc['title']}]\n{doc['content'][:2000]}")
        if results:
            return "\n\n".join(results)

    # 파일 없으면 실시간 크롤링
    try:
        url = "https://plus.cnu.ac.kr/html/kr/sub05/sub05_050403.html"
        resp = _SESSION.get(url, timeout=30)
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        body = soup.find("body")
        if body:
            text = body.get_text(separator="\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text)
            return f"[셔틀버스 시간표]\n{text[:3000]}"
    except Exception as e:
        return f"셔틀버스 정보 조회 실패: {e}"

    return "셔틀버스 정보를 가져올 수 없습니다."


def get_academic_calendar(month: int | None = None) -> str:
    """학사일정을 반환한다.

    Args:
        month: 조회할 월 (1-12). None이면 전체.

    Returns:
        학사일정 텍스트
    """
    # 가장 최신 portal 파일 사용
    raw_dir = Path("data/corpus/raw")
    portal_files = sorted(raw_dir.glob("portal_*.jsonl"), reverse=True)
    if not portal_files:
        return "학사일정 데이터를 찾을 수 없습니다."
    portal_file = portal_files[0]

    with open(portal_file, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]

    # 첫 번째 레코드가 학사일정
    calendar_doc = None
    for doc in docs:
        if "학사일정" in doc["title"]:
            calendar_doc = doc
            break

    if not calendar_doc:
        return "학사일정 데이터를 찾을 수 없습니다."

    content = calendar_doc["content"]

    if month is not None:
        # 해당 월 섹션만 추출
        pattern = rf"## \d{{4}}년 {month}월\n(.*?)(?=## \d{{4}}년 \d+월|\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return f"{month}월 학사일정:\n{match.group(0).strip()}"
        return f"{month}월 학사일정을 찾을 수 없습니다."

    return content[:3000]


def get_notices(count: int = 5) -> str:
    """최신 공지사항을 크롤링하여 반환한다.

    Args:
        count: 가져올 공지 수 (최대 10)

    Returns:
        공지사항 텍스트
    """
    count = min(max(count, 1), 10)
    board_url = "https://computer.cnu.ac.kr/computer/notice/bachelor.do"

    try:
        resp = _SESSION.get(board_url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        article_links: list[str] = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "articleNo" in href:
                from urllib.parse import urljoin

                full_url = urljoin(board_url, href).split("#")[0]
                if full_url not in article_links:
                    article_links.append(full_url)

        results: list[str] = []
        for url in article_links[:count]:
            try:
                resp = _SESSION.get(url, timeout=30)
                resp.raise_for_status()
                art_soup = BeautifulSoup(resp.text, "html.parser")

                for tag in art_soup.find_all(["script", "style", "nav", "footer"]):
                    tag.decompose()

                title_tag = art_soup.find("h1") or art_soup.find("title")
                title = title_tag.get_text(strip=True) if title_tag else ""
                body = art_soup.find("div", class_=re.compile(r"content|view|body", re.I))
                text = (body or art_soup.find("body")).get_text(separator="\n", strip=True)
                text = re.sub(r"\n{3,}", "\n\n", text)

                results.append(f"[{title}]\n{text[:500]}\n출처: {url}")
            except Exception:
                pass
            time.sleep(CRAWL_DELAY)

        if results:
            return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"공지사항 조회 실패: {e}"

    return "공지사항을 가져올 수 없습니다."


# ── Tool 디스패처 ───────────────────────────────────────────────────────────

_TOOL_FUNCTIONS: dict[str, callable] = {
    "get_meal_menu": get_meal_menu,
    "get_shuttle_schedule": get_shuttle_schedule,
    "get_academic_calendar": get_academic_calendar,
    "get_notices": get_notices,
}


def execute_tool(name: str, arguments: dict) -> str:
    """이름으로 tool을 찾아 실행하고 결과 문자열을 반환한다.

    Args:
        name: tool 이름
        arguments: tool 인자 딕셔너리

    Returns:
        tool 실행 결과 텍스트
    """
    func = _TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"알 수 없는 도구: {name}"
    try:
        return func(**arguments)
    except Exception as e:
        return f"도구 실행 오류 ({name}): {e}"
