"""키워드 기반 tool 감지 모듈.

사용자 질문에서 키워드를 매칭하여 실행할 tool과 인자를 결정한다.
모델 기반 tool calling 대신 사용 — 모델을 2번 돌릴 필요 없어 빠르고 확실하다.
"""

from __future__ import annotations

import re
from datetime import datetime

# 키워드 → tool 매핑 (순서 중요: 먼저 매칭되는 tool이 선택됨)
TOOL_KEYWORDS: dict[str, list[str]] = {
    "get_meal_menu": [
        "식단", "메뉴", "학식", "밥", "점심", "저녁", "아침",
        "조식", "중식", "석식", "기숙사 식당", "학생회관", "급식",
    ],
    "get_shuttle_schedule": [
        "셔틀", "버스", "통학", "노선", "시간표", "정류장",
    ],
    "get_academic_calendar": [
        "학사일정", "기말", "중간", "방학", "개강", "종강", "휴강",
    ],
    "get_notices": [
        "공지", "알림", "공지사항",
    ],
}


def _infer_meal_args(question: str) -> dict:
    """식단 질문에서 location과 date 인자를 추론한다."""
    args: dict = {}

    if "기숙사" in question:
        args["location"] = "dormitory"
    elif any(kw in question for kw in ["학생회관", "학생 회관", "학식", "학생식당"]):
        args["location"] = "student_hall"

    # 날짜 추출: "6월 5일", "6/5" 등
    date_match = re.search(r"(\d{1,2})월\s*(\d{1,2})일", question)
    if date_match:
        month, day = int(date_match.group(1)), int(date_match.group(2))
        year = datetime.now().year
        args["date"] = f"{year}-{month:02d}-{day:02d}"

    return args


def _infer_calendar_args(question: str) -> dict:
    """학사일정 질문에서 month, year 인자를 추론한다."""
    args: dict = {}
    year_match = re.search(r"(20\d{2})년", question)
    if year_match:
        args["year"] = int(year_match.group(1))
    elif "작년" in question or "지난해" in question:
        args["year"] = datetime.now().year - 1
    elif "내년" in question:
        args["year"] = datetime.now().year + 1
    month_match = re.search(r"(\d{1,2})월", question)
    if month_match:
        month = int(month_match.group(1))
        if 1 <= month <= 12:
            args["month"] = month
    return args


def _infer_notice_args(question: str) -> dict:
    """공지사항 질문에서 count 인자를 추론한다."""
    count_match = re.search(r"(\d+)\s*개", question)
    if count_match:
        count = min(max(int(count_match.group(1)), 1), 10)
        return {"count": count}
    return {}


_ARG_INFERRERS: dict[str, callable] = {
    "get_meal_menu": _infer_meal_args,
    "get_shuttle_schedule": lambda q: {},
    "get_academic_calendar": _infer_calendar_args,
    "get_notices": _infer_notice_args,
}


def detect_tool(question: str) -> tuple[str | None, dict]:
    """질문에서 키워드 매칭으로 tool을 감지한다.

    Args:
        question: 사용자 질문

    Returns:
        (tool_name, arguments) 튜플. tool이 불필요하면 (None, {}).
    """
    for tool_name, keywords in TOOL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in question:
                args = _ARG_INFERRERS[tool_name](question)
                return tool_name, args
    return None, {}
