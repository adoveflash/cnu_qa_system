"""키워드 기반 tool 감지 모듈.

사용자 질문에서 키워드를 매칭하여 실행할 tool과 인자를 결정한다.
모델 기반 tool calling 대신 사용 — 모델을 2번 돌릴 필요 없어 빠르고 확실하다.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def _now() -> datetime:
    return datetime.now(KST)


# 키워드 → tool 매핑
TOOL_KEYWORDS: dict[str, list[str]] = {
    "get_meal_menu": [
        "식단",
        "메뉴",
        "학식",
        "학생식당",
        "기숙사 식당",
        "학생회관",
        "조식",
        "중식",
        "석식",
        "급식",
        "점심 메뉴",
        "저녁 메뉴",
        "아침 메뉴",
        "오늘 밥",
        "기숙사 밥",
        "기숙사 식사",
        "기숙사 메뉴",
        "기숙사 점심",
        "기숙사 저녁",
        "기숙사 아침",
    ],
    "get_shuttle_schedule": [
        "셔틀",
        "셔틀버스",
        "통학버스",
        "통학",
    ],
    "get_academic_calendar": [
        "학사일정",
        "방학",
        "개강",
        "종강",
        "휴강",
        "기말고사",
        "중간고사",
        "계절학기",
        "수강신청 기간",
        "수강정정",
        "성적발표",
    ],
    "get_notices": [
        "공지",
        "알림",
        "공지사항",
    ],
}

# 특정 tool로 잘못 매칭되는 것을 방지하는 부정 키워드
NEGATIVE_KEYWORDS: dict[str, list[str]] = {
    "get_shuttle_schedule": ["시내버스", "시내 버스", "수업 시간표", "수강 시간표", "강의 시간표"],
    "get_academic_calendar": ["기말 레포트", "기말 과제", "중간 레포트", "중간 과제", "중간 정도"],
}


def _infer_meal_args(question: str) -> dict:
    """식단 질문에서 location과 date 인자를 추론한다."""
    args: dict = {}

    if "기숙사" in question:
        args["location"] = "dormitory"
    elif any(kw in question for kw in ["학생회관", "학생 회관", "학식", "학생식당"]):
        args["location"] = "student_hall"

    # 상대적 날짜 처리
    if "내일" in question:
        target = _now() + timedelta(days=1)
        args["date"] = target.strftime("%Y-%m-%d")
    elif "모레" in question:
        target = _now() + timedelta(days=2)
        args["date"] = target.strftime("%Y-%m-%d")
    elif "어제" in question:
        target = _now() - timedelta(days=1)
        args["date"] = target.strftime("%Y-%m-%d")
    else:
        # 절대 날짜: "6월 5일" 형태
        date_match = re.search(r"(\d{1,2})월\s*(\d{1,2})일", question)
        if date_match:
            month, day = int(date_match.group(1)), int(date_match.group(2))
            year = _now().year
            args["date"] = f"{year}-{month:02d}-{day:02d}"
        else:
            # "6/5", "6.5" 형태
            slash_match = re.search(r"(\d{1,2})[/.](\d{1,2})", question)
            if slash_match:
                month, day = int(slash_match.group(1)), int(slash_match.group(2))
                if 1 <= month <= 12 and 1 <= day <= 31:
                    year = _now().year
                    args["date"] = f"{year}-{month:02d}-{day:02d}"

    return args


def _infer_calendar_args(question: str) -> dict:
    """학사일정 질문에서 month, year 인자를 추론한다."""
    args: dict = {}
    year_match = re.search(r"(20\d{2})년", question)
    if year_match:
        args["year"] = int(year_match.group(1))
    elif "작년" in question or "지난해" in question:
        args["year"] = _now().year - 1
    elif "내년" in question:
        args["year"] = _now().year + 1
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
    matched: list[tuple[str, dict, int]] = []  # (tool_name, args, keyword_len)

    for tool_name, keywords in TOOL_KEYWORDS.items():
        # 부정 키워드 체크
        neg_keywords = NEGATIVE_KEYWORDS.get(tool_name, [])
        if any(neg in question for neg in neg_keywords):
            continue

        for keyword in keywords:
            if keyword in question:
                args = _ARG_INFERRERS[tool_name](question)
                matched.append((tool_name, args, len(keyword)))
                break

    if not matched:
        return None, {}

    # 가장 긴 키워드가 매칭된 tool 우선 (더 구체적인 매칭)
    matched.sort(key=lambda x: x[2], reverse=True)
    return matched[0][0], matched[0][1]
