"""학사일정 스킬 — plus.cnu.ac.kr 실시간 조회 + 2026 내장 폴백.

라이브 크롤링 실패 시 내장 학사일정(2026년) 으로 폴백한다.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from src.skills.base import SESSION, Skill, now_kst
from src.skills.registry import register

_BUILTIN_CALENDAR = {
    1: (
        "1월 학사일정:\n"
        "- 1/1(목): 신정\n"
        "- 1/13(화): 제2학기 성적발표\n"
        "- 1/20(화): 동기 계절학기 성적발표\n"
        "- 1/21(수)~2/10(화): 특별학기(신입학예정자)\n"
        "- 1/26(월)~1/28(수): 제1학기 예비수강신청\n"
        "- 1/29(목)~2/4(수): 학사학위취득 유예 신청"
    ),
    2: (
        "2월 학사일정:\n"
        "- 2/2(월)~2/27(금): 휴학 및 복학 신청\n"
        "- 2/2(월)~2/6(금): 제1학기 수강신청\n"
        "- 2/17(화): 설날\n"
        "- 2/24(화)~2/27(금): 제1학기 재학생 등록금 납부\n"
        "- 2/25(수): 전기 학위수여식\n"
        "- 2/27(금): 입학식"
    ),
    3: (
        "3월 학사일정:\n"
        "- 3/1(일): 3·1절\n"
        "- 3/2(월): 대체공휴일\n"
        "- 3/3(화): 제1학기 개강일\n"
        "- 3/3(화)~3/9(월): 수강신청 확인 및 변경\n"
        "- 3/16(월): 폐강과목 결정\n"
        "- 3/16(월)~3/20(금): 융복합창의전공 신청·취소\n"
        "- 3/23(월)~3/26(목): 수강신청 취소\n"
        "- 3/27(금): 수업일수 1/4선\n"
        "- 3/30(월)~4/3(금): 후기 조기졸업 신청"
    ),
    4: ("4월 학사일정:\n- 4/6(월): 수업일수 1/3선\n- 4/23(목): 수업일수 1/2선"),
    5: (
        "5월 학사일정:\n"
        "- 5/1(금): 노동절\n"
        "- 5/5(화): 어린이날\n"
        "- 5/7(목)~5/11(월): 하기 계절학기 수강신청\n"
        "- 5/13(수): 수업일수 2/3선\n"
        "- 5/22(금): 수업일수 3/4선\n"
        "- 5/24(일): 부처님오신날\n"
        "- 5/25(월): 개교기념일, 대체공휴일"
    ),
    6: (
        "6월 학사일정:\n"
        "- 6/3(수): 지방선거일\n"
        "- 6/6(토): 현충일\n"
        "- 6/9(화)~6/12(금): 보충강의 기간\n"
        "- 6/22(월): 여름방학(하기방학) 시작 (8/31까지)\n"
        "- 6/22(월)~7/10(금): 하기 계절학기 (방학 중 선택 수업, 방학과 별개)"
    ),
    7: (
        "7월 학사일정:\n"
        "- 여름방학 계속 (6/22~8/31)\n"
        "- 7/10(금): 제1학기 성적발표\n"
        "- 7/20(월): 하기 계절학기 성적발표\n"
        "- 7/27(월)~7/29(수): 제2학기 예비수강신청"
    ),
    8: (
        "8월 학사일정:\n"
        "- 8/3(월)~8/7(금): 제2학기 수강신청\n"
        "- 8/3(월)~8/31(월): 휴학 및 복학 신청\n"
        "- 8/15(토): 광복절\n"
        "- 8/17(월): 대체공휴일\n"
        "- 8/25(화): 후기 학위수여식\n"
        "- 8/25(화)~8/28(금): 2학기 재학생 등록금 납부"
    ),
    9: (
        "9월 학사일정:\n"
        "- 9/1(화): 제2학기 개강일\n"
        "- 9/1(화)~9/7(월): 수강신청 확인 및 변경\n"
        "- 9/14(월): 폐강교과목 결정\n"
        "- 9/14(월)~9/18(금): 융복합창의전공 신청·취소\n"
        "- 9/21(월)~9/28(월): 수강신청 취소\n"
        "- 9/25(금): 추석\n"
        "- 9/29(화): 수업일수 1/4선\n"
        "- 9/30(수)~10/7(수): 전기 조기졸업 신청"
    ),
    10: (
        "10월 학사일정:\n"
        "- 10/3(토): 개천절\n"
        "- 10/5(월): 대체공휴일\n"
        "- 10/8(목): 수업일수 1/3선\n"
        "- 10/9(금): 한글날\n"
        "- 10/28(수): 수업일수 1/2선"
    ),
    11: (
        "11월 학사일정:\n"
        "- 11/5(목)~11/10(화): 동기 계절학기 수강신청\n"
        "- 11/13(금): 수업일수 2/3선\n"
        "- 11/24(화): 수업일수 3/4선"
    ),
    12: (
        "12월 학사일정:\n"
        "- 12/8(화)~12/11(금): 보충강의 기간\n"
        "- 12/21(월): 겨울방학(동기방학) 시작 (2월말까지)\n"
        "- 12/21(월)~1/12(화): 동기 계절학기 (방학 중 선택 수업, 방학과 별개)\n"
        "- 12/25(금): 기독탄신일"
    ),
}


def _get_builtin_calendar(month: int | None = None) -> str:
    """내장 학사일정 데이터를 반환한다."""
    if month is not None:
        return _BUILTIN_CALENDAR.get(month, f"{month}월 학사일정 정보가 없습니다.")
    now_month = now_kst().month
    result = []
    for offset in range(-1, 3):
        m = ((now_month - 1 + offset) % 12) + 1
        if m in _BUILTIN_CALENDAR:
            result.append(_BUILTIN_CALENDAR[m])
    return "\n\n".join(result) if result else "학사일정 정보를 찾을 수 없습니다."


def _fetch_calendar_from_web(year: int, month: int | None = None) -> str | None:
    """plus.cnu.ac.kr에서 학사일정을 실시간으로 가져온다."""
    url = (
        f"https://plus.cnu.ac.kr/_prog/academic_calendar/"
        f"?site_dvs_cd=kr&menu_dvs_cd=05020101&year={year}"
    )
    try:
        resp = SESSION.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")

        results: list[str] = []
        current_month = None
        month_events: list[str] = []

        for row in soup.find_all(["tr", "li", "dl"]):
            text = row.get_text(separator=" ", strip=True)
            month_header = re.search(r"^(\d{1,2})월$", text)
            if month_header:
                if current_month and month_events:
                    if month is None or current_month == month:
                        results.append(f"{current_month}월 학사일정:\n" + "\n".join(month_events))
                current_month = int(month_header.group(1))
                month_events = []
                continue
            if current_month and re.search(r"\d{2}\.\d{2}", text):
                month_events.append(f"- {text}")

        if current_month and month_events:
            if month is None or current_month == month:
                results.append(f"{current_month}월 학사일정:\n" + "\n".join(month_events))

        if results:
            header = f"{year}학년도 학사일정\n\n"
            return header + "\n\n".join(results)
    except Exception:
        pass
    return None


def get_academic_calendar(month: int | None = None, year: int | None = None) -> str:
    """학사일정을 반환한다.

    Args:
        month: 조회할 월 (1-12). None이면 현재 월 기준 앞뒤.
        year: 조회할 연도. None이면 현재 연도.

    Returns:
        학사일정 텍스트
    """
    if year is None:
        year = now_kst().year

    # 1순위: 실시간 크롤링
    web_result = _fetch_calendar_from_web(year, month)
    if web_result:
        return web_result

    # 2순위: 내장 데이터 (2026년만)
    if year == 2026:
        return _get_builtin_calendar(month)

    return f"{year}년 학사일정을 가져올 수 없습니다. 충남대 홈페이지를 확인해주세요."


def _infer_calendar_args(question: str) -> dict:
    """학사일정 질문에서 month, year 인자를 추론한다."""
    args: dict = {}
    year_match = re.search(r"(20\d{2})년", question)
    if year_match:
        args["year"] = int(year_match.group(1))
    elif "작년" in question or "지난해" in question:
        args["year"] = now_kst().year - 1
    elif "내년" in question:
        args["year"] = now_kst().year + 1
    month_match = re.search(r"(\d{1,2})월", question)
    if month_match:
        month = int(month_match.group(1))
        if 1 <= month <= 12:
            args["month"] = month
    return args


register(
    Skill(
        name="get_academic_calendar",
        description=(
            "충남대학교 학사일정을 조회합니다. "
            "수강신청, 개강, 중간고사, 기말고사, 방학 등의 일정을 확인할 수 있습니다."
        ),
        keywords=[
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
        negative_keywords=["기말 레포트", "기말 과제", "중간 레포트", "중간 과제", "중간 정도"],
        parameters={
            "month": {"type": "integer", "description": "조회할 월(1-12). 비우면 이번 달 기준."},
            "year": {"type": "integer", "description": "조회할 연도. 비우면 올해."},
        },
        run=get_academic_calendar,
        infer_args=_infer_calendar_args,
    )
)
