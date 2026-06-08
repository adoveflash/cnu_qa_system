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
            "description": "충남대학교 교내 식당의 오늘 식단 메뉴를 조회합니다. 기숙사 식당과 학생회관 식당을 구분하여 조회 가능합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "조회할 날짜 (YYYY-MM-DD). 생략 시 오늘 날짜.",
                    },
                    "location": {
                        "type": "string",
                        "enum": ["all", "dormitory", "student_hall"],
                        "description": "조회할 식당. dormitory=기숙사 식당, student_hall=학생회관(1~4학생회관), all=전체. 생략 시 all.",
                    },
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

    페이지는 텍스트 기반. 날짜 헤더("7(토)") 후 조식/중식/석식 블록으로 구성.

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
            if target_label in line and re.match(rf"^{day_num}\({weekday_kr}\)", line):
                capture = True
                continue
            if capture and re.match(r"^\d{1,2}\([월화수목금토일]\)", line):
                break
            if capture:
                day_lines.append(line)

        if not day_lines:
            return ""

        def _clean_menu_item(item: str) -> str:
            """메뉴 항목에서 알레르기/원산지/영문 제거."""
            item = re.sub(r"\s+[\d,]+$", "", item)  # 알레르기 번호
            item = re.sub(r"\[.+?\]", "", item)  # 원산지
            item = item.strip("* ")
            return item.strip()

        # 한국어 메뉴 블록만 추출 (영어 번역 블록 제거)
        # 구조: 메인A(kcal) + 한국어 → 메인A(kcal) + 영어(중복, 스킵)
        #        메인C(kcal) + 한국어 → 메인C(kcal) + 영어(중복, 스킵)
        # 각 끼니(아침/점심/저녁)에 A, C 두 가지 타입이 있음
        meal_blocks: list[tuple[str, list[str]]] = []  # (타입명, 메뉴라인들)
        current_block: list[str] = []
        current_type: str = ""
        seen_headers: set[str] = set()  # "메인A(822kcal)" 전체를 키로
        skip_mode = False

        for line in day_lines:
            m = re.match(r"^(메인[A-Z]?)\s*\((\d+)\s*kcal\)", line)
            if m:
                header_key = f"{m.group(1)}({m.group(2)})"
                if header_key in seen_headers:
                    # 같은 헤더 반복 → 영어 번역 블록, 스킵
                    if current_block:
                        meal_blocks.append((current_type, current_block))
                        current_block = []
                    skip_mode = True
                    continue
                else:
                    if current_block:
                        meal_blocks.append((current_type, current_block))
                    current_block = []
                    current_type = m.group(1)  # "메인A" or "메인C"
                    seen_headers.add(header_key)
                    skip_mode = False
                    continue
            if not skip_mode:
                current_block.append(line)

        if current_block:
            meal_blocks.append((current_type, current_block))

        # 블록→끼니 배정: 매 2블록이 1끼니 (A타입 + C타입)
        # 블록 순서: 아침A, 아침C, 점심A, 점심C, 저녁A, 저녁C
        meals: dict[str, dict[str, list[str]]] = {
            "아침": {}, "점심": {}, "저녁": {}
        }
        meal_order = ["아침", "점심", "저녁"]

        for idx, (type_name, block) in enumerate(meal_blocks):
            meal_idx = idx // 2  # 0,1→아침, 2,3→점심, 4,5→저녁
            if meal_idx >= len(meal_order):
                break
            label = meal_order[meal_idx]
            menu_items: list[str] = []
            for line in block:
                if re.match(r"^[A-Za-z\s,&.()*]+$", line):
                    continue
                if re.match(r"^\[.+:.+\]$", line):
                    continue
                if "우유(공통)" in line or "우유 (공통)" in line:
                    menu_items.append("우유")
                    continue
                cleaned = _clean_menu_item(line)
                if cleaned and len(cleaned) > 1:
                    menu_items.append(cleaned)
            if menu_items:
                meals[label][type_name] = menu_items

        # 결과 구성
        result_parts = [f"[기숙사 식단 {date} ({weekday_kr})]"]
        for label in ["아침", "점심", "저녁"]:
            if meals[label]:
                type_strs = []
                for type_name, items in meals[label].items():
                    type_strs.append(f"{type_name}: {', '.join(items)}")
                result_parts.append(f"\n■ {label}\n  " + "\n  ".join(type_strs))
            else:
                result_parts.append(f"\n■ {label}: 운영 안 함")

        if len(result_parts) > 1:
            return "\n".join(result_parts)
        return ""
    except Exception as e:
        print(f"  [tool] 기숙사 식단 오류: {e}")
        return f"[기숙사] 식단 정보를 가져올 수 없어요."


def _fetch_student_hall_meal(date: str) -> str:
    """학생회관 식단을 requests로 크롤링한다 (mobileadmin.cnu.ac.kr).

    서버사이드 렌더링이므로 requests + BeautifulSoup만으로 추출 가능.

    Args:
        date: YYYY-MM-DD 형식 날짜

    Returns:
        정리된 학생회관 식단 텍스트
    """
    date_dot = date.replace("-", ".")
    url = f"https://mobileadmin.cnu.ac.kr/food/index.jsp?searchYmd={date_dot}"
    print(f"  [tool] 학생회관 식단 크롤링: {url}")

    try:
        resp = _SESSION.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        table = soup.find("table", class_="menu-tbl")
        if not table:
            return f"[학생회관] {date} 식단 테이블을 찾을 수 없습니다."

        # 헤더: ["구분", "제1학생회관", "제2학생회관", ...]
        ths = table.find("thead").find_all("th", scope="col")
        hall_names = [th.get_text(strip=True) for th in ths]
        # "구분" 제거 → 식당 이름만
        hall_names = [h for h in hall_names if h and h != "구분"]

        meal_map = {"조식": "아침", "중식": "점심", "석식": "저녁"}
        parts: list[str] = []
        current_meal = ""
        current_type = ""  # 직원/학생

        for tr in table.find("tbody").find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue

            building_td = tr.find("td", class_="building")
            if building_td:
                raw_meal = building_td.get_text(strip=True)
                current_meal = meal_map.get(raw_meal, raw_meal)

            # 메뉴 td만 추출 (building, 직원/학생, rowspan td 제외)
            menu_tds = []
            for td in tds:
                if "building" in td.get("class", []):
                    continue
                text = td.get_text(strip=True)
                if text in ("직원", "학생"):
                    current_type = text
                    continue
                if td.get("rowspan", "") == "100":
                    continue
                menu_tds.append(td)

            # menu_tds를 식당에 매핑 (제1학생회관은 rowspan으로 빠짐)
            # 직원행: 2~4학+생활과학 = 4개, 학생행: 2~4학+생활과학 = 4개
            offset = 1  # 제1학생회관은 rowspan=100으로 스킵
            for i, td in enumerate(menu_tds):
                hall_idx = offset + i
                if hall_idx >= len(hall_names):
                    break
                hall_name = hall_names[hall_idx]

                text = td.get_text(strip=True)
                if not text or "운영안함" in text or "메뉴운영" in text:
                    continue

                menu_items = []
                for li in td.find_all("li"):
                    title_tag = li.find("h3")
                    title = title_tag.get_text(strip=True) if title_tag else ""
                    p_tag = li.find("p")
                    if p_tag:
                        items = [s.strip() for s in p_tag.stripped_strings if s.strip()]
                        if title:
                            menu_items.append(f"  {title}: {', '.join(items)}")
                        else:
                            menu_items.append(f"  {', '.join(items)}")

                if menu_items:
                    label = f"■ {current_meal} ({hall_name}, {current_type})"
                    parts.append(label)
                    parts.extend(menu_items)

        print(f"  [tool] 학생회관 식단: {len(parts)}행 추출")

        print(f"  [tool] 학생회관 식단: {len(parts)}행 추출")

        if parts:
            note = "※ 제1학생회관 메뉴는 별도 시스템으로 운영되어 조회할 수 없습니다."
            return f"[학생회관 식단 {date}]\n{note}\n" + "\n".join(parts)
        else:
            return f"[학생회관] {date} 식단 데이터 없음 (주말 또는 미운영)"
    except Exception as e:
        print(f"  [tool] 학생회관 조회 실패: {e}")
        return f"[학생회관] 조회 실패: {e}"


def get_meal_menu(date: str | None = None, location: str = "all") -> str:
    """교내 식당 식단을 크롤링하여 반환한다.

    Args:
        date: 조회 날짜 (YYYY-MM-DD). None이면 오늘.
        location: 조회 식당 (all/dormitory/student_hall). 기본 all.

    Returns:
        식단 정보 텍스트
    """
    if date is None:
        from datetime import timezone, timedelta
        KST = timezone(timedelta(hours=9))
        date = datetime.now(KST).strftime("%Y-%m-%d")

    results: list[str] = []

    # 기숙사 식단
    if location in ("all", "dormitory"):
        dorm = _fetch_dorm_meal(date)
        if dorm:
            results.append(dorm)
        elif location == "dormitory":
            results.append(f"[기숙사] {date} 식단 정보를 찾을 수 없습니다.")

    if location == "all":
        time.sleep(CRAWL_DELAY)

    # 학생회관 식단
    if location in ("all", "student_hall"):
        hall = _fetch_student_hall_meal(date)
        if hall:
            results.append(hall)
        elif location == "student_hall":
            results.append(f"[학생회관] {date} 식단 정보를 찾을 수 없습니다. 학생회관 식단은 https://mobileadmin.cnu.ac.kr/food/index.jsp 에서 직접 확인해주세요.")

    if not results:
        return f"{date} 식단 정보를 가져올 수 없습니다. 주말이거나 운영하지 않는 날일 수 있습니다."
    return "\n\n".join(results)


def get_shuttle_schedule() -> str:
    """셔틀버스 시간표를 반환한다.

    Returns:
        셔틀버스 시간표 텍스트
    """
    schedule = (
        "충남대학교 셔틀버스 운행 안내\n"
        "운영기준: 학기 중 평일 주간 운영 (야간/주말/공휴일/방학 미운영)\n\n"
        "■ 교내 순환 (유성캠퍼스 내)\n"
        "- 첫차 08:30, 막차 17:30 (1시간 간격, 하루 10회)\n"
        "- 08:20 월평역 출발 등교편 1회 운행\n"
        "- 운행시간: 08:30, 09:30, 10:30, 11:30, 13:30, 14:30, 15:30, 16:30, 17:30\n"
        "- 노선: 정심화 국제문화회관 → 사회과학대학 입구 → 서문 → 음악2호관 → "
        "공동동물실험센터(회차) → 체육관 입구 → 예술대학 → 도서관 → "
        "학생생활관 3거리 → 농업생명과학대학 → 동문주차장 → 도서관 → "
        "예술대학 → 서문 → 사회과학대학 입구 → 산학연교육연구관 → 정심화 국제문화회관\n\n"
        "■ 캠퍼스 순환 (대덕캠퍼스 ↔ 보운캠퍼스)\n"
        "- 하루 1회(회차), 오전만 운행\n"
        "- 08:10 골프연습장 출발 → 중앙도서관 → 산학연교육연구관 → "
        "충남대입구 버스정류장 → 월평역 → 보운캠퍼스(08:50 회차) → 복귀\n\n"
        "※ 교통 상황에 따라 5분 내외 오차 발생 가능\n"
        "※ 문의: 총무과(배차/운행) ☏042-821-5115"
    )
    return schedule


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
    4: (
        "4월 학사일정:\n"
        "- 4/6(월): 수업일수 1/3선\n"
        "- 4/23(목): 수업일수 1/2선"
    ),
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
    # 현재 월 기준 앞뒤 2개월 반환
    from datetime import datetime
    now_month = datetime.now().month
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
        resp = _SESSION.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")

        # 월별 일정 추출
        results: list[str] = []
        current_month = None
        month_events: list[str] = []

        # 테이블 또는 리스트에서 일정 추출
        for row in soup.find_all(["tr", "li", "dl"]):
            text = row.get_text(separator=" ", strip=True)
            # "1월", "2월" 등 월 헤더 감지
            month_header = re.search(r"^(\d{1,2})월$", text)
            if month_header:
                if current_month and month_events:
                    if month is None or current_month == month:
                        results.append(
                            f"{current_month}월 학사일정:\n" + "\n".join(month_events)
                        )
                current_month = int(month_header.group(1))
                month_events = []
                continue
            # 일정 항목 (날짜 패턴 포함)
            if current_month and re.search(r"\d{2}\.\d{2}", text):
                month_events.append(f"- {text}")

        # 마지막 월 처리
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
        year = datetime.now().year

    # 1순위: 실시간 크롤링
    web_result = _fetch_calendar_from_web(year, month)
    if web_result:
        return web_result

    # 2순위: 내장 데이터 (2026년만)
    if year == 2026:
        return _get_builtin_calendar(month)

    return f"{year}년 학사일정을 가져올 수 없습니다. 충남대 홈페이지를 확인해주세요."


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

        from urllib.parse import urljoin

        # 게시글 링크 + 날짜 추출
        article_items: list[tuple[str, str]] = []  # (url, date_str)
        rows = soup.find_all("tr")
        for row in rows:
            a_tag = row.find("a", href=True)
            if not a_tag or "articleNo" not in a_tag["href"]:
                continue
            full_url = urljoin(board_url, a_tag["href"]).split("#")[0]
            # 날짜 추출: td 중 날짜 패턴 찾기 (YY.MM.DD 또는 YYYY.MM.DD)
            date_str = "0000.00.00"
            for td in row.find_all("td"):
                td_text = td.get_text(strip=True)
                m = re.match(r"(\d{2,4})\.(\d{2})\.(\d{2})", td_text)
                if m:
                    year = m.group(1)
                    if len(year) == 2:
                        year = "20" + year
                    date_str = f"{year}.{m.group(2)}.{m.group(3)}"
                    break
            if full_url not in [x[0] for x in article_items]:
                article_items.append((full_url, date_str))

        # 날짜 내림차순 정렬 (최신 우선)
        article_items.sort(key=lambda x: x[1], reverse=True)
        article_links = [url for url, _ in article_items]

        # 목록에서 제목+날짜 빠르게 추출 (상세 페이지 접근 최소화)
        results: list[str] = []
        for i, (url, date_str) in enumerate(article_items[:count]):
            rank = i + 1
            # 목록 페이지의 a 태그에서 제목 추출
            for row in rows:
                a_tag = row.find("a", href=True)
                if a_tag and url.endswith(a_tag["href"].split("#")[0].split("?")[-1]):
                    title = a_tag.get_text(strip=True)
                    results.append(f"[{rank}번째 최신 공지] ({date_str}) {title}\n출처: {url}")
                    break

        # 첫 번째 공지만 상세 내용 가져오기 (응답 속도 위해)
        if article_items:
            try:
                first_url = article_items[0][0]
                resp = _SESSION.get(first_url, timeout=30)
                resp.raise_for_status()
                art_soup = BeautifulSoup(resp.text, "html.parser")
                for tag in art_soup.find_all(["script", "style", "nav", "footer"]):
                    tag.decompose()
                body = art_soup.find("div", class_=re.compile(r"content|view|body", re.I))
                text = (body or art_soup.find("body")).get_text(separator="\n", strip=True)
                text = re.sub(r"\n{3,}", "\n\n", text)
                if results:
                    results[0] += f"\n\n[상세 내용]\n{text[:500]}"
            except Exception:
                pass

        if results:
            return f"최신순 정렬 (1번이 가장 최근):\n\n" + "\n\n---\n\n".join(results)
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
