"""식단 스킬 — 기숙사 식당 + 학생회관 식단 라이브 크롤링.

- 기숙사: dorm.cnu.ac.kr (텍스트 기반, 날짜 헤더 파싱)
- 학생회관: mobileadmin.cnu.ac.kr (SSR 테이블), 제1학생회관은 코너별 고정 메뉴 상수
"""

from __future__ import annotations

import re
import time
from datetime import timedelta

from bs4 import BeautifulSoup

from src.skills.base import CRAWL_DELAY, SESSION, Skill, now_kst
from src.skills.registry import register


def _fetch_dorm_meal(date: str) -> str:
    """기숙사 식단을 크롤링한다 (dorm.cnu.ac.kr).

    페이지는 텍스트 기반. 날짜 헤더("7(토)") 후 조식/중식/석식 블록으로 구성.

    Args:
        date: YYYY-MM-DD 형식 날짜

    Returns:
        정리된 기숙사 식단 텍스트
    """
    try:
        resp = SESSION.get(
            "https://dorm.cnu.ac.kr/html/kr/sub03/sub03_0304.html",
            timeout=30,
        )
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        from datetime import datetime

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
        meals: dict[str, dict[str, list[str]]] = {"아침": {}, "점심": {}, "저녁": {}}
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
        print(f"  [skill] 기숙사 식단 오류: {e}")
        return "[기숙사] 식단 정보를 가져올 수 없어요."


# 제1학생회관 고정 메뉴 (코너별 상시 운영).
# mobileadmin 식단표엔 제1학생회관 일일 메뉴가 안 올라와(병합 빈 칸) 라이브 크롤링이 불가하므로,
# 라면·양식·한식·일식·중식 코너별 고정 메뉴를 상수로 둔다.
_BUILDING1_MENU = """■ 제1학생회관 (코너별 고정 메뉴)

라면&간식 (10:00~14:00)
- 라면 2,500 / 떡만두라면 3,000 / 해장라면 3,000 / 치즈라면 3,000 / 부대라면 4,000 / 김밥 3,000 / 공기밥 500

양식 (11:00~14:00)
- 등심왕돈까스 5,500 / 눈꽃치즈돈까스 6,000 / 파채돈까스 6,000 / 돈까스&파스타 6,500 / 닭다리살스테이크 6,000 / 파닭스테이크 6,500 / 베이컨로제파스타 5,800

스낵 (11:00~14:30)
- 버터김치별달알밥 5,600 / 버터김치치즈알밥 6,500 / 컵버터감자칩 2,000 (토핑: 치즈 1,000·소불고기 2,000·떡갈비 1,000·감자고로케 1,000·날치알 1,000·구운계란 500)

한식 (11:00~14:00 / 저녁 17:00~19:00)
- 비빔밥 7,000 / 묵은지김치찌개 5,800 / 부대햄플러스김치찌개 6,800 / 우삼겹된장찌개 6,600 / 별달소고기국밥 7,000 / 똑배기닭갈비덮밥 5,800

일식 (11:00~19:00)
- 얼큰소두부우동국밥 6,300 / 매운부타동고기덮밥 7,500 / 고소한카레덮밥 5,800 / 부타동고기덮밥 5,800 / 치킨가라아게마요 6,500 / 마제소바 6,000 / 1L치킨포케샐러드 6,800 / 세겹배기생우동 6,500

중식 (점심 11:00~14:00 / 저녁 16:00~19:00)
- 차돌온면 6,500 / 매운차돌온면 6,500 / 온국밥 6,500 / 매운온국밥 6,500 / 비빔면 5,800 / 냉면 5,800"""


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
    print(f"  [skill] 학생회관 식단 크롤링: {url}")

    try:
        resp = SESSION.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        table = soup.find("table", class_="menu-tbl")
        if not table:
            return f"[학생회관] {date} 식단 테이블을 찾을 수 없습니다."

        # 헤더: ["구분", "제1학생회관", "제2학생회관", ...]
        ths = table.find("thead").find_all("th", scope="col")
        hall_names = [th.get_text(strip=True) for th in ths]
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

        print(f"  [skill] 학생회관 식단: {len(parts)}행 추출")

        building1 = _BUILDING1_MENU
        b1_block = (
            f"{building1}\n\n"
            if building1
            else "※ 제1학생회관은 코너별 고정 메뉴로 운영됩니다.\n\n"
        )
        if parts:
            return f"[학생회관 식단 {date}]\n{b1_block}" + "\n".join(parts)
        elif building1:
            return f"[학생회관 식단 {date}]\n{b1_block}".rstrip()
        else:
            return f"[학생회관] {date} 식단 데이터 없음 (주말 또는 미운영)"
    except Exception as e:
        print(f"  [skill] 학생회관 조회 실패: {e}")
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
        date = now_kst().strftime("%Y-%m-%d")

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
            results.append(
                f"[학생회관] {date} 식단 정보를 찾을 수 없습니다. "
                "학생회관 식단은 https://mobileadmin.cnu.ac.kr/food/index.jsp 에서 직접 확인해주세요."
            )

    if not results:
        return f"{date} 식단 정보를 가져올 수 없습니다. 주말이거나 운영하지 않는 날일 수 있습니다."
    return "\n\n".join(results)


def _infer_meal_args(question: str) -> dict:
    """식단 질문에서 location과 date 인자를 추론한다."""
    args: dict = {}

    if "기숙사" in question:
        args["location"] = "dormitory"
    elif any(kw in question for kw in ["학생회관", "학생 회관", "학식", "학생식당"]):
        args["location"] = "student_hall"

    # 상대적 날짜 처리
    if "내일" in question:
        args["date"] = (now_kst() + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "모레" in question:
        args["date"] = (now_kst() + timedelta(days=2)).strftime("%Y-%m-%d")
    elif "어제" in question:
        args["date"] = (now_kst() - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        # 절대 날짜: "6월 5일" 형태
        date_match = re.search(r"(\d{1,2})월\s*(\d{1,2})일", question)
        if date_match:
            month, day = int(date_match.group(1)), int(date_match.group(2))
            args["date"] = f"{now_kst().year}-{month:02d}-{day:02d}"
        else:
            # "6/5", "6.5" 형태
            slash_match = re.search(r"(\d{1,2})[/.](\d{1,2})", question)
            if slash_match:
                month, day = int(slash_match.group(1)), int(slash_match.group(2))
                if 1 <= month <= 12 and 1 <= day <= 31:
                    args["date"] = f"{now_kst().year}-{month:02d}-{day:02d}"

    return args


register(
    Skill(
        name="get_meal_menu",
        description=(
            "충남대학교 교내 식당의 오늘 식단 메뉴를 조회합니다. "
            "기숙사 식당과 학생회관 식당을 구분하여 조회 가능합니다."
        ),
        keywords=[
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
        run=get_meal_menu,
        infer_args=_infer_meal_args,
    )
)
