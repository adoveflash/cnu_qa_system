"""식단 정보 크롤러.

수집 대상:
1. mobileadmin.cnu.ac.kr/food/index.jsp — 주간 식단 메뉴 (POST 요청)
2. plus.cnu.ac.kr/html/kr/ — 식당 위치/운영시간
3. www.cnucoop.co.kr — 생협 식당 안내 (기숙사 식당 포함, 테이블 구조 보존)
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .robots_checker import USER_AGENT

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})
SESSION.verify = False

CRAWL_DELAY = 3.0
OUTPUT_DIR = Path("data/corpus/raw")

# mobileadmin 식당 목록
_CAFETERIAS = ["제1학생회관", "제2학생회관", "제3학생회관", "제4학생회관", "생활과학대학"]


def _clean_text(soup: BeautifulSoup) -> str:
    """HTML에서 불필요한 태그를 제거하고 본문 텍스트를 반환한다."""
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("div", {"id": "content"})
        or soup.find("div", class_=re.compile(r"content|main|body", re.I))
        or soup.find("body")
    )
    if main is None:
        return ""

    text = main.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_table_as_text(table: BeautifulSoup) -> str:
    """HTML 테이블을 읽기 좋은 텍스트로 변환한다."""
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["td", "th"])]
        if cells and any(c for c in cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _extract_title(soup: BeautifulSoup) -> str:
    """페이지 제목을 추출한다."""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()
    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text(strip=True)
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return ""


def _save_record(
    fout,
    url: str,
    title: str,
    content: str,
    source: str,
    category: str = "식단",
) -> bool:
    """레코드를 JSONL 파일에 저장한다. 50자 미만이면 건너뛴다."""
    if len(content) < 50:
        return False
    record = {
        "url": url,
        "title": title,
        "content": content,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "category": category,
    }
    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True


def crawl_mobileadmin_meal(fout) -> int:
    """mobileadmin.cnu.ac.kr에서 식당별 주간 식단을 수집한다.

    JS 렌더링 불필요 — POST 요청으로 테이블 데이터를 직접 가져온다.
    """
    base_url = "https://mobileadmin.cnu.ac.kr/food/index.jsp"
    print("  [mobileadmin] 주간 식단 (POST 요청)")

    count = 0
    today = datetime.now()
    # 이번 주 월요일부터 시작하는 날짜 계산
    monday = today - timedelta(days=today.weekday())
    search_date = monday.strftime("%Y.%m.%d")

    for cafeteria in _CAFETERIAS:
        print(f"    {cafeteria}...", end=" ", flush=True)
        try:
            resp = SESSION.post(
                base_url,
                data={
                    "searchView": "day",
                    "searchYmd": search_date,
                    "language": "kor",
                    "searchCafeteria": cafeteria,
                },
                timeout=30,
            )
            resp.encoding = "utf-8"
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}")
                time.sleep(CRAWL_DELAY)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            menu_div = soup.find("div", class_="menu-wr")
            if not menu_div:
                print("메뉴 없음")
                time.sleep(CRAWL_DELAY)
                continue

            table = menu_div.find("table")
            if not table:
                print("테이블 없음")
                time.sleep(CRAWL_DELAY)
                continue

            # 테이블을 구조화된 텍스트로 변환
            menu_text = _format_weekly_menu(table, cafeteria)

            # 실질 메뉴가 없는 경우 (직원만 있거나 운영안함만 있는 경우) 건너뛰기
            has_real_menu = any(
                kw in menu_text
                for kw in ["정식", "백반", "밥", "국", "찌개", "볶음", "탕", "덮밥"]
            )
            if not menu_text or not has_real_menu:
                print("운영안함 (빈 식단)")
                time.sleep(CRAWL_DELAY)
                continue

            week_str = monday.strftime("%m/%d") + "~" + (monday + timedelta(days=5)).strftime("%m/%d")
            title = f"{cafeteria} 주간식단 ({week_str})"

            if _save_record(fout, base_url, title, menu_text, "mobileadmin", "식단"):
                count += 1
                print(f"OK ({len(menu_text)}자)")
            else:
                print("건너뜀")

        except Exception as e:
            print(f"오류: {e}")
        time.sleep(CRAWL_DELAY)

    # 기본 요청 (식당 선택 없이) — 전체 요약
    try:
        resp = SESSION.post(
            base_url,
            data={"searchView": "day", "searchYmd": search_date, "language": "kor"},
            timeout=30,
        )
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        menu_div = soup.find("div", class_="menu-wr")
        if menu_div:
            table = menu_div.find("table")
            if table:
                menu_text = _format_weekly_menu(table, "전체")
                has_real = any(
                    kw in menu_text
                    for kw in ["정식", "백반", "밥", "국", "찌개", "볶음"]
                )
                if menu_text and has_real:
                    week_str = (
                        monday.strftime("%m/%d")
                        + "~"
                        + (monday + timedelta(days=5)).strftime("%m/%d")
                    )
                    title = f"충남대 전체 주간식단 ({week_str})"
                    if _save_record(fout, base_url, title, menu_text, "mobileadmin", "식단"):
                        count += 1
                        print(f"    전체 식단 OK ({len(menu_text)}자)")
    except Exception as e:
        print(f"    전체 식단 오류: {e}")
    time.sleep(CRAWL_DELAY)

    return count


def _format_weekly_menu(table: BeautifulSoup, cafeteria_name: str) -> str:
    """식단 테이블을 사람이 읽기 좋은 형식으로 변환한다."""
    rows = table.find_all("tr")
    if not rows:
        return ""

    # 헤더 행 (요일)
    header_cells = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]

    lines = [f"## {cafeteria_name} 주간 식단", ""]

    for row in rows[1:]:
        cells = [td.get_text(separator=" ", strip=True) for td in row.find_all(["td", "th"])]
        if not cells or not any(c for c in cells):
            continue

        meal_type = cells[0] if cells else ""
        if meal_type in ("조식", "중식", "석식"):
            lines.append(f"### {meal_type}")

        for i, cell_text in enumerate(cells[1:], 1):
            if cell_text and cell_text != "운영안함":
                day_label = header_cells[i] if i < len(header_cells) else f"Day{i}"
                lines.append(f"- {day_label}: {cell_text}")

    return "\n".join(lines)


def crawl_plus_meal_pages(fout) -> int:
    """plus.cnu.ac.kr 내 식당/식단 관련 페이지를 수집한다."""
    urls = [
        ("https://plus.cnu.ac.kr/html/kr/sub05/sub05_05050101.html", "교내 편의시설(식당) 안내"),
        ("https://plus.cnu.ac.kr/html/kr/sub05/sub05_050404.html", "기타 서비스안내(금주의식단)"),
    ]

    count = 0
    for url, desc in urls:
        print(f"  [plus 식단] {desc}: {url}")
        try:
            resp = SESSION.get(url, timeout=30)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
            soup = BeautifulSoup(resp.text, "html.parser")
            title = _extract_title(soup) or desc
            content = _clean_text(soup)
            if _save_record(fout, url, title, content, "plus_kr", "식단"):
                count += 1
                print(f"    OK — {len(content)}자")
            else:
                print("    건너뜀 — 내용 부족")
        except Exception as e:
            print(f"    오류: {e}")
        time.sleep(CRAWL_DELAY)

    return count


def crawl_cnucoop(fout) -> int:
    """cnucoop.co.kr 생활협동조합 식당 안내 페이지를 수집한다 (테이블 구조 보존)."""
    urls = [
        ("https://www.cnucoop.co.kr/ezhtml2.php?html=canteen", "생협 식당안내 (전체)"),
        ("https://www.cnucoop.co.kr/ezhtml2.php?html=canteen2", "편의점이용안내"),
        ("https://www.cnucoop.co.kr/ezhtml2.php?html=canteen3", "매점안내"),
    ]

    count = 0
    for url, desc in urls:
        print(f"  [cnucoop] {desc}: {url}")
        try:
            resp = SESSION.get(url, timeout=30)
            resp.raise_for_status()
            if "euc-kr" in resp.headers.get("Content-Type", "").lower():
                resp.encoding = "euc-kr"
            elif resp.apparent_encoding:
                resp.encoding = resp.apparent_encoding
            soup = BeautifulSoup(resp.text, "html.parser")
            title = _extract_title(soup) or desc

            # 식당 정보 테이블 추출 (테이블 구조 보존)
            content = _extract_cnucoop_restaurant_info(soup)
            if not content:
                content = _clean_text(soup)

            if _save_record(fout, url, title, content, "cnucoop", "식단"):
                count += 1
                print(f"    OK — {len(content)}자")
            else:
                print("    건너뜀 — 내용 부족")
        except Exception as e:
            print(f"    오류: {e}")
        time.sleep(CRAWL_DELAY)

    return count


def _extract_cnucoop_restaurant_info(soup: BeautifulSoup) -> str:
    """cnucoop 식당 안내 페이지에서 식당별 정보를 구조화하여 추출한다."""
    lines = ["## 충남대 생활협동조합 식당 안내", ""]

    # 식당 정보가 포함된 테이블 찾기 (좌석수, 전화번호, 위치 등)
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        has_restaurant_info = False
        for row in rows[:5]:
            text = row.get_text(strip=True)
            if any(kw in text for kw in ["좌석수", "전화번호", "운영시간", "주요품목", "위치"]):
                has_restaurant_info = True
                break

        if has_restaurant_info and len(rows) > 2:
            for row in rows:
                cells = [td.get_text(separator=" ", strip=True) for td in row.find_all(["td", "th"])]
                cells = [c for c in cells if c]  # 빈 셀 제거
                if cells:
                    lines.append(" | ".join(cells))
            lines.append("")

    result = "\n".join(lines)
    return result if len(result) > 100 else ""


def crawl_mobile_meal(fout) -> int:
    """mobileadmin.cnu.ac.kr에서 주간 식단 메뉴를 수집한다 (레거시 호환).

    playwright가 필요했던 구버전. 이제 crawl_mobileadmin_meal()을 사용한다.
    """
    return crawl_mobileadmin_meal(fout)


def crawl_meal(output_path: Path | None = None) -> int:
    """식단 관련 모든 소스를 크롤링한다.

    Args:
        output_path: 출력 JSONL 경로 (None이면 기본 경로 사용)

    Returns:
        크롤링된 문서 수
    """
    if output_path is None:
        date_str = datetime.now().strftime("%Y%m%d")
        output_path = OUTPUT_DIR / f"meal_{date_str}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    print("\n[식단 크롤링 시작]")

    with open(output_path, "w", encoding="utf-8") as fout:
        # 1. mobileadmin.cnu.ac.kr 주간 식단 (POST — playwright 불필요)
        n = crawl_mobileadmin_meal(fout)
        total += n
        print(f"  mobileadmin.cnu.ac.kr: {n}건")

        # 2. plus.cnu.ac.kr 식당 안내
        n = crawl_plus_meal_pages(fout)
        total += n
        print(f"  plus.cnu.ac.kr: {n}건")

        # 3. cnucoop.co.kr 생협 식당 (기숙사 포함)
        n = crawl_cnucoop(fout)
        total += n
        print(f"  cnucoop.co.kr: {n}건")

    print(f"\n[식단 크롤링 완료] 총 {total}건 → {output_path}")
    return total


if __name__ == "__main__":
    crawl_meal()
