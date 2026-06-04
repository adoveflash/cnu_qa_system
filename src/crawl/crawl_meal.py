"""식단 정보 크롤러.

수집 대상:
1. plus.cnu.ac.kr/html/kr/ — 식당 위치/운영시간, 금주의식단 안내
2. www.cnucoop.co.kr — 생협 식당 안내 (기숙사 식당 포함)
3. mobileadmin.cnu.ac.kr/food/index.jsp — 주간 식단 메뉴 (JS 렌더링)
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .robots_checker import USER_AGENT

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})
SESSION.verify = False

CRAWL_DELAY = 3.0
OUTPUT_DIR = Path("data/corpus/raw")


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
                print(f"    건너뜀 — 내용 부족")
        except Exception as e:
            print(f"    오류: {e}")
        time.sleep(CRAWL_DELAY)

    return count


def crawl_cnucoop(fout) -> int:
    """cnucoop.co.kr 생활협동조합 식당 안내 페이지를 수집한다 (기숙사 식당 포함)."""
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
            # EUC-KR 인코딩 처리
            if "euc-kr" in resp.headers.get("Content-Type", "").lower():
                resp.encoding = "euc-kr"
            elif resp.apparent_encoding:
                resp.encoding = resp.apparent_encoding
            soup = BeautifulSoup(resp.text, "html.parser")
            title = _extract_title(soup) or desc
            content = _clean_text(soup)
            if _save_record(fout, url, title, content, "cnucoop", "식단"):
                count += 1
                print(f"    OK — {len(content)}자")
            else:
                print(f"    건너뜀 — 내용 부족")
        except Exception as e:
            print(f"    오류: {e}")
        time.sleep(CRAWL_DELAY)

    return count


def crawl_mobile_meal(fout) -> int:
    """mobileadmin.cnu.ac.kr에서 주간 식단 메뉴를 수집한다.

    JS 렌더링 페이지이므로 playwright가 필요하다.
    playwright가 없으면 건너뛴다.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [mobileadmin] playwright 미설치 → 건너뜀")
        return 0

    base_url = "https://mobileadmin.cnu.ac.kr/food/index.jsp"
    print(f"  [mobileadmin] 주간 식단 (playwright): {base_url}")

    count = 0
    from datetime import timedelta

    today = datetime.now()
    dates = [(today + timedelta(days=i)).strftime("%Y%m%d") for i in range(7)]

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            for date_str in dates:
                try:
                    url = f"{base_url}?searchYmd={date_str}"
                    page.goto(url, wait_until="networkidle", timeout=30000)
                    page.wait_for_timeout(2000)

                    content = page.inner_text("body")
                    content = re.sub(r"\n{3,}", "\n\n", content).strip()

                    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                    title = f"충남대 식단표 ({formatted_date})"

                    if _save_record(fout, url, title, content, "mobileadmin", "식단"):
                        count += 1
                        print(f"    OK — {formatted_date} ({len(content)}자)")
                    else:
                        print(f"    건너뜀 — {formatted_date} 내용 부족")

                except Exception as e:
                    print(f"    오류 ({date_str}): {e}")
                time.sleep(CRAWL_DELAY)

            browser.close()
    except Exception as e:
        print(f"  [mobileadmin] playwright 실행 오류: {e}")

    return count


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
        # 1. plus.cnu.ac.kr 식당 안내
        n = crawl_plus_meal_pages(fout)
        total += n
        print(f"  plus.cnu.ac.kr: {n}건")

        # 2. cnucoop.co.kr 생협 식당 (기숙사 포함)
        n = crawl_cnucoop(fout)
        total += n
        print(f"  cnucoop.co.kr: {n}건")

        # 3. mobileadmin.cnu.ac.kr 주간 식단
        n = crawl_mobile_meal(fout)
        total += n
        print(f"  mobileadmin.cnu.ac.kr: {n}건")

    print(f"\n[식단 크롤링 완료] 총 {total}건 → {output_path}")
    return total


if __name__ == "__main__":
    crawl_meal()
