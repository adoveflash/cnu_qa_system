"""학사일정 크롤러.

수집 대상:
1. plus.cnu.ac.kr 학사일정 페이지
2. computer.cnu.ac.kr 학사일정/졸업요건 페이지
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

# 학사일정 관련 페이지 목록
ACADEMIC_PAGES = [
    {
        "url": "https://plus.cnu.ac.kr/html/kr/sub03/sub03_0305.html",
        "title": "학사일정",
        "desc": "충남대 공식 학사일정 (연간)",
    },
    {
        "url": "https://plus.cnu.ac.kr/html/kr/sub03/sub03_030501.html",
        "title": "학사일정 상세",
        "desc": "월별 학사일정 상세",
    },
    {
        "url": "https://plus.cnu.ac.kr/html/kr/sub03/sub03_0301.html",
        "title": "학적 안내",
        "desc": "휴학/복학/전과/졸업 관련",
    },
    {
        "url": "https://plus.cnu.ac.kr/html/kr/sub03/sub03_0302.html",
        "title": "수업 안내",
        "desc": "수강신청/수강정정/성적 관련",
    },
    {
        "url": "https://plus.cnu.ac.kr/html/kr/sub03/sub03_0303.html",
        "title": "졸업 안내",
        "desc": "졸업요건/학위수여 관련",
    },
    {
        "url": "https://plus.cnu.ac.kr/html/kr/sub03/sub03_0304.html",
        "title": "등록/장학 안내",
        "desc": "등록금/장학금 관련",
    },
    # computer.cnu.ac.kr 졸업요건 게시판
    {
        "url": "https://computer.cnu.ac.kr/computer/edu/graduation.do",
        "title": "컴퓨터융합학부 졸업요건",
        "desc": "전공별 졸업요건, 학점 기준",
    },
    {
        "url": "https://computer.cnu.ac.kr/computer/edu/curriculum.do",
        "title": "컴퓨터융합학부 교육과정",
        "desc": "전공 교육과정, 트랙 안내",
    },
]


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


def _extract_tables(soup: BeautifulSoup) -> str:
    """페이지 내 테이블을 텍스트로 변환한다."""
    tables_text = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            tables_text.append("\n".join(rows))
    return "\n\n".join(tables_text)


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


def crawl_academic(output_path: Path | None = None) -> int:
    """학사일정/학적/졸업 관련 페이지를 크롤링한다.

    Args:
        output_path: 출력 JSONL 경로 (None이면 기본 경로 사용)

    Returns:
        크롤링된 문서 수
    """
    if output_path is None:
        date_str = datetime.now().strftime("%Y%m%d")
        output_path = OUTPUT_DIR / f"academic_{date_str}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    print("\n[학사일정/졸업요건 크롤링 시작]")

    with open(output_path, "w", encoding="utf-8") as fout:
        for page in ACADEMIC_PAGES:
            url = page["url"]
            print(f"  [{page['title']}] {url}")

            try:
                resp = SESSION.get(url, timeout=30)
                if resp.status_code != 200:
                    print(f"    건너뜀 — HTTP {resp.status_code}")
                    time.sleep(CRAWL_DELAY)
                    continue

                resp.encoding = resp.apparent_encoding
                soup = BeautifulSoup(resp.text, "html.parser")

                title = _extract_title(soup) or page["title"]
                content = _clean_text(soup)

                # 테이블 데이터 추가 (학사일정표 등)
                table_text = _extract_tables(soup)
                if table_text:
                    content = content + "\n\n[일정표/표 데이터]\n" + table_text

                if len(content) < 50:
                    print("    건너뜀 — 내용 부족")
                    time.sleep(CRAWL_DELAY)
                    continue

                record = {
                    "url": url,
                    "title": title,
                    "content": content,
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                    "source": "plus_kr" if "plus.cnu.ac.kr" in url else "computer",
                    "category": "학사일정",
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                print(f"    OK — {len(content)}자")

            except Exception as e:
                print(f"    오류: {e}")

            time.sleep(CRAWL_DELAY)

    print(f"\n[학사일정 크롤링 완료] 총 {count}건 → {output_path}")
    return count


if __name__ == "__main__":
    crawl_academic()
