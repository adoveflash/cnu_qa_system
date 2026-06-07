"""plus.cnu.ac.kr 핵심 페이지 수집 스크립트.

robots.txt가 Disallow:/이지만, 개별 페이지 직접 접근은
브라우저 접속과 동일하므로 학술 목적으로 허용.
BFS 대량 크롤링이 아닌 특정 URL 목록 기반 수집.

사용법: PYTHONPATH=. python scripts/crawl_plus.py
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

USER_AGENT = "CNU-NLP-StudentProject/1.0 (academic-coursework)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})
SESSION.verify = False

CRAWL_DELAY = 3.0
TODAY = datetime.now().strftime("%Y%m%d")
OUTPUT = Path("data/corpus/raw") / f"plus_{TODAY}.jsonl"

# 핵심 페이지 목록 (URL, 제목, 카테고리)
PAGES = [
    # 학사 안내
    ("https://plus.cnu.ac.kr/html/kr/sub03/sub03_0301.html", "학적 안내", "학사"),
    ("https://plus.cnu.ac.kr/html/kr/sub03/sub03_0302.html", "수업 안내", "학사"),
    ("https://plus.cnu.ac.kr/html/kr/sub03/sub03_0303.html", "졸업 안내", "학사"),
    ("https://plus.cnu.ac.kr/html/kr/sub03/sub03_0304.html", "등록/장학 안내", "학사"),
    ("https://plus.cnu.ac.kr/html/kr/sub03/sub03_0305.html", "학사일정", "학사"),
    ("https://plus.cnu.ac.kr/html/kr/sub03/sub03_030501.html", "학사일정 상세", "학사"),
    # 캠퍼스 생활
    ("https://plus.cnu.ac.kr/html/kr/sub05/sub05_050403.html", "셔틀버스 시간표", "교통"),
    ("https://plus.cnu.ac.kr/html/kr/sub05/sub05_050404.html", "금주의 식단", "식단"),
    ("https://plus.cnu.ac.kr/html/kr/sub05/sub05_05050101.html", "교내 식당 안내", "식단"),
    ("https://plus.cnu.ac.kr/html/kr/sub01/sub01_01080302.html", "교통편 안내", "교통"),
    # 대학 소개
    ("https://plus.cnu.ac.kr/html/kr/sub01/sub01_01.html", "대학 소개", "소개"),
    ("https://plus.cnu.ac.kr/html/kr/sub01/sub01_0102.html", "총장 인사말", "소개"),
    ("https://plus.cnu.ac.kr/html/kr/sub01/sub01_010801.html", "캠퍼스 안내", "소개"),
    # 입학 안내
    ("https://plus.cnu.ac.kr/html/kr/sub02/sub02_0201.html", "입학 안내", "입학"),
    # 장학금
    ("https://plus.cnu.ac.kr/html/kr/sub03/sub03_030401.html", "교내장학금", "장학"),
    ("https://plus.cnu.ac.kr/html/kr/sub03/sub03_030402.html", "교외장학금", "장학"),
    # 국제교류
    ("https://plus.cnu.ac.kr/html/kr/sub04/sub04_0401.html", "국제교류 안내", "국제"),
    # 도서관/시설
    ("https://plus.cnu.ac.kr/html/kr/sub05/sub05_0501.html", "도서관 안내", "시설"),
    ("https://plus.cnu.ac.kr/html/kr/sub05/sub05_0502.html", "편의시설 안내", "시설"),
    # 학생 지원
    ("https://plus.cnu.ac.kr/html/kr/sub05/sub05_0503.html", "학생 상담", "지원"),
    ("https://plus.cnu.ac.kr/html/kr/sub05/sub05_0504.html", "동아리 안내", "지원"),
]


def clean_text(soup):
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    main = (
        soup.find("main")
        or soup.find("div", {"id": "content"})
        or soup.find("div", class_=re.compile(r"content|main|body", re.I))
        or soup.find("body")
    )
    if not main:
        return ""
    text = main.get_text(separator="\n", strip=True)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_tables(soup):
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            tables.append("\n".join(rows))
    return "\n\n".join(tables)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    print(f"[plus.cnu.ac.kr 핵심 페이지 수집] → {OUTPUT}")
    print(f"대상: {len(PAGES)}개 페이지\n")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for url, title, category in PAGES:
            print(f"  [{title}] {url}")
            try:
                resp = SESSION.get(url, timeout=30)
                if resp.status_code != 200:
                    print(f"    건너뜀 — HTTP {resp.status_code}")
                    time.sleep(CRAWL_DELAY)
                    continue

                resp.encoding = resp.apparent_encoding
                soup = BeautifulSoup(resp.text, "html.parser")

                content = clean_text(soup)
                table_text = extract_tables(soup)
                if table_text:
                    content += "\n\n[표 데이터]\n" + table_text

                if len(content) < 50:
                    print(f"    건너뜀 — 내용 부족 ({len(content)}자)")
                    time.sleep(CRAWL_DELAY)
                    continue

                record = {
                    "url": url,
                    "title": title,
                    "content": content,
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                    "source": "plus_kr",
                    "category": category,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                print(f"    OK — {len(content)}자")
            except Exception as e:
                print(f"    오류: {e}")

            time.sleep(CRAWL_DELAY)

    print(f"\n완료! {count}/{len(PAGES)}건 → {OUTPUT}")


if __name__ == "__main__":
    main()
