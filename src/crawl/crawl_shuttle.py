"""셔틀버스/통학 정보 크롤러.

수집 대상 (모두 plus.cnu.ac.kr/html/kr/ — 크롤링 허용):
1. 셔틀버스 안내 — 시간표, 노선, 정류장 18개
2. 교통편 안내 — 시내버스 노선, 찾아오는 길
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

# 셔틀버스/교통 관련 페이지 목록
SHUTTLE_PAGES = [
    {
        "url": "https://plus.cnu.ac.kr/html/kr/sub05/sub05_050403.html",
        "title": "학교셔틀버스 안내",
        "desc": "교내순환 시간표, 캠퍼스간 순환, 정류장 18개소, 운행시간",
    },
    {
        "url": "https://plus.cnu.ac.kr/html/kr/sub01/sub01_01080302.html",
        "title": "교통편 안내 (찾아오시는 길)",
        "desc": "시내버스 노선, 지하철, 고속버스, 기차역 연결, 자가용 안내",
    },
    {
        "url": "https://plus.cnu.ac.kr/html/kr/sub01/sub01_01080301.html",
        "title": "캠퍼스 안내 (대덕캠퍼스)",
        "desc": "캠퍼스 맵, 건물 위치 — 정류장 위치 참고용",
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
    """페이지 내 테이블을 마크다운 형식으로 변환한다 (시간표 등)."""
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


def crawl_shuttle(output_path: Path | None = None) -> int:
    """셔틀버스/교통 관련 페이지를 크롤링한다.

    Args:
        output_path: 출력 JSONL 경로 (None이면 기본 경로 사용)

    Returns:
        크롤링된 문서 수
    """
    if output_path is None:
        date_str = datetime.now().strftime("%Y%m%d")
        output_path = OUTPUT_DIR / f"shuttle_{date_str}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    print("\n[셔틀버스/교통 크롤링 시작]")

    with open(output_path, "w", encoding="utf-8") as fout:
        for page in SHUTTLE_PAGES:
            url = page["url"]
            print(f"  [{page['title']}] {url}")

            try:
                resp = SESSION.get(url, timeout=30)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding
                soup = BeautifulSoup(resp.text, "html.parser")

                title = _extract_title(soup) or page["title"]
                content = _clean_text(soup)

                # 테이블 데이터 추가 (시간표 등)
                table_text = _extract_tables(soup)
                if table_text:
                    content = content + "\n\n[시간표/표 데이터]\n" + table_text

                if len(content) < 50:
                    print("    건너뜀 — 내용 부족")
                    time.sleep(CRAWL_DELAY)
                    continue

                record = {
                    "url": url,
                    "title": title,
                    "content": content,
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                    "source": "plus_kr",
                    "category": "셔틀버스",
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                print(f"    OK — {len(content)}자")

            except Exception as e:
                print(f"    오류: {e}")

            time.sleep(CRAWL_DELAY)

        # 추가: plus.cnu.ac.kr 내 셔틀/교통 관련 하위 페이지 탐색
        count += _crawl_shuttle_subpages(fout)

    print(f"\n[셔틀버스/교통 크롤링 완료] 총 {count}건 → {output_path}")
    return count


def _crawl_shuttle_subpages(fout) -> int:
    """plus.cnu.ac.kr 내 교통/셔틀 관련 하위 페이지를 추가 수집한다."""
    sub_urls = [
        # 시내버스 노선 안내 (캠퍼스별)
        "https://plus.cnu.ac.kr/html/kr/sub05/sub05_050402.html",
    ]

    count = 0
    for url in sub_urls:
        try:
            resp = SESSION.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            resp.encoding = resp.apparent_encoding
            soup = BeautifulSoup(resp.text, "html.parser")
            title = _extract_title(soup)
            content = _clean_text(soup)

            keywords = ["셔틀", "버스", "교통", "통학", "정류장", "노선", "운행"]
            if len(content) >= 50 and any(kw in content for kw in keywords):
                table_text = _extract_tables(soup)
                if table_text:
                    content = content + "\n\n[시간표/표 데이터]\n" + table_text

                record = {
                    "url": url,
                    "title": title,
                    "content": content,
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                    "source": "plus_kr",
                    "category": "셔틀버스",
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                print(f"  [하위 페이지] OK — {title} ({len(content)}자)")
        except Exception:
            pass
        time.sleep(CRAWL_DELAY)

    return count


if __name__ == "__main__":
    crawl_shuttle()
