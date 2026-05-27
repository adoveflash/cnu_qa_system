"""plus.cnu.ac.kr BFS 크롤러.

사이트맵에서 시드 URL을 추출하고, 크롤링한 페이지 내 링크를 BFS로 따라가며
plus.cnu.ac.kr 내 접근 가능한 모든 페이지를 수집한다.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .robots_checker import USER_AGENT, is_allowed

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})
SESSION.verify = False

CRAWL_DELAY = 3.0
BASE_URL = "https://plus.cnu.ac.kr"
SITEMAP_URL = f"{BASE_URL}/html/kr/guide/guide_0801.html"
OUTPUT_DIR = Path("data/corpus/raw")

# 크롤링 제외 확장자 / 패턴
_SKIP_EXTENSIONS = {".hwp", ".pdf", ".zip", ".xlsx", ".xls", ".doc", ".docx", ".ppt", ".pptx",
                    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".mp4", ".mp3"}
_SKIP_PATTERNS = {"#", "javascript:", "mailto:", "tel:"}


def _should_skip_url(url: str) -> bool:
    """크롤링하지 않을 URL인지 판별한다."""
    lower = url.lower()
    for pat in _SKIP_PATTERNS:
        if pat in lower:
            return True
    parsed = urlparse(lower)
    suffix = Path(parsed.path).suffix
    return suffix in _SKIP_EXTENSIONS


def _is_plus_cnu(url: str) -> bool:
    """plus.cnu.ac.kr 도메인의 허용된 경로인지 확인한다."""
    parsed = urlparse(url)
    if parsed.netloc != "plus.cnu.ac.kr":
        return False
    path = parsed.path
    return path.startswith("/html/kr/") or path.startswith("/html/hub/") or path.startswith("/_prog/")


def _clean_text(soup: BeautifulSoup) -> str:
    """HTML에서 불필요한 태그를 제거하고 본문 텍스트를 반환한다."""
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    main = soup.find("main") or soup.find("div", {"id": "content"}) or soup.find("body")
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


def _extract_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """페이지 내 plus.cnu.ac.kr 링크를 추출한다."""
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)
        # 쿼리스트링은 유지하되 fragment는 제거
        parsed = urlparse(full_url)
        clean_url = parsed._replace(fragment="").geturl()
        if _is_plus_cnu(clean_url) and not _should_skip_url(clean_url):
            links.append(clean_url)
    return links


def extract_sitemap_urls() -> list[str]:
    """사이트맵 페이지에서 시드 URL 목록을 추출한다."""
    print(f"  사이트맵 로드: {SITEMAP_URL}")
    resp = SESSION.get(SITEMAP_URL, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, "html.parser")

    urls: list[str] = []
    seen: set[str] = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(BASE_URL, href)
        if _is_plus_cnu(full_url) and not _should_skip_url(full_url) and full_url not in seen:
            seen.add(full_url)
            urls.append(full_url)

    print(f"  사이트맵에서 {len(urls)}개 시드 URL 추출")
    return urls


def crawl_plus(output_path: Path | None = None) -> int:
    """plus.cnu.ac.kr를 BFS로 크롤링한다.

    사이트맵에서 시드 URL을 추출한 뒤, 각 페이지 내 링크를 따라가며
    접근 가능한 모든 페이지를 수집한다.

    Args:
        output_path: 출력 JSONL 경로 (None이면 기본 경로 사용)

    Returns:
        크롤링된 문서 수
    """
    if output_path is None:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        output_path = OUTPUT_DIR / f"plus_kr_{date_str}.jsonl"

    seed_urls = extract_sitemap_urls()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    queue: deque[str] = deque(seed_urls)
    visited: set[str] = set()
    count = 0
    total_discovered = len(seed_urls)

    with open(output_path, "w", encoding="utf-8") as f:
        while queue:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            if not is_allowed(url):
                print(f"  [{len(visited)}/{total_discovered}] robots.txt 차단: {url}")
                continue

            try:
                print(f"  [{len(visited)}/{total_discovered}] {url}", end=" ", flush=True)
                resp = SESSION.get(url, timeout=30)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding
                soup = BeautifulSoup(resp.text, "html.parser")

                # BFS: 페이지 내 새 링크 발견
                new_links = _extract_links(soup, url)
                for link in new_links:
                    if link not in visited:
                        queue.append(link)
                        total_discovered += 1

                title = _extract_title(soup)
                content = _clean_text(soup)

                if len(content) < 50:
                    print("— 콘텐츠 부족, 건너뜀")
                    continue

                record = {
                    "url": url,
                    "title": title,
                    "content": content,
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                    "source": "plus_kr",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                count += 1
                print(f"— {len(content)}자 (큐: {len(queue)})")

            except requests.RequestException as e:
                print(f"— 오류: {e}")

            time.sleep(CRAWL_DELAY)

    print(f"\n크롤링 완료: {count}건 (방문 {len(visited)}개) → {output_path}")
    return count


if __name__ == "__main__":
    crawl_plus()
