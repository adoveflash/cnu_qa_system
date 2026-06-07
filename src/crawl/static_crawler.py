"""정적 HTML 페이지 BFS 크롤러 (requests + BeautifulSoup)."""

import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .robots_checker import USER_AGENT, is_allowed

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})

CRAWL_DELAY = 3.0  # 서버 부담 최소화


def _clean_text(soup: BeautifulSoup) -> str:
    """HTML에서 불필요한 태그를 제거하고 본문 텍스트를 반환한다.

    Args:
        soup: 파싱된 BeautifulSoup 객체

    Returns:
        정제된 텍스트 문자열
    """
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()

    # 본문 영역 우선 탐색
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"content|main|body", re.I))
        or soup.find(class_=re.compile(r"content|main|body", re.I))
        or soup.find("body")
    )
    text = (main or soup).get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def _extract_title(soup: BeautifulSoup) -> str:
    """페이지 제목을 추출한다."""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return ""


def _same_scope(url: str, base_prefix: str) -> bool:
    """URL이 크롤링 대상 경로 범위 안에 있는지 확인한다."""
    parsed = urlparse(url)
    base_parsed = urlparse(base_prefix)
    return parsed.netloc == base_parsed.netloc and parsed.path.startswith(base_parsed.path)


def crawl_static(
    seed_url: str,
    path_prefix: str,
    output_path: Path,
    max_pages: int = 200,
    source_name: Optional[str] = None,
) -> int:
    """seed_url에서 시작해 path_prefix 범위 내 페이지를 BFS 크롤링한다.

    Args:
        seed_url: 크롤링 시작 URL
        path_prefix: 이 경로 이하만 크롤링 (예: "https://example.com/html/")
        output_path: 결과를 저장할 .jsonl 파일 경로
        max_pages: 최대 수집 페이지 수
        source_name: JSONL source 필드에 사용할 이름

    Returns:
        실제로 저장된 문서 수
    """
    visited: set[str] = set()
    queue: deque[str] = deque([seed_url])
    saved = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        while queue and saved < max_pages:
            url = queue.popleft()
            # 정규화 (fragment 제거)
            url = url.split("#")[0]
            if url in visited:
                continue
            visited.add(url)

            if not _same_scope(url, path_prefix):
                continue
            if not is_allowed(url):
                print(f"  [SKIP robots.txt] {url}")
                continue

            try:
                resp = SESSION.get(url, timeout=15, allow_redirects=True)
                resp.raise_for_status()
            except Exception as e:
                print(f"  [ERROR] {url}: {e}")
                time.sleep(CRAWL_DELAY)
                continue

            content_type = resp.headers.get("Content-Type", "")
            if "html" not in content_type:
                time.sleep(CRAWL_DELAY)
                continue

            soup = BeautifulSoup(resp.text, "lxml")
            title = _extract_title(soup)
            content = _clean_text(soup)

            if len(content) < 50:  # 내용이 거의 없는 페이지는 건너뜀
                time.sleep(CRAWL_DELAY)
                continue

            record = {
                "url": url,
                "title": title,
                "content": content,
                "crawled_at": datetime.now(timezone.utc).isoformat(),
                "source": source_name or urlparse(url).netloc,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1
            print(f"  [OK {saved:3d}] {title[:60]} — {url}")

            # 같은 도메인 내 링크 수집
            for a in soup.find_all("a", href=True):
                next_url = urljoin(url, a["href"]).split("#")[0]
                if next_url not in visited and _same_scope(next_url, path_prefix):
                    queue.append(next_url)

            time.sleep(CRAWL_DELAY)

    return saved


def crawl_board_offset(
    board_url: str,
    output_path: Path,
    source_name: str,
    article_limit: int = 10,
    max_articles: int = 500,
) -> int:
    """article.offset 파라미터로 게시판 목록을 순회해 게시글 본문을 수집한다.

    게시판 목록이 정적 HTML에 articleNo 링크를 포함하지만 JS 기반 페이지네이션을
    쓰는 경우, article.offset=0,10,20,... 으로 직접 오프셋을 지정해 수집한다.

    Args:
        board_url: 게시판 목록 기본 URL (쿼리스트링 제외)
        output_path: 결과를 추가(append)할 .jsonl 파일 경로
        source_name: JSONL source 필드에 사용할 이름
        article_limit: 한 목록 페이지당 게시글 수 (기본 10)
        max_articles: 최대 수집 게시글 수

    Returns:
        저장된 게시글 수
    """
    import re as _re

    output_path.parent.mkdir(parents=True, exist_ok=True)
    visited: set[str] = set()
    saved = 0

    with output_path.open("a", encoding="utf-8") as fout:
        offset = 0
        consecutive_empty = 0

        while saved < max_articles:
            list_url = f"{board_url}?article.offset={offset}&articleLimit={article_limit}"
            resp = None
            for attempt in range(3):
                try:
                    resp = SESSION.get(list_url, timeout=30, allow_redirects=True)
                    resp.raise_for_status()
                    break
                except Exception as e:
                    print(f"  [LIST RETRY {attempt + 1}] offset={offset}: {e}")
                    time.sleep(5)
            if resp is None:
                break

            soup = BeautifulSoup(resp.text, "lxml")
            article_hrefs = [
                a["href"] for a in soup.find_all("a", href=True) if "articleNo" in a.get("href", "")
            ]

            # offset=0의 고정 공지글 수 기억 (이후 페이지에서 제외할 기준)
            all_nos = set(
                _re.search(r"articleNo=(\d+)", h).group(1)
                for h in article_hrefs
                if _re.search(r"articleNo=(\d+)", h)
            )
            new_nos = all_nos - visited
            if not new_nos:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    break
                offset += article_limit
                time.sleep(CRAWL_DELAY)
                continue
            consecutive_empty = 0

            for href in article_hrefs:
                m = _re.search(r"articleNo=(\d+)", href)
                if not m:
                    continue
                article_no = m.group(1)
                if article_no in visited:
                    continue
                visited.add(article_no)

                article_url = urljoin(board_url, href).split("#")[0]
                if not is_allowed(article_url):
                    continue

                art_resp = None
                for attempt in range(3):
                    try:
                        art_resp = SESSION.get(article_url, timeout=30, allow_redirects=True)
                        art_resp.raise_for_status()
                        break
                    except Exception as e:
                        print(f"  [ART RETRY {attempt + 1}] {article_url}: {e}")
                        time.sleep(5)
                if art_resp is None:
                    continue

                art_soup = BeautifulSoup(art_resp.text, "lxml")
                title = _extract_title(art_soup)
                content = _clean_text(art_soup)

                if len(content) < 50:
                    time.sleep(CRAWL_DELAY)
                    continue

                record = {
                    "url": article_url,
                    "title": title,
                    "content": content,
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                    "source": source_name,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved += 1
                print(f"  [OK {saved:3d}] {title[:55]} — articleNo={article_no}")

                time.sleep(CRAWL_DELAY)

                if saved >= max_articles:
                    break

            offset += article_limit
            time.sleep(CRAWL_DELAY)

    return saved
