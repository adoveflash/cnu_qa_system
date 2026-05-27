"""JS 렌더링 페이지용 크롤러 (playwright).

게시판 목록처럼 AJAX로 동적 로딩되는 페이지의 article URL을 수집한 뒤,
실제 게시글은 requests(static)로 가져온다.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.sync_api import sync_playwright

from .robots_checker import USER_AGENT, is_allowed
from .static_crawler import CRAWL_DELAY, _clean_text, _extract_title

try:
    from bs4 import BeautifulSoup
except ImportError:
    pass


def _collect_article_urls_from_board(
    page,
    board_url: str,
    path_prefix: str,
    max_pages: int = 30,
) -> list[str]:
    """playwright page 객체로 게시판 목록을 순회해 게시글 URL을 수집한다.

    Args:
        page: playwright Page 객체
        board_url: 게시판 목록 시작 URL
        path_prefix: 이 경로 이하의 링크만 수집
        max_pages: 게시판 최대 페이지 수

    Returns:
        게시글 URL 목록
    """
    article_urls: list[str] = []
    visited_pages: set[str] = set()

    current_url = board_url
    for _ in range(max_pages):
        if current_url in visited_pages:
            break
        visited_pages.add(current_url)

        page.goto(current_url, wait_until="load", timeout=30000)
        time.sleep(1.5)  # JS 렌더링 완료 대기

        # 현재 페이지의 게시글 링크 수집
        anchors = page.query_selector_all("a[href]")
        new_articles = []
        for a in anchors:
            href = a.get_attribute("href") or ""
            full_url = urljoin(current_url, href).split("#")[0]
            parsed = urlparse(full_url)
            base_parsed = urlparse(path_prefix)
            if (
                "articleNo" in href
                and parsed.netloc == base_parsed.netloc
                and full_url not in article_urls
            ):
                new_articles.append(full_url)
                article_urls.append(full_url)

        if not new_articles:
            break

        # 다음 페이지 버튼 탐색
        next_btn = page.query_selector(
            "a.next, a[class*='next'], button[class*='next'], .pagination a:last-child"
        )
        if next_btn:
            next_href = next_btn.get_attribute("href")
            if next_href:
                next_url = urljoin(current_url, next_href).split("#")[0]
                if next_url != current_url and next_url not in visited_pages:
                    current_url = next_url
                    time.sleep(CRAWL_DELAY)
                    continue
            # href 없으면 클릭
            try:
                next_btn.click()
                page.wait_for_load_state("networkidle", timeout=10000)
                time.sleep(1.5)
                current_url = page.url
                continue
            except Exception:
                pass
        break

    return article_urls


def _fetch_article_content(page, url: str) -> dict | None:
    """playwright로 게시글 페이지 본문을 가져온다.

    Args:
        page: playwright Page 객체
        url: 게시글 URL

    Returns:
        {url, title, content, crawled_at, source} 또는 None
    """
    if not is_allowed(url):
        return None
    try:
        page.goto(url, wait_until="networkidle", timeout=20000)
        time.sleep(1.0)
        html = page.content()
        soup = BeautifulSoup(html, "lxml")
        title = _extract_title(soup)
        content = _clean_text(soup)
        if len(content) < 50:
            return None
        return {
            "url": url,
            "title": title,
            "content": content,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "source": urlparse(url).netloc,
        }
    except Exception as e:
        print(f"  [JS ERROR] {url}: {e}")
        return None


def crawl_js_boards(
    board_configs: list[dict],
    output_dir: Path,
    max_board_pages: int = 30,
    max_articles: int = 500,
) -> int:
    """JS 렌더링 게시판들을 크롤링한다.

    Args:
        board_configs: [{"board_url": ..., "path_prefix": ..., "source_name": ..., "output_path": ...}]
        output_dir: JSONL 저장 디렉터리
        max_board_pages: 게시판 목록 최대 페이지 수
        max_articles: 도메인당 최대 게시글 수

    Returns:
        총 저장 문서 수
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-images",
                "--blink-settings=imagesEnabled=false",
                "--js-flags=--max-old-space-size=256",
            ],
        )
        context = browser.new_context(
            user_agent=USER_AGENT,
            locale="ko-KR",
            java_script_enabled=True,
        )
        page = context.new_page()

        for cfg in board_configs:
            board_url: str = cfg["board_url"]
            path_prefix: str = cfg["path_prefix"]
            source_name: str = cfg["source_name"]
            output_path: Path = cfg["output_path"]

            print(f"\n[{source_name}] 게시판 URL 수집 중: {board_url}")
            article_urls = _collect_article_urls_from_board(
                page, board_url, path_prefix, max_pages=max_board_pages
            )
            article_urls = article_urls[:max_articles]
            print(f"  발견된 게시글 URL: {len(article_urls)}개")

            saved = 0
            with output_path.open("a", encoding="utf-8") as fout:
                for i, url in enumerate(article_urls, 1):
                    record = _fetch_article_content(page, url)
                    if record:
                        record["source"] = source_name
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        saved += 1
                        print(f"  [OK {saved:3d}/{len(article_urls)}] {record['title'][:50]}")
                    time.sleep(CRAWL_DELAY)

            print(f"  저장 완료: {saved}건")
            total += saved

        context.close()
        browser.close()

    return total
