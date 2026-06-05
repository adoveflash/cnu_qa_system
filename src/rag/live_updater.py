"""실시간(동적) 정보 RAG 갱신 모듈.

주기적으로 변하는 정보(식단, 공지사항 등)를 크롤링하여
ChromaDB 'live' 컬렉션을 갱신한다.

사용법:
    python -m src.rag.live_updater          # 전체 갱신
    python -m src.rag.live_updater --meal   # 식단만
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import chromadb
import requests
from bs4 import BeautifulSoup

from src.rag.embedder import embed_texts, load_embedding_model

_LIVE_COLLECTION = "cnu_live"
_DEFAULT_DB_PATH = Path("data/vector_db")
USER_AGENT = "CNU-NLP-StudentProject/1.0 (academic-coursework)"
CRAWL_DELAY = 3.0

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})
SESSION.verify = False


# ── 식단 크롤링 ─────────────────────────────────────────────────────────


def _fetch_cnucoop_menu() -> list[dict[str, str]]:
    """cnucoop.co.kr에서 식당 정보를 가져온다 (기숙사 식당 포함)."""
    url = "https://www.cnucoop.co.kr/ezhtml2.php?html=canteen"
    docs = []
    try:
        resp = SESSION.get(url, timeout=30)
        resp.raise_for_status()
        if "euc-kr" in resp.headers.get("Content-Type", "").lower():
            resp.encoding = "euc-kr"
        elif resp.apparent_encoding:
            resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        body = soup.find("body")
        if body:
            content = body.get_text(separator="\n", strip=True)
            content = re.sub(r"\n{3,}", "\n\n", content)
            if len(content) > 50:
                docs.append(
                    {
                        "text": content,
                        "url": url,
                        "title": "충남대 생협 식당 안내 (기숙사 식당 포함)",
                        "source": "cnucoop",
                        "category": "식단",
                    }
                )
    except Exception as e:
        print(f"  [cnucoop] 오류: {e}")
    return docs


def _fetch_mobile_meal_requests() -> list[dict[str, str]]:
    """mobileadmin.cnu.ac.kr에서 주간 식단을 requests로 가져온다.

    JS 렌더링 페이지라 빈 결과일 수 있다. 그래도 시도.
    """
    base_url = "https://mobileadmin.cnu.ac.kr/food/index.jsp"
    docs = []
    today = datetime.now()

    for delta in range(7):
        date = today + timedelta(days=delta)
        date_str = date.strftime("%Y%m%d")
        formatted = date.strftime("%Y-%m-%d")

        try:
            # GET 파라미터 시도
            resp = SESSION.get(f"{base_url}?searchYmd={date_str}", timeout=30)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            soup = BeautifulSoup(resp.text, "html.parser")

            for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # 테이블 기반으로 식단 추출
            tables = soup.find_all("table")
            content_parts = []
            for table in tables:
                for tr in table.find_all("tr"):
                    cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                    row_text = " | ".join(c for c in cells if c)
                    if row_text:
                        content_parts.append(row_text)

            if not content_parts:
                body = soup.find("body")
                if body:
                    text = body.get_text(separator="\n", strip=True)
                    text = re.sub(r"\n{3,}", "\n\n", text)
                    if len(text) > 50:
                        content_parts = [text]

            if content_parts:
                content = f"충남대학교 식단표 ({formatted})\n\n" + "\n".join(content_parts)
                docs.append(
                    {
                        "text": content,
                        "url": f"{base_url}?searchYmd={date_str}",
                        "title": f"충남대 식단표 ({formatted})",
                        "source": "mobileadmin",
                        "category": "식단",
                    }
                )
        except Exception:
            pass
        time.sleep(CRAWL_DELAY)

    return docs


def _fetch_mobile_meal_playwright() -> list[dict[str, str]]:
    """mobileadmin.cnu.ac.kr에서 playwright로 주간 식단을 가져온다."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return []

    base_url = "https://mobileadmin.cnu.ac.kr/food/index.jsp"
    docs = []
    today = datetime.now()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            for delta in range(7):
                date = today + timedelta(days=delta)
                date_str = date.strftime("%Y%m%d")
                formatted = date.strftime("%Y-%m-%d")

                try:
                    url = f"{base_url}?searchYmd={date_str}"
                    page.goto(url, wait_until="networkidle", timeout=30000)
                    page.wait_for_timeout(2000)
                    content = page.inner_text("body")
                    content = re.sub(r"\n{3,}", "\n\n", content).strip()

                    if len(content) > 50:
                        content = f"충남대학교 식단표 ({formatted})\n\n{content}"
                        docs.append(
                            {
                                "text": content,
                                "url": url,
                                "title": f"충남대 식단표 ({formatted})",
                                "source": "mobileadmin",
                                "category": "식단",
                            }
                        )
                except Exception:
                    pass
                time.sleep(CRAWL_DELAY)

            browser.close()
    except Exception:
        pass

    return docs


def fetch_meal_docs() -> list[dict[str, str]]:
    """식단 관련 문서를 모두 가져온다."""
    docs = []

    print("  [식단] cnucoop.co.kr 크롤링...")
    docs.extend(_fetch_cnucoop_menu())

    print("  [식단] mobileadmin.cnu.ac.kr 크롤링...")
    pw_docs = _fetch_mobile_meal_playwright()
    if pw_docs:
        docs.extend(pw_docs)
        print(f"    playwright: {len(pw_docs)}건")
    else:
        req_docs = _fetch_mobile_meal_requests()
        docs.extend(req_docs)
        print(f"    requests fallback: {len(req_docs)}건")

    return docs


# ── 공지사항 (최신) ──────────────────────────────────────────────────────


def fetch_notice_docs() -> list[dict[str, str]]:
    """최신 공지사항을 가져온다 (computer.cnu.ac.kr 최근 10건)."""
    board_url = "https://computer.cnu.ac.kr/computer/notice/bachelor.do"
    docs = []

    try:
        resp = SESSION.get(board_url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        article_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "articleNo" in href:
                from urllib.parse import urljoin

                full_url = urljoin(board_url, href).split("#")[0]
                if full_url not in article_links:
                    article_links.append(full_url)

        for url in article_links[:10]:
            try:
                resp = SESSION.get(url, timeout=30)
                resp.raise_for_status()
                art_soup = BeautifulSoup(resp.text, "html.parser")

                for tag in art_soup.find_all(["script", "style", "nav", "footer"]):
                    tag.decompose()

                title_tag = art_soup.find("h1") or art_soup.find("title")
                title = title_tag.get_text(strip=True) if title_tag else ""
                body = art_soup.find("div", class_=re.compile(r"content|view|body", re.I))
                content = (body or art_soup.find("body")).get_text(separator="\n", strip=True)
                content = re.sub(r"\n{3,}", "\n\n", content)

                if len(content) > 50:
                    docs.append(
                        {
                            "text": content,
                            "url": url,
                            "title": title,
                            "source": "computer",
                            "category": "공지사항",
                        }
                    )
            except Exception:
                pass
            time.sleep(CRAWL_DELAY)

    except Exception as e:
        print(f"  [공지] 오류: {e}")

    return docs


# ── 벡터DB 갱신 ──────────────────────────────────────────────────────────


def refresh_live_collection(
    db_path: Path = _DEFAULT_DB_PATH,
    collection_name: str = _LIVE_COLLECTION,
    embedding_model_name: str = "BAAI/bge-m3",
    include_meal: bool = True,
    include_notice: bool = True,
) -> int:
    """동적 정보를 크롤링하여 live 컬렉션을 갱신한다.

    기존 live 컬렉션을 삭제하고 최신 데이터로 재생성한다.

    Args:
        db_path: ChromaDB 경로
        collection_name: live 컬렉션 이름
        embedding_model_name: 임베딩 모델명
        include_meal: 식단 정보 포함 여부
        include_notice: 공지사항 포함 여부

    Returns:
        인덱싱된 문서 수
    """
    print("=" * 50)
    print(f"[Live 컬렉션 갱신] {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)

    all_docs: list[dict[str, str]] = []

    if include_meal:
        meal_docs = fetch_meal_docs()
        all_docs.extend(meal_docs)
        print(f"  식단: {len(meal_docs)}건")

    if include_notice:
        notice_docs = fetch_notice_docs()
        all_docs.extend(notice_docs)
        print(f"  공지사항: {len(notice_docs)}건")

    if not all_docs:
        print("  수집된 문서 없음 — 갱신 건너뜀")
        return 0

    # 임베딩
    print(f"\n  {len(all_docs)}건 임베딩 중...")
    model = load_embedding_model(embedding_model_name)
    texts = [d["text"] for d in all_docs]
    embeddings = embed_texts(texts, model, batch_size=32)

    # ChromaDB 갱신 (기존 삭제 → 재생성)
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))

    try:
        client.delete_collection(collection_name)
    except (ValueError, Exception):
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    now_str = datetime.now(timezone.utc).isoformat()
    ids = [f"live_{i}" for i in range(len(all_docs))]
    metadatas = [
        {
            "url": d["url"],
            "title": d["title"],
            "source": d["source"],
            "category": d.get("category", ""),
            "updated_at": now_str,
        }
        for d in all_docs
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"\n  Live 컬렉션 갱신 완료: {collection.count()}건")
    return collection.count()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live 컬렉션 갱신")
    parser.add_argument("--meal", action="store_true", help="식단만 갱신")
    parser.add_argument("--notice", action="store_true", help="공지만 갱신")
    args = parser.parse_args()

    # 둘 다 지정 안 하면 전부 갱신
    if not args.meal and not args.notice:
        args.meal = True
        args.notice = True

    refresh_live_collection(include_meal=args.meal, include_notice=args.notice)
