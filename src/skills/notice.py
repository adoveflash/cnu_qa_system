"""공지사항 스킬 — 컴퓨터융합학부 학사공지 최신순 조회.

목록에서 제목·날짜를 추출하고, 첫 공지만 상세 본문을 가져온다(응답 속도).
"""

from __future__ import annotations

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from src.skills.base import SESSION, Skill
from src.skills.registry import register


def get_notices(count: int = 5) -> str:
    """최신 공지사항을 크롤링하여 반환한다.

    Args:
        count: 가져올 공지 수 (최대 10)

    Returns:
        공지사항 텍스트
    """
    count = min(max(count, 1), 10)
    board_url = "https://computer.cnu.ac.kr/computer/notice/bachelor.do"

    try:
        resp = SESSION.get(board_url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 게시글 링크 + 날짜 추출
        article_items: list[tuple[str, str]] = []  # (url, date_str)
        rows = soup.find_all("tr")
        for row in rows:
            a_tag = row.find("a", href=True)
            if not a_tag or "articleNo" not in a_tag["href"]:
                continue
            full_url = urljoin(board_url, a_tag["href"]).split("#")[0]
            date_str = "0000.00.00"
            for td in row.find_all("td"):
                td_text = td.get_text(strip=True)
                m = re.match(r"(\d{2,4})\.(\d{2})\.(\d{2})", td_text)
                if m:
                    year = m.group(1)
                    if len(year) == 2:
                        year = "20" + year
                    date_str = f"{year}.{m.group(2)}.{m.group(3)}"
                    break
            if full_url not in [x[0] for x in article_items]:
                article_items.append((full_url, date_str))

        # 날짜 내림차순 정렬 (최신 우선)
        article_items.sort(key=lambda x: x[1], reverse=True)
        results: list[str] = []
        for i, (url, date_str) in enumerate(article_items[:count]):
            rank = i + 1
            for row in rows:
                a_tag = row.find("a", href=True)
                if a_tag and url.endswith(a_tag["href"].split("#")[0].split("?")[-1]):
                    title = a_tag.get_text(strip=True)
                    results.append(f"[{rank}번째 최신 공지] ({date_str}) {title}\n출처: {url}")
                    break

        # 첫 번째 공지만 상세 내용 가져오기 (응답 속도 위해)
        if article_items:
            try:
                first_url = article_items[0][0]
                resp = SESSION.get(first_url, timeout=30)
                resp.raise_for_status()
                art_soup = BeautifulSoup(resp.text, "html.parser")
                for tag in art_soup.find_all(["script", "style", "nav", "footer"]):
                    tag.decompose()
                body = art_soup.find("div", class_=re.compile(r"content|view|body", re.I))
                text = (body or art_soup.find("body")).get_text(separator="\n", strip=True)
                text = re.sub(r"\n{3,}", "\n\n", text)
                if results:
                    results[0] += f"\n\n[상세 내용]\n{text[:500]}"
            except Exception:
                pass

        if results:
            return "최신순 정렬 (1번이 가장 최근):\n\n" + "\n\n---\n\n".join(results)
    except Exception as e:
        return f"공지사항 조회 실패: {e}"

    return "공지사항을 가져올 수 없습니다."


def _infer_notice_args(question: str) -> dict:
    """공지사항 질문에서 count 인자를 추론한다."""
    count_match = re.search(r"(\d+)\s*개", question)
    if count_match:
        count = min(max(int(count_match.group(1)), 1), 10)
        return {"count": count}
    return {}


register(
    Skill(
        name="get_notices",
        description="충남대학교 컴퓨터융합학부 최신 공지사항을 조회합니다.",
        keywords=["공지사항", "공지", "알림"],
        parameters={
            "count": {"type": "integer", "description": "가져올 공지 수(기본 5)."},
        },
        run=get_notices,
        infer_args=_infer_notice_args,
    )
)
