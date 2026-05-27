"""크롤링 원본 데이터 정제 모듈.

로그인 페이지 필터링, 보일러플레이트 제거, 중복 제거를 수행한다.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

# 네비게이션 헤더 줄 (정확 매칭으로 제거)
_NAV_HEADER_LINES = frozenset(
    {
        "본문 바로가기",
        "주메뉴 바로가기",
        "서브메뉴 바로가기",
    }
)

# 페이지네이션/푸터에 등장하는 키워드
_PAGINATION_KEYWORDS = frozenset(
    {
        "다음 페이지로 이동하기",
        "이전 페이지로 이동하기",
        "마지막 페이지로 이동하기",
        "처음 페이지로 이동하기",
        "전체목록",
    }
)

# 사이드바 메뉴 감지용 키워드 (이 중 2개 이상 포함된 짧은 줄 블록은 메뉴)
_MENU_KEYWORDS = frozenset(
    {
        "Home",
        "공지사항",
        "더보기",
        "학사공지",
        "교내 일반소식",
        "교외 활동.인턴.취업",
        "사업단소식",
        "우리학부 News",
        "교과과정",
        "교과목게시판",
        "입학안내",
        "처음으로",
        "인재개발원 소개",
        "진로·취업",
        "현장실습",
        "커뮤니티",
        "주요업무",
    }
)


def load_raw_documents(raw_dir: Path) -> list[dict[str, Any]]:
    """data/corpus/raw/ 아래의 모든 .jsonl 파일을 읽어 리스트로 반환한다.

    Args:
        raw_dir: raw JSONL 파일이 있는 디렉터리

    Returns:
        파싱된 문서 리스트 (url, title, content, crawled_at, source 키)
    """
    docs: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
    return docs


def is_login_page(content: str) -> bool:
    """로그인 리다이렉트 페이지인지 판별한다.

    Args:
        content: 문서 본문 텍스트

    Returns:
        로그인 페이지이면 True
    """
    # 로그인 폼 텍스트가 본문 앞부분에 있고 전체가 짧으면 로그인 페이지
    markers = ["로그인\n아이디저장", "아이디저장\n아이디\n비밀번호"]
    return any(m in content for m in markers) and len(content) < 800


def remove_boilerplate(content: str) -> str:
    """네비게이션 헤더, 사이드바 메뉴, 페이지네이션 등 보일러플레이트를 제거한다.

    Args:
        content: 원본 텍스트

    Returns:
        정제된 텍스트
    """
    lines = content.split("\n")
    result: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # 1) 네비게이션 헤더 줄 제거
        if line in _NAV_HEADER_LINES:
            i += 1
            continue

        # 2) 페이지네이션 키워드 줄 제거
        if line in _PAGINATION_KEYWORDS:
            i += 1
            continue

        # 3) 사이드바 메뉴 블록 감지 및 제거
        # 짧은 줄(15자 이하)이 5개 이상 연속되고, 메뉴 키워드가 2개 이상 포함
        if len(line) <= 15 and i + 4 < len(lines):
            block_end = i
            menu_kw_count = 0
            while block_end < len(lines) and len(lines[block_end].strip()) <= 15:
                if lines[block_end].strip() in _MENU_KEYWORDS:
                    menu_kw_count += 1
                block_end += 1
            block_len = block_end - i
            if block_len >= 5 and menu_kw_count >= 2:
                i = block_end
                continue

        # 4) 이전글/다음글 네비게이션 제거
        if line == "목록" and i + 1 < len(lines) and lines[i + 1].strip() in ("이전글", "다음글"):
            # 목록~끝까지 제거 (보통 문서 마지막)
            break

        # 5) 순수 숫자 페이지네이션 줄 제거 (1, 2, 3, ... 10 같은 패턴)
        if re.match(r"^\d+$", line) and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if re.match(r"^\d+$", next_line) or next_line in _PAGINATION_KEYWORDS:
                # 연속 숫자 페이지네이션 블록 스킵
                while i < len(lines):
                    stripped = lines[i].strip()
                    if re.match(r"^\d+$", stripped) or stripped in _PAGINATION_KEYWORDS:
                        i += 1
                    else:
                        break
                continue

        result.append(lines[i])
        i += 1

    text = "\n".join(result)
    # 3개 이상 연속 빈 줄을 2개로 축소
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def deduplicate(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """content 해시 기준으로 중복 문서를 제거한다.

    Args:
        docs: 문서 리스트

    Returns:
        중복 제거된 리스트 (먼저 나온 문서 유지)
    """
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for doc in docs:
        h = hashlib.md5(doc["content"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    return unique


def clean_all(
    raw_dir: Path,
    output_path: Path,
    min_content_length: int = 80,
) -> int:
    """전체 클리닝 파이프라인: load → filter login → boilerplate 제거 → short 필터 → dedup → save.

    Args:
        raw_dir: raw 디렉터리 경로
        output_path: 출력 JSONL 경로
        min_content_length: 보일러플레이트 제거 후 최소 글자 수

    Returns:
        저장된 문서 수
    """
    docs = load_raw_documents(raw_dir)
    total = len(docs)
    print(f"  로드: {total}건")

    # 로그인 페이지 필터
    docs = [d for d in docs if not is_login_page(d["content"])]
    print(f"  로그인 페이지 제거 후: {len(docs)}건")

    # 보일러플레이트 제거
    for doc in docs:
        doc["content"] = remove_boilerplate(doc["content"])

    # 짧은 문서 필터
    docs = [d for d in docs if len(d["content"]) >= min_content_length]
    print(f"  짧은 문서 제거 후: {len(docs)}건")

    # 중복 제거
    docs = deduplicate(docs)
    print(f"  중복 제거 후: {len(docs)}건")

    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return len(docs)
