"""크롤링 실행 스크립트. python -m src.crawl.run_crawl 으로 실행한다."""

from datetime import datetime
from pathlib import Path

from .crawl_academic import crawl_academic
from .crawl_meal import crawl_meal
from .crawl_shuttle import crawl_shuttle
from .pdf_crawler import crawl_pdf
from .static_crawler import crawl_board_offset, crawl_static

try:
    from .js_crawler import crawl_js_boards

    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False
    print("[INFO] playwright 미설치 → JS 게시판 크롤링 건너뜀")

TODAY = datetime.now().strftime("%Y%m%d")
RAW_DIR = Path("data/corpus/raw")

# plus.cnu.ac.kr: robots.txt에서 User-agent:* Disallow:/ → 크롤링 불가
# (Googlebot 등 명시된 봇만 /html/ 허용, 본 User-Agent는 해당 없음)

# ── 1단계: BFS 정적 크롤링 (메뉴·안내 페이지) ───────────────────────────
HTML_TARGETS: list[tuple[str, str, str, int]] = [
    ("https://computer.cnu.ac.kr", "https://computer.cnu.ac.kr", "computer", 200),
    ("https://job.cnu.ac.kr", "https://job.cnu.ac.kr", "job", 200),
]

# ── 2단계: article.offset 게시판 크롤링 (computer - 정적 HTML로 가능) ────
# computer.cnu.ac.kr 게시판은 article.offset 파라미터로 목록 페이지네이션 가능
BOARD_OFFSET_TARGETS: list[dict] = [
    {
        "board_url": "https://computer.cnu.ac.kr/computer/notice/bachelor.do",
        "source_name": "computer",
        "output_path": RAW_DIR / f"computer_{TODAY}.jsonl",
        "max_articles": 300,
    },
    {
        "board_url": "https://computer.cnu.ac.kr/computer/notice/job.do",
        "source_name": "computer",
        "output_path": RAW_DIR / f"computer_{TODAY}.jsonl",
        "max_articles": 200,
    },
    {
        "board_url": "https://computer.cnu.ac.kr/computer/notice/notice.do",
        "source_name": "computer",
        "output_path": RAW_DIR / f"computer_{TODAY}.jsonl",
        "max_articles": 200,
    },
    {
        "board_url": "https://computer.cnu.ac.kr/computer/scholarship/list.do",
        "source_name": "computer",
        "output_path": RAW_DIR / f"computer_{TODAY}.jsonl",
        "max_articles": 200,
    },
]

# ── 3단계: JS 게시판 크롤링 (job - AJAX 동적 로딩) ───────────────────────
JS_BOARD_TARGETS: list[dict] = [
    {
        "board_url": "https://job.cnu.ac.kr/job/jobcenter/notice2.do",
        "path_prefix": "https://job.cnu.ac.kr",
        "source_name": "job",
        "output_path": RAW_DIR / f"job_{TODAY}.jsonl",
    },
    {
        "board_url": "https://job.cnu.ac.kr/job/community/notice3.do",
        "path_prefix": "https://job.cnu.ac.kr",
        "source_name": "job",
        "output_path": RAW_DIR / f"job_{TODAY}.jsonl",
    },
    {
        "board_url": "https://job.cnu.ac.kr/job/placement/placement01.do",
        "path_prefix": "https://job.cnu.ac.kr",
        "source_name": "job",
        "output_path": RAW_DIR / f"job_{TODAY}.jsonl",
    },
    {
        "board_url": "https://job.cnu.ac.kr/job/local/notice.do",
        "path_prefix": "https://job.cnu.ac.kr",
        "source_name": "job",
        "output_path": RAW_DIR / f"job_{TODAY}.jsonl",
    },
]

# ── 4단계: PDF ────────────────────────────────────────────────────────────
PDF_TARGETS: list[tuple[str, str]] = [
    ("https://sugang.cnu.ac.kr/login/data/2026_Sugang.pdf", "sugang_manual"),
]


def main() -> None:
    total = 0

    print("=" * 60)
    print("1단계: BFS 정적 HTML 크롤링")
    print("=" * 60)
    for seed, prefix, name, max_p in HTML_TARGETS:
        out = RAW_DIR / f"{name}_{TODAY}.jsonl"
        print(f"\n[{name}] {seed}  →  {out}")
        n = crawl_static(seed, prefix, out, max_pages=max_p, source_name=name)
        print(f"  저장 완료: {n}건")
        total += n

    print("\n" + "=" * 60)
    print("2단계: article.offset 게시판 크롤링 (computer)")
    print("=" * 60)
    for cfg in BOARD_OFFSET_TARGETS:
        board = cfg["board_url"].split("/")[-1].replace(".do", "")
        print(f"\n[computer/{board}] max={cfg['max_articles']}건")
        n = crawl_board_offset(
            board_url=cfg["board_url"],
            output_path=cfg["output_path"],
            source_name=cfg["source_name"],
            max_articles=cfg["max_articles"],
        )
        print(f"  저장 완료: {n}건")
        total += n

    print("\n" + "=" * 60)
    print("3단계: JS 게시판 크롤링 (job, playwright)")
    print("=" * 60)
    if _HAS_PLAYWRIGHT:
        n = crawl_js_boards(JS_BOARD_TARGETS, RAW_DIR, max_board_pages=30, max_articles=300)
        print(f"\nJS 게시판 총 저장: {n}건")
        total += n
    else:
        print("playwright 없음 → 건너뜀 (pip install playwright && playwright install chromium)")

    print("\n" + "=" * 60)
    print("4단계: PDF 크롤링")
    print("=" * 60)
    for url, name in PDF_TARGETS:
        out = RAW_DIR / f"{name}_{TODAY}.jsonl"
        print(f"\n[{name}] {url}  →  {out}")
        n = crawl_pdf(url, out, source_name=name)
        print(f"  저장 완료: {n}건")
        total += n

    print("\n" + "=" * 60)
    print("5단계: 학사일정/졸업요건 크롤링")
    print("=" * 60)
    out = RAW_DIR / f"academic_{TODAY}.jsonl"
    n = crawl_academic(out)
    print(f"  저장 완료: {n}건")
    total += n

    print("\n" + "=" * 60)
    print("6단계: 식단 크롤링 (plus, cnucoop, mobileadmin)")
    print("=" * 60)
    out = RAW_DIR / f"meal_{TODAY}.jsonl"
    n = crawl_meal(out)
    print(f"  저장 완료: {n}건")
    total += n

    print("\n" + "=" * 60)
    print("7단계: 셔틀버스/교통 크롤링 (plus)")
    print("=" * 60)
    out = RAW_DIR / f"shuttle_{TODAY}.jsonl"
    n = crawl_shuttle(out)
    print(f"  저장 완료: {n}건")
    total += n

    print("\n" + "=" * 60)
    print(f"총 저장 문서 수: {total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
