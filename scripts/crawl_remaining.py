"""4~7단계만 실행하는 스크립트. python scripts/crawl_remaining.py"""

from src.crawl.run_crawl import PDF_TARGETS, RAW_DIR, TODAY
from src.crawl.crawl_academic import crawl_academic
from src.crawl.crawl_meal import crawl_meal
from src.crawl.crawl_shuttle import crawl_shuttle
from src.crawl.pdf_crawler import crawl_pdf

print("4단계: PDF")
for url, name in PDF_TARGETS:
    out = RAW_DIR / f"{name}_{TODAY}.jsonl"
    print(f"  [{name}] {url}")
    n = crawl_pdf(url, out, source_name=name)
    print(f"  저장: {n}건")

print("\n5단계: 학사일정")
n = crawl_academic(RAW_DIR / f"academic_{TODAY}.jsonl")
print(f"  저장: {n}건")

print("\n6단계: 식단")
n = crawl_meal(RAW_DIR / f"meal_{TODAY}.jsonl")
print(f"  저장: {n}건")

print("\n7단계: 셔틀")
n = crawl_shuttle(RAW_DIR / f"shuttle_{TODAY}.jsonl")
print(f"  저장: {n}건")

print("\n완료!")
