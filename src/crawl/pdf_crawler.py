"""PDF 다운로드 및 텍스트 추출 (pdfplumber)."""

import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pdfplumber
import requests

from .robots_checker import USER_AGENT, is_allowed

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})

CRAWL_DELAY = 3.0


def crawl_pdf(
    pdf_url: str,
    output_path: Path,
    source_name: str | None = None,
) -> int:
    """PDF URL을 다운로드해 페이지별 텍스트를 추출하고 JSONL로 저장한다.

    Args:
        pdf_url: 다운로드할 PDF URL
        output_path: 결과를 저장할 .jsonl 파일 경로
        source_name: JSONL source 필드에 사용할 이름

    Returns:
        저장된 레코드 수 (정상이면 1)
    """
    if not is_allowed(pdf_url):
        print(f"  [SKIP robots.txt] {pdf_url}")
        return 0

    time.sleep(CRAWL_DELAY)
    try:
        resp = SESSION.get(pdf_url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [ERROR] {pdf_url}: {e}")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    pages_text: list[str] = []
    try:
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    pages_text.append(text)
    except Exception as e:
        print(f"  [PDF ERROR] {pdf_url}: {e}")
        return 0
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    full_text = "\n\n".join(pages_text)
    if not full_text.strip():
        print(f"  [EMPTY PDF] {pdf_url}")
        return 0

    # 파일명을 제목으로 사용
    filename = Path(urlparse(pdf_url).path).stem

    record = {
        "url": pdf_url,
        "title": filename,
        "content": full_text,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "source": source_name or urlparse(pdf_url).netloc,
    }

    with output_path.open("w", encoding="utf-8") as fout:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  [PDF OK] {filename} — {len(full_text)} chars, {len(pages_text)} pages")
    return 1
