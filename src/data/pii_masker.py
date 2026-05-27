"""개인정보(PII) 마스킹 모듈.

학번, 전화번호, 이메일, 주민번호 등을 마스킹 토큰으로 치환한다.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import NamedTuple


class PIIPattern(NamedTuple):
    """PII 정규식 패턴 정의."""

    name: str
    pattern: re.Pattern[str]
    replacement: str


# 적용 순서가 중요: 주민번호 → 학번 → 전화번호 → 이메일
PII_PATTERNS: list[PIIPattern] = [
    PIIPattern(
        name="주민번호",
        pattern=re.compile(r"\b\d{6}\s*[-–]\s*[1-4]\d{6}\b"),
        replacement="[주민번호]",
    ),
    PIIPattern(
        name="학번",
        # 20XX로 시작하는 8~10자리 숫자 (CNU 학번 형식)
        pattern=re.compile(r"\b20[0-3]\d{5,7}\b"),
        replacement="[학번]",
    ),
    PIIPattern(
        name="전화번호",
        pattern=re.compile(r"0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}"),
        replacement="[전화번호]",
    ),
    PIIPattern(
        name="이메일",
        pattern=re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
        replacement="[이메일]",
    ),
]


def _get_literal_pii() -> dict[str, str]:
    """환경변수에서 리터럴 PII 매칭 대상을 로드한다.

    환경변수 MY_NAME이 설정되어 있으면 해당 이름을 [이름]으로 마스킹한다.

    Returns:
        {원본 문자열: 마스킹 토큰} 딕셔너리
    """
    literals: dict[str, str] = {}
    my_name = os.environ.get("MY_NAME", "")
    if my_name:
        literals[my_name] = "[이름]"
    return literals


def mask_pii(text: str) -> tuple[str, int]:
    """텍스트에서 개인정보를 마스킹 토큰으로 치환한다.

    Args:
        text: 입력 텍스트

    Returns:
        (마스킹된 텍스트, 치환 횟수) 튜플
    """
    total_count = 0

    # 정규식 패턴 적용
    for pii in PII_PATTERNS:
        text, count = pii.pattern.subn(pii.replacement, text)
        total_count += count

    # 리터럴 매칭
    for literal, replacement in _get_literal_pii().items():
        count = text.count(literal)
        if count > 0:
            text = text.replace(literal, replacement)
            total_count += count

    return text, total_count


def mask_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    """JSONL 파일을 읽어 content와 title에 PII 마스킹을 적용한 후 저장한다.

    Args:
        input_path: 입력 JSONL 경로
        output_path: 출력 JSONL 경로 (input_path와 같으면 in-place)

    Returns:
        (처리 문서 수, 총 마스킹 치환 수) 튜플
    """
    docs: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    total_replacements = 0
    for doc in docs:
        doc["content"], count = mask_pii(doc["content"])
        total_replacements += count
        doc["title"], count = mask_pii(doc["title"])
        total_replacements += count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return len(docs), total_replacements
