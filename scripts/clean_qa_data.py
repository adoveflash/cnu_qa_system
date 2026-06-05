"""QA 학습 데이터 정제 스크립트.

저품질 QA 쌍을 필터링한다:
- URL placeholder가 답변에 포함된 항목
- 30자 미만의 너무 짧은 답변
- "문의하시기 바랍니다" 류 회피 답변
"""

from __future__ import annotations

import json
from pathlib import Path

TRAIN_PATH = Path("data/qa/train.jsonl")
OUTPUT_PATH = Path("data/qa/train_clean.jsonl")

# 회피성 답변 패턴
_DEFLECTION_PATTERNS = [
    "문의하시기 바랍니다",
    "문의하세요",
    "확인하시기 바랍니다",
    "참고하시기 바랍니다",
    "참고하세요",
    "알 수 없습니다",
    "정보가 없습니다",
    "확인되지 않",
]

# URL placeholder 패턴
_URL_PLACEHOLDER_PATTERNS = [
    "[제공된 정보",
    "URL](",
    "[출처 URL]",
]

MIN_ANSWER_LENGTH = 30


def is_low_quality(qa: dict[str, str]) -> str | None:
    """저품질 여부를 판단한다. 저품질이면 사유를 반환, 아니면 None."""
    answer = qa.get("answer", "")

    # URL placeholder
    for pattern in _URL_PLACEHOLDER_PATTERNS:
        if pattern in answer:
            return f"url_placeholder: {pattern}"

    # 너무 짧은 답변
    if len(answer) < MIN_ANSWER_LENGTH:
        return f"too_short: {len(answer)}자"

    # 회피성 답변 (답변의 주요 내용이 회피인 경우만)
    for pattern in _DEFLECTION_PATTERNS:
        if pattern in answer:
            # 회피 패턴이 답변의 후반부에만 있으면 (실제 정보 + 부가 안내) 유지
            idx = answer.find(pattern)
            if idx < len(answer) * 0.5:
                return f"deflection: {pattern}"

    return None


def main() -> None:
    with open(TRAIN_PATH, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"원본 데이터: {len(data)}건")

    clean = []
    removed_reasons: dict[str, int] = {}

    for qa in data:
        reason = is_low_quality(qa)
        if reason:
            category = reason.split(":")[0]
            removed_reasons[category] = removed_reasons.get(category, 0) + 1
        else:
            clean.append(qa)

    print(f"정제 후: {len(clean)}건 (제거: {len(data) - len(clean)}건)")
    print("제거 사유:")
    for reason, count in sorted(removed_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}건")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for qa in clean:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"\n저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
