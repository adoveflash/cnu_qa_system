"""청크 데이터 전처리 — 노이즈 제거 + 의미없는 청크 필터링."""
import json
import re
from pathlib import Path

CHUNKS_PATH = Path("data/corpus/chunks.jsonl")
OUTPUT_PATH = Path("data/corpus/chunks.jsonl")  # 덮어쓰기

# 삭제할 보일러플레이트 라인 패턴
BOILERPLATE_LINES = [
    "사이드메뉴 바로가기",
    "주요메뉴 바로가기",
    "하단메뉴 바로가기",
    "바로가기 메뉴",
    "The Strong CNU",
    "코러스시스템",
    "웹메일",
    "CNU포털",
    "SNS 및 프린트",
    "URL복사",
    "facebook",
    "카카오톡",
    "Nav",
    "페이지 관리자",
    "관리자메일",
    "프린트",
    "확대",
    "축소",
]

# 정규식으로 제거할 패턴
REGEX_PATTERNS = [
    r"^\s*홈\s*$",
    r"^\s*>\s*$",
    r"^\s*SNS\s*$",
    r"^페이지 관리자\s*\|.*$",
    r"^관리자메일\s*$",
]

# 통째로 필터링할 청크 조건
MIN_MEANINGFUL_LENGTH = 50  # 정제 후 50자 미만이면 삭제


def clean_text(text: str) -> str:
    """청크 텍스트에서 노이즈를 제거한다."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # 보일러플레이트 라인 스킵
        if stripped in BOILERPLATE_LINES:
            continue
        # 빈 줄 연속 3개 이상 방지
        if not stripped and cleaned and not cleaned[-1].strip():
            continue
        # 정규식 패턴 매치 스킵
        skip = False
        for pattern in REGEX_PATTERNS:
            if re.match(pattern, stripped):
                skip = True
                break
        if skip:
            continue
        cleaned.append(line)

    result = "\n".join(cleaned).strip()
    # 연속 빈줄 정리
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def is_meaningful(chunk: dict) -> bool:
    """청크가 의미 있는 정보를 담고 있는지 판단한다."""
    text = chunk["text"]
    # 너무 짧으면 제거
    if len(text) < MIN_MEANINGFUL_LENGTH:
        return False
    # 대학원 목록만 나열된 청크 제거
    grad_schools = ["대학원", "경영대학원", "교육대학원", "행정대학원", "보건대학원"]
    grad_count = sum(1 for gs in grad_schools if gs in text)
    words = text.split()
    if grad_count >= 3 and len(words) < 30:
        return False
    return True


def main():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    print(f"원본 청크: {len(chunks)}건")

    # 1단계: 텍스트 정제
    for chunk in chunks:
        chunk["text"] = clean_text(chunk["text"])

    # 2단계: 의미없는 청크 필터링
    filtered = [c for c in chunks if is_meaningful(c)]
    removed = len(chunks) - len(filtered)
    print(f"제거된 청크: {removed}건")
    print(f"최종 청크: {len(filtered)}건")

    # 3단계: token_count 재계산 (대략적 — 한글 기준 글자수/2)
    for chunk in filtered:
        chunk["token_count"] = len(chunk["text"]) // 2

    # 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in filtered:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
