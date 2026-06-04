"""
Task 1 질문 분류 데이터 증강 스크립트.

기존 train.json의 질문을 Claude Haiku로 paraphrase하여
라벨당 2,000개 이상 (총 10,000+) 데이터를 생성한다.
"""

import json
import os
import random
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")
TRAIN_PATH = BASE_DIR / "data" / "train.json"
OUTPUT_PATH = BASE_DIR / "data" / "train_augmented.json"

LABEL_NAMES = {
    0: "졸업요건",
    1: "학교 공지사항",
    2: "학사일정",
    3: "식단 안내",
    4: "통학/셔틀 버스",
}

TARGET_PER_LABEL = 2200  # 라벨당 목표 (여유분 포함)

BATCH_SIZE = 20  # 한 번의 API 호출로 생성할 질문 수

SYSTEM_PROMPT = """\
너는 충남대학교 재학생들이 사용할 챗봇의 학습 데이터를 만드는 도우미야.
주어진 카테고리와 예시 질문들을 참고해서, 같은 카테고리에 속하는 새로운 질문을 생성해.

규칙:
1. 충남대학교 학생이 실제로 물어볼 법한 자연스러운 질문을 만들어
2. 다양한 말투를 사용해 (존댓말, 반말, 구어체, 문어체 등)
3. 같은 의미라도 다양한 표현과 어휘를 사용해
4. 질문의 길이도 짧은 것부터 긴 것까지 다양하게
5. 오타나 줄임말이 포함된 질문도 일부 섞어줘
6. 각 질문은 반드시 해당 카테고리에만 해당해야 해
7. 예시 질문과 완전히 동일한 질문은 만들지 마
8. 한 줄에 하나의 질문만 출력하고, 번호나 기호 없이 질문만 출력해"""


def load_train_data() -> dict[int, list[str]]:
    """라벨별 질문 목록을 반환한다."""
    with open(TRAIN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    label_questions: dict[int, list[str]] = {i: [] for i in range(5)}
    for item in data:
        label_questions[item["label"]].append(item["question"])
    return label_questions


def generate_questions(
    client: anthropic.Anthropic,
    label: int,
    examples: list[str],
    num_to_generate: int,
) -> list[str]:
    """Claude Haiku를 사용하여 질문을 생성한다."""
    sample_examples = random.sample(examples, min(15, len(examples)))
    examples_text = "\n".join(f"- {q}" for q in sample_examples)

    user_prompt = f"""카테고리: {LABEL_NAMES[label]}

예시 질문들:
{examples_text}

위 카테고리에 해당하는 새로운 질문을 {num_to_generate}개 만들어줘.
한 줄에 하나씩, 번호 없이 질문만 출력해."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=1.0,
    )

    text = response.content[0].text
    questions = []
    for line in text.strip().split("\n"):
        line = line.strip()
        # 번호, 기호 제거
        if line and len(line) > 3:
            # "1. ", "- ", "• " 등 제거
            for prefix in ["- ", "• ", "· "]:
                if line.startswith(prefix):
                    line = line[len(prefix):]
                    break
            if line[0].isdigit() and ("." in line[:4] or ")" in line[:4]):
                line = line.split(".", 1)[-1].split(")", 1)[-1].strip()
            if line and len(line) > 3:
                questions.append(line)
    return questions


def deduplicate(questions: list[str]) -> list[str]:
    """정규화 후 중복 제거."""
    seen: set[str] = set()
    result = []
    for q in questions:
        normalized = q.strip().lower().replace(" ", "")
        if normalized not in seen:
            seen.add(normalized)
            result.append(q)
    return result


def main() -> None:
    client = anthropic.Anthropic()
    label_questions = load_train_data()
    all_augmented: list[dict] = []

    # 기존 데이터를 먼저 포함
    with open(TRAIN_PATH, encoding="utf-8") as f:
        original_data = json.load(f)
    all_augmented.extend(original_data)

    for label in range(5):
        existing = label_questions[label]
        existing_count = len(existing)
        needed = TARGET_PER_LABEL - existing_count
        print(f"\n[라벨 {label}: {LABEL_NAMES[label]}] 기존 {existing_count}개, {needed}개 생성 필요")

        generated: list[str] = []
        call_count = 0

        while len(generated) < needed:
            remaining = needed - len(generated)
            batch = min(BATCH_SIZE, remaining)
            try:
                new_questions = generate_questions(client, label, existing, batch)
                generated.extend(new_questions)
                call_count += 1
                print(f"  API 호출 #{call_count}: {len(new_questions)}개 생성 (누적 {len(generated)}/{needed})")

                # rate limit 방지
                time.sleep(0.3)
            except anthropic.RateLimitError:
                print("  Rate limit 도달, 30초 대기...")
                time.sleep(30)
            except Exception as e:
                print(f"  오류 발생: {e}, 5초 후 재시도")
                time.sleep(5)

        # 중복 제거 (기존 질문 포함하여)
        all_for_label = existing + generated
        deduped = deduplicate(all_for_label)
        new_only = deduped[existing_count:]  # 기존 질문은 이미 all_augmented에 포함됨

        for q in new_only:
            all_augmented.append({"question": q, "label": label})

        print(f"  [완료] 중복 제거 후 신규 {len(new_only)}개 추가 (라벨 총 {existing_count + len(new_only)}개)")

    # 셔플
    random.shuffle(all_augmented)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_augmented, f, ensure_ascii=False, indent=2)

    # 통계 출력
    from collections import Counter
    label_counts = Counter(item["label"] for item in all_augmented)
    print(f"\n{'='*50}")
    print(f"총 데이터: {len(all_augmented)}개")
    for label in sorted(label_counts):
        print(f"  라벨 {label} ({LABEL_NAMES[label]}): {label_counts[label]}개")
    print(f"저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
