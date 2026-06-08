"""주요 카테고리별 빠른 답변 테스트.

Run on SSH: python tests/test_quick.py
"""

import sys

sys.path.insert(0, ".")

from src.model.inference import load_model_with_lora, generate_answer

print("loading model...")
model, tokenizer = load_model_with_lora()

questions = [
    ("졸업요건", "컴퓨터융합학부 졸업 요건이 어떻게 되나요?"),
    ("수강신청", "이번 학기 수강신청은 언제야?"),
    ("셔틀버스", "충남대 셔틀버스 시간표 알려줘"),
    ("식단(tool)", "오늘 기숙사 점심 메뉴 뭐야?"),
    ("학사일정(tool)", "이번 달 학사일정 알려줘"),
    ("공지사항(tool)", "최근 공지사항 3개 보여줘"),
    ("장학금", "교내 장학금 종류가 뭐가 있어?"),
    ("인사", "안녕하세요"),
]

for label, q in questions:
    print(f"\n{'=' * 60}")
    print(f"[{label}] {q}")
    print("-" * 60)
    answer = generate_answer(q, "", [], model, tokenizer, use_tools=True)
    print(answer)

print(f"\n{'=' * 60}")
print("DONE")
