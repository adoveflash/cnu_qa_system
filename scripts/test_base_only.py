"""Base 모델(LoRA 없이) 추론 테스트."""
from src.rag.retriever import Retriever
from src.model.base import load_model, load_tokenizer
from src.model.inference import generate_answer

r = Retriever()
m = load_model()
t = load_tokenizer()

questions = [
    "이번 학기 수강신청은 언제야?",
    "교내 장학금 종류 알려줘",
    "인재개발원 취업 프로그램 뭐 있어?",
    "충남대 셔틀버스 시간표 알려줘",
    "컴퓨터융합학부 졸업요건 알려줘",
]

for q in questions:
    context, urls = r.build_context(q)
    ans = generate_answer(q, context, urls, m, t)
    print(f"Q: {q}")
    print(f"A: {ans}")
    print()
