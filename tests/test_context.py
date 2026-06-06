"""수강신청 질문에 대해 모델이 context를 읽는지 테스트."""
import sys
sys.path.insert(0, ".")
from src.model.inference import load_model_with_lora, generate_answer

print("loading model...")
m, t = load_model_with_lora()
print("generating...")
a = generate_answer(
    "수강신청은 언제야?",
    "1학기 수강신청: 2026.02.02~02.06",
    [],
    m,
    t,
)
print(a)
