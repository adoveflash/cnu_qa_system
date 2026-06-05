"""Tool calling diagnostic script.

Run on SSH: python tests/test_tool_calling.py
"""
import sys
sys.path.insert(0, ".")

print("1. importing modules...")
from src.model.inference import (
    load_model_with_lora,
    generate_answer,
    _SYSTEM_PROMPT,
)

print("2. loading model + LoRA adapter...")
model, tokenizer = load_model_with_lora()
print(f"   model device: {model.device}")

# --- Test 1: food question (should trigger tool call) ---
print("\n=== TEST 1: food question (should call get_meal_menu) ===")
answer1 = generate_answer(
    question="today meal menu?",
    context="",
    urls=[],
    model=model,
    tokenizer=tokenizer,
    use_tools=True,
)
print(f"ANSWER:\n{answer1}")

# --- Test 2: general question (should NOT trigger tool call) ---
print("\n=== TEST 2: general question (no tool needed) ===")
answer2 = generate_answer(
    question="graduation credits?",
    context="[ref1] total 130 credits needed for graduation",
    urls=["https://computer.cnu.ac.kr"],
    model=model,
    tokenizer=tokenizer,
    use_tools=True,
)
print(f"ANSWER:\n{answer2}")

print("\n=== DONE ===")
