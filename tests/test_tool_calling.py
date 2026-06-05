"""Tool calling diagnostic script.

Run on SSH: python tests/test_tool_calling.py
"""
import sys
sys.path.insert(0, ".")

print("1. importing modules...")
from src.model.inference import (
    load_model_with_lora,
    _generate_once,
    _parse_tool_calls,
    _SYSTEM_PROMPT,
)
from src.tools.definitions import TOOLS

print("2. loading model + LoRA adapter...")
model, tokenizer = load_model_with_lora()
print(f"   model device: {model.device}")
print(f"   model type: {type(model).__name__}")

# --- Test 1: with tools, food question ---
print("\n=== TEST 1: food question WITH tools ===")
messages1 = [
    {"role": "system", "content": _SYSTEM_PROMPT},
    {"role": "user", "content": "today meal menu?"},
]
raw1 = _generate_once(messages1, model, tokenizer, tools=TOOLS)
print(f"RAW OUTPUT:\n{raw1}")
print(f"TOOL CALLS: {_parse_tool_calls(raw1)}")

# --- Test 2: check what the template looks like ---
print("\n=== TEST 2: template check ===")
template_text = tokenizer.apply_chat_template(
    messages1, tokenize=False, add_generation_prompt=True, enable_thinking=False, tools=TOOLS
)
# Print last 500 chars to see tool definitions
print(f"TEMPLATE (last 500 chars):\n{template_text[-500:]}")

# --- Test 3: without LoRA ---
print("\n=== TEST 3: food question WITHOUT LoRA (base only) ===")
from src.model.base import load_model, load_tokenizer
base_model = load_model("Qwen/Qwen3-8B")
base_tokenizer = load_tokenizer("Qwen/Qwen3-8B")
base_model.eval()

messages3 = [
    {"role": "system", "content": _SYSTEM_PROMPT},
    {"role": "user", "content": "today meal menu?"},
]
raw3 = _generate_once(messages3, base_model, base_tokenizer, tools=TOOLS)
print(f"RAW OUTPUT:\n{raw3}")
print(f"TOOL CALLS: {_parse_tool_calls(raw3)}")

print("\n=== DONE ===")
