"""QLoRA 학습 + 벡터 DB 재구축 스크립트.

SSH 서버에서 실행:
    python train.py

환경변수:
    HF_TOKEN: HuggingFace 토큰 (없으면 huggingface-cli login 필요)
"""

import gc
import json
import os
import random
import re
import shutil

import torch
from datasets import Dataset
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# ── 설정 ─────────────────────────────────────────────────────────────────────
SEED = 42
MODEL_NAME = "Qwen/Qwen3-8B"
HF_REPO = "adoveflash/cnu-qa-system"
LOCAL_OUTPUT = "models/lora_adapter"
CKPT_DIR = "checkpoints"
MAX_LENGTH = 768
NUM_EPOCHS = 5
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 2e-4

TRAIN_PATH = "data/qa/train_clean.jsonl"
EVAL_PATH = "data/qa/eval.jsonl"
CHUNKS_PATH = "data/corpus/chunks.jsonl"

SYSTEM_PROMPT = (
    "당신은 충남대학교 학내 정보 안내 도우미입니다.\n"
    "규칙:\n"
    "1. 반드시 주어진 참고 자료에 있는 정보만 사용하여 답변하세요.\n"
    "2. 참고 자료에 없는 내용은 추측하지 말고 '확인되지 않은 정보입니다'라고 답하세요.\n"
    "3. 답변은 간결하고 정확하게 작성하세요.\n"
    "4. 답변 끝에 출처 URL을 포함하세요."
)

torch.manual_seed(SEED)
random.seed(SEED)


# ── 1. 데이터 다운로드 ───────────────────────────────────────────────────────
print("=" * 60)
print("1. 데이터 다운로드")
print("=" * 60)

for path in [TRAIN_PATH, EVAL_PATH, CHUNKS_PATH]:
    if not os.path.exists(path):
        snapshot_download(repo_id=HF_REPO, local_dir=".", allow_patterns=[path])

with open(TRAIN_PATH) as f:
    train_data = [json.loads(line) for line in f if line.strip()]
with open(EVAL_PATH) as f:
    eval_data = [json.loads(line) for line in f if line.strip()]

chunks_map = {}
with open(CHUNKS_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            chunk = json.loads(line)
            chunks_map[chunk["chunk_id"]] = chunk["text"]

print(f"학습: {len(train_data)}건, 검증: {len(eval_data)}건, 청크: {len(chunks_map)}개")


# ── 2. 모델 & 토크나이저 로드 ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. 모델 & 토크나이저 로드")
print("=" * 60)

print("[1/2] 토크나이저 로드...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[2/2] 모델 로드 (fp16)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

if torch.cuda.is_available():
    vram_gb = torch.cuda.memory_reserved() / 1024**3
    print(f"모델 로드 완료 — VRAM: {vram_gb:.2f} GB")


# ── 3. LoRA 설정 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. LoRA 설정")
print("=" * 60)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ── 4. 데이터셋 준비 ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. 데이터셋 준비")
print("=" * 60)


def format_messages(qa, chunk_text=""):
    if chunk_text:
        user_content = f"참고 자료:\n{chunk_text}\n\n질문: {qa['question']}"
    else:
        user_content = qa["question"]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": qa["answer"]},
    ]


def find_assistant_start(messages):
    prompt_only = messages[:2]
    prompt_text = tokenizer.apply_chat_template(
        prompt_only, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    return len(prompt_ids)


def build_dataset(data_list):
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for qa in data_list:
        chunk_text = chunks_map.get(qa.get("chunk_id", ""), "")
        messages = format_messages(qa, chunk_text)

        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            add_special_tokens=False,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        assistant_start = find_assistant_start(messages)
        labels = [-100] * min(assistant_start, len(input_ids))
        labels += input_ids[len(labels) :]

        labels = [lb if am == 1 else -100 for lb, am in zip(labels, attention_mask)]
        if len(labels) < MAX_LENGTH:
            labels += [-100] * (MAX_LENGTH - len(labels))
        labels = labels[:MAX_LENGTH]

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)

    return Dataset.from_dict(
        {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }
    )


random.seed(SEED)
random.shuffle(train_data)

train_dataset = build_dataset(train_data)
eval_dataset = build_dataset(eval_data)
print(f"학습 데이터셋: {len(train_dataset)}건")
print(f"검증 데이터셋: {len(eval_dataset)}건")

sample_labels = train_dataset[0]["labels"]
loss_tokens = sum(1 for lb in sample_labels if lb != -100)
total_tokens = sum(1 for a in train_dataset[0]["attention_mask"] if a == 1)
print(f"샘플 — 전체 토큰: {total_tokens}, loss 토큰: {loss_tokens} ({loss_tokens/max(total_tokens,1)*100:.1f}%)")


# ── 5. 학습 ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. 학습")
print("=" * 60)


class HubBackupCallback(TrainerCallback):
    """에폭 종료 시 HF Hub에 백업."""

    def on_save(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch else 0
        print(f"\n에폭 {epoch} → HF Hub 백업 중...")
        try:
            api = HfApi()
            ckpts = sorted(
                [d for d in os.listdir(CKPT_DIR) if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[1]),
            )
            if ckpts:
                latest_path = os.path.join(CKPT_DIR, ckpts[-1])
                api.upload_folder(
                    folder_path=latest_path,
                    path_in_repo="models/lora_adapter",
                    repo_id=HF_REPO,
                )
                print(f"  백업 완료: {ckpts[-1]}")
        except Exception as e:
            print(f"  백업 실패 (학습은 계속): {e}")


training_args = TrainingArguments(
    output_dir=CKPT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.1,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    seed=SEED,
    fp16=True,
    report_to="none",
    optim="adamw_torch",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[HubBackupCallback()],
)

# 체크포인트에서 이어서 학습
resume_ckpt = None
if os.path.exists(CKPT_DIR):
    ckpts = [d for d in os.listdir(CKPT_DIR) if d.startswith("checkpoint-")]
    if ckpts:
        resume_ckpt = os.path.join(
            CKPT_DIR,
            sorted(ckpts, key=lambda x: int(x.split("-")[1]))[-1],
        )
        print(f"체크포인트에서 재개: {resume_ckpt}")

print("학습 시작!")
trainer.train(resume_from_checkpoint=resume_ckpt)
print("학습 완료!")

# 학습 로그 출력
for log in trainer.state.log_history:
    if "eval_loss" in log:
        print(
            f"에폭 {log.get('epoch', '?'):.0f} — "
            f"train_loss: {log.get('loss', 'N/A')}, eval_loss: {log['eval_loss']:.4f}"
        )


# ── 6. 어댑터 저장 & 업로드 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. 어댑터 저장 & 업로드")
print("=" * 60)

os.makedirs(LOCAL_OUTPUT, exist_ok=True)
model.save_pretrained(LOCAL_OUTPUT)
tokenizer.save_pretrained(LOCAL_OUTPUT)
print(f"로컬 저장 완료: {LOCAL_OUTPUT}")

for f in os.listdir(LOCAL_OUTPUT):
    size = os.path.getsize(os.path.join(LOCAL_OUTPUT, f))
    print(f"  {f}: {size / 1024**2:.1f} MB")

api = HfApi()
api.upload_folder(
    folder_path=LOCAL_OUTPUT,
    path_in_repo="models/lora_adapter",
    repo_id=HF_REPO,
)
print(f"HF Hub 업로드 완료: {HF_REPO}/models/lora_adapter")


# ── 7. 추론 테스트 ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. 추론 테스트")
print("=" * 60)

model.eval()
model.config.use_cache = True

_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)

test_questions = [
    "컴퓨터융합학부 졸업 요건이 어떻게 되나요?",
    "수강신청은 언제 하나요?",
    "오늘 학식 뭐 나와요?",
    "셔틀버스 시간표 알려주세요",
]

for q in test_questions:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    torch.manual_seed(SEED)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=192, do_sample=False, repetition_penalty=1.2
        )
    gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    answer = _THINK_RE.sub("", answer).strip()
    print(f"Q: {q}")
    print(f"A: {answer}")
    print("-" * 60)


# ── 8. 벡터 DB 재구축 ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. 벡터 DB 재구축")
print("=" * 60)

import chromadb
from sentence_transformers import SentenceTransformer

# LLM 메모리 해제
del model, trainer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"메모리 해제 후 VRAM: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

print("임베딩 모델 로드: BAAI/bge-m3")
embed_model = SentenceTransformer("BAAI/bge-m3")

chunks = []
with open(CHUNKS_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(json.loads(line))
print(f"청크 수: {len(chunks)}")

texts = [c["text"] for c in chunks]
print(f"{len(texts)}개 청크 임베딩 중...")
embeddings = embed_model.encode(texts, batch_size=32, show_progress_bar=True)
print(f"임베딩 완료: shape={embeddings.shape}")

DB_PATH = "data/vector_db"
os.makedirs(DB_PATH, exist_ok=True)
client = chromadb.PersistentClient(path=DB_PATH)

try:
    client.delete_collection("cnu_chunks")
except Exception:
    pass

collection = client.create_collection(
    name="cnu_chunks",
    metadata={"hnsw:space": "cosine"},
)

BATCH = 500
for i in range(0, len(chunks), BATCH):
    batch_chunks = chunks[i : i + BATCH]
    batch_embeds = embeddings[i : i + BATCH].tolist()
    collection.add(
        ids=[c["chunk_id"] for c in batch_chunks],
        embeddings=batch_embeds,
        documents=[c["text"] for c in batch_chunks],
        metadatas=[
            {"url": c["url"], "title": c["title"], "source": c["source"]} for c in batch_chunks
        ],
    )

print(f"벡터 DB 구축 완료: {collection.count()}개 인덱싱")

# 검색 테스트
test_q = "졸업 요건이 어떻게 되나요?"
q_emb = embed_model.encode([test_q]).tolist()[0]
results = collection.query(query_embeddings=[q_emb], n_results=3, include=["documents"])
print(f"\n검색 테스트: '{test_q}'")
for i, doc in enumerate(results["documents"][0]):
    print(f"  [{i + 1}] {doc[:80]}...")

# HF Hub 업로드
api = HfApi()
api.upload_folder(
    folder_path=DB_PATH,
    path_in_repo="data/vector_db",
    repo_id=HF_REPO,
)
print(f"벡터 DB 업로드 완료: {HF_REPO}/data/vector_db")

print("\n" + "=" * 60)
print("전체 완료!")
print("=" * 60)
