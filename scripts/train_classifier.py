"""Task 1 — BERT 분류기 학습 스크립트.

klue/bert-base를 5개 카테고리로 파인튜닝한다.
SSH 서버 또는 Colab에서 실행 가능.

사용법: PYTHONPATH=. python scripts/train_classifier.py
"""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, classification_report

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODEL_NAME = "klue/bert-base"
NUM_LABELS = 5
MAX_LENGTH = 128
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 2e-5

TRAIN_PATH = "data/train_augmented.json"
VALID_PATH = "data/valid.json"
TEST_PATH = "data/test_cls.json"
OUTPUT_DIR = "outputs"
MODEL_SAVE_DIR = "model/classifier"

LABEL_NAMES = ["졸업요건", "학교 공지사항", "학사일정", "식단 안내", "통학/셔틀 버스"]


class QuestionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["question"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        result = {k: v.squeeze(0) for k, v in encoding.items()}
        if "label" in item:
            result["labels"] = torch.tensor(item["label"], dtype=torch.long)
        return result


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    f1_macro = f1_score(eval_pred.label_ids, predictions, average="macro")
    f1_weighted = f1_score(eval_pred.label_ids, predictions, average="weighted")
    return {"f1_macro": f1_macro, "f1_weighted": f1_weighted}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # 데이터 로드
    with open(TRAIN_PATH, encoding="utf-8") as f:
        train_data = json.load(f)
    with open(VALID_PATH, encoding="utf-8") as f:
        valid_data = json.load(f)
    print(f"학습: {len(train_data)}건, 검증: {len(valid_data)}건")

    # 토크나이저 & 데이터셋
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = QuestionDataset(train_data, tokenizer)
    valid_dataset = QuestionDataset(valid_data, tokenizer)

    # 모델
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    print(f"파라미터: {sum(p.numel() for p in model.parameters()):,}")

    # 학습
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=10,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    print("학습 시작...")
    trainer.train()
    print("학습 완료!")

    # 검증 평가
    eval_results = trainer.evaluate()
    print(f"\nF1 (macro): {eval_results['eval_f1_macro']:.4f}")
    print(f"F1 (weighted): {eval_results['eval_f1_weighted']:.4f}")

    preds = trainer.predict(valid_dataset)
    pred_labels = np.argmax(preds.predictions, axis=-1)
    true_labels = np.array([d["label"] for d in valid_data])
    print("\n" + classification_report(true_labels, pred_labels, target_names=LABEL_NAMES, digits=4))

    # 모델 저장
    trainer.save_model(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    print(f"모델 저장: {MODEL_SAVE_DIR}")

    # 테스트셋 추론
    if os.path.exists(TEST_PATH):
        with open(TEST_PATH, encoding="utf-8") as f:
            test_data = json.load(f)

        cls_model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_DIR)
        cls_tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls_model.to(device)
        cls_model.eval()

        questions = [d["question"] for d in test_data]
        all_preds = []
        for i in range(0, len(questions), BATCH_SIZE):
            batch = questions[i : i + BATCH_SIZE]
            enc = cls_tokenizer(batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = cls_model(**enc)
                all_preds.extend(torch.argmax(out.logits, dim=-1).cpu().tolist())

        cls_output = [{"question": q, "label": l} for q, l in zip(questions, all_preds)]
        output_path = os.path.join(OUTPUT_DIR, "cls_output.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cls_output, f, ensure_ascii=False, indent=2)
        print(f"\ncls_output.json 저장: {output_path} ({len(cls_output)}건)")
    else:
        print(f"\n{TEST_PATH} 없음 — 테스트 추론 건너뜀")


if __name__ == "__main__":
    main()
