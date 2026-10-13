import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "messages.csv"
OUTPUT_DIR = ROOT / "models" / "bert_multilabel"

MODEL_NAME = "klue/bert-base"
LABELS = ["배송", "환불", "계정"]
MAX_LENGTH = 128
SEED = 42


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def sigmoid(logits):
    logits = np.clip(logits, -50, 50)
    return 1.0 / (1.0 + np.exp(-logits))


def compute_metrics(eval_pred):
    logits = eval_pred.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    probs = sigmoid(logits)
    preds = (probs >= 0.5).astype(int)

    return {
        "micro_f1": f1_score(
            eval_pred.label_ids,
            preds,
            average="micro",
            zero_division=0,
        )
    }


def main():
    set_seed(SEED)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "labels"])
    df = df.drop_duplicates(subset=["text"])
    df = df.reset_index(drop=True)

    X = df["text"].astype(str).tolist()

    label_lists = [
        [s.strip() for s in value.split("|")] for value in df["labels"].astype(str)
    ]

    mlb = MultiLabelBinarizer(classes=LABELS)

    Y = mlb.fit_transform(label_lists).astype(np.float32)  # type: ignore

    # 3. 기존 TF-IDF와 동일한 분할
    indices = np.arange(len(X))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=SEED,
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=SEED,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = TextDataset(
        [X[i] for i in train_idx],
        Y[train_idx],
        tokenizer,
    )

    val_ds = TextDataset(
        [X[i] for i in val_idx],
        Y[val_idx],
        tokenizer,
    )

    test_ds = TextDataset(
        [X[i] for i in test_idx],
        Y[test_idx],
        tokenizer,
    )

    id2label = {i: name for i, name in enumerate(LABELS)}
    label2id = {name: i for i, name in enumerate(LABELS)}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    args = TrainingArguments(
        output_dir=str(ROOT / "models" / "bert_checkpoints"),
        num_train_epochs=5,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        warmup_steps=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        save_total_limit=1,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    val_output = trainer.predict(val_ds)
    val_probs = sigmoid(val_output.predictions)

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.arange(0.20, 0.81, 0.05):
        pred = (val_probs >= threshold).astype(int)

        score = f1_score(
            Y[val_idx],
            pred,
            average="micro",
            zero_division=0,
        )

        if score > best_f1 or (
            np.isclose(score, best_f1)
            and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_f1 = score
            best_threshold = float(threshold)

    print(f"Best threshold: {best_threshold:.2f}")

    output = trainer.predict(test_ds)

    probs = sigmoid(output.predictions)
    Y_pred = (probs >= best_threshold).astype(int)
    Y_test = Y[test_idx].astype(int)

    print("\n=== BERT Test Results ===")

    print("Exact Match:", accuracy_score(Y_test, Y_pred))

    print(
        "Micro F1:",
        f1_score(
            Y_test,
            Y_pred,
            average="micro",
            zero_division=0,
        ),
    )

    print(
        "Macro F1:",
        f1_score(
            Y_test,
            Y_pred,
            average="macro",
            zero_division=0,
        ),
    )

    print(
        "Hamming Loss:",
        hamming_loss(Y_test, Y_pred),
    )

    print(
        classification_report(
            Y_test,
            Y_pred,
            target_names=LABELS,
            zero_division=0,
        )
    )

    # 10. 모델과 설정 저장
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    config = {
        "labels": LABELS,
        "threshold": best_threshold,
        "max_length": MAX_LENGTH,
        "base_model": MODEL_NAME,
    }

    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
