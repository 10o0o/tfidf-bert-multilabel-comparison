import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parent


def metrics(y_true, y_pred):
    return {
        "Exact Match": accuracy_score(y_true, y_pred),
        "Micro F1": f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "Macro F1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "Hamming Loss": hamming_loss(y_true, y_pred),
    }


def main():
    df = pd.read_csv(ROOT / "data/messages.csv")
    df = df.dropna(subset=["text", "labels"])
    df = df.drop_duplicates(subset=["text"])
    df = df.reset_index(drop=True)

    texts = df["text"].astype(str).tolist()
    label_lists = [
        [label.strip() for label in value.split("|")]
        for value in df["labels"].astype(str)
    ]

    # 동일한 테스트 데이터 복원
    indices = np.arange(len(texts))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
    )

    # 기존 TF-IDF 모델 로드
    tfidf_artifact = joblib.load(ROOT / "models" / "multilabel.joblib")

    mlb = tfidf_artifact["mlb"]

    X_test = [texts[i] for i in test_idx]
    Y_test = mlb.transform(label_lists)[test_idx]

    tfidf_probs = tfidf_artifact["pipeline"].predict_proba(X_test)

    tfidf_pred = (tfidf_probs >= tfidf_artifact["threshold"]).astype(int)

    # BERT 모델 로드
    model_dir = ROOT / "models" / "bert_multilabel"

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))

    assert list(mlb.classes_) == metadata["labels"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    bert = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # 평가 지표 비교는 CPU에서도 가능
    bert.eval()

    encoded = tokenizer(
        X_test,
        truncation=True,
        padding=True,
        max_length=metadata["max_length"],
        return_tensors="pt",
    )

    with torch.inference_mode():
        logits = bert(**encoded).logits
        bert_probs = torch.sigmoid(logits).numpy()

    bert_pred = (bert_probs >= metadata["threshold"]).astype(int)

    # 동일한 지표로 비교
    results = pd.DataFrame(
        {
            "TF-IDF": metrics(Y_test, tfidf_pred),
            "BERT": metrics(Y_test, bert_pred),
        }
    )

    print("\n=== 모델 성능 비교 ===")
    print(results.round(4).to_string())

    print("\n=== 두 모델의 예측이 다른 문장 ===")

    for text, actual, tf_pred, b_pred in zip(X_test, Y_test, tfidf_pred, bert_pred):
        if np.array_equal(tf_pred, b_pred):
            continue

        print(f"\n문장: {text}")
        print("정답:", mlb.inverse_transform(actual.reshape(1, -1)))
        print("TF-IDF:", mlb.inverse_transform(tf_pred.reshape(1, -1)))
        print("BERT:", mlb.inverse_transform(b_pred.reshape(1, -1)))


if __name__ == "__main__":
    main()
