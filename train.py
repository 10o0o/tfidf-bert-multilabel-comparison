from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "messages.csv"
MODEL_PATH = ROOT / "models" / "multilabel.joblib"

LABELS = ["배송", "환불", "계정"]


def main():
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["text", "labels"])
    df = df.drop_duplicates(subset=["text"])
    df = df.reset_index(drop=True)

    X = df["text"].astype(str).tolist()

    label_lists = [
        [label.strip() for label in value.split("|")]
        for value in df["labels"].astype(str)
    ]

    unknown = {
        label for labels in label_lists for label in labels if label not in LABELS
    }

    if unknown:
        raise ValueError(f"알 수 없는 라벨: {unknown}")

    mlb = MultiLabelBinarizer(classes=LABELS)
    Y = mlb.fit_transform(label_lists)
    assert isinstance(Y, np.ndarray)

    # print("클래스:", mlb.classes_)
    # print("정답 행렬 크기:", Y.shape)
    # print("처음 5개 정답:\n", Y[:5])

    indices = np.arange(len(X))

    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_STATE
    )

    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.25, random_state=RANDOM_STATE
    )

    X_train = [X[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    X_test = [X[i] for i in test_idx]

    Y_train = Y[train_idx]
    Y_val = Y[val_idx]
    Y_test = Y[test_idx]

    # print(f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    # print(f"mlb.classes_: {mlb.classes_}")

    for i, label in enumerate(mlb.classes_):
        values = np.unique(Y_train[:, i])
        if len(values) < 2:
            raise ValueError(f"{label} 라벨에 양성/음성 샘플이 모두 필요합니다.")

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char", ngram_range=(2, 4), min_df=1, sublinear_tf=True
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
                ),
            ),
        ]
    )

    model.fit(X_train, Y_train)

    val_probs = model.predict_proba(X_val)
    candidates = []

    for threshold in np.arange(0.20, 0.81, 0.05):
        preds = (val_probs >= threshold).astype(int)

        score = f1_score(Y_val, preds, average="micro", zero_division=0)

        candidates.append((float(threshold), score))

        print(f"threshold={threshold:.2f} micro_f1={score:.4f}")
    # print(f"candidates: {candidates}")

    best_threshold, best_score = max(
        candidates, key=lambda item: (item[1], -abs(item[0] - 0.5))
    )

    print(f"\n선택한 임계값: {best_threshold:.2f} (Validation F1={best_score:.4f})")

    test_probs = model.predict_proba(X_test)

    Y_pred = (test_probs >= best_threshold).astype(int)

    print("\n=== Test 결과 ===")

    print(
        "Exact Match Accuracy:",
        accuracy_score(Y_test, Y_pred),
    )

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

    print("\n=== 라벨별 성능 ===")
    print(
        classification_report(
            Y_test,
            Y_pred,
            target_names=mlb.classes_,
            zero_division=0,
        )
    )

    print("\n=== 오분류 문장 ===")

    error_count = 0

    for text, actual, predicted in zip(X_test, Y_test, Y_pred):
        if np.array_equal(actual, predicted):
            continue

        actual_labels = [
            label for label, value in zip(mlb.classes_, actual) if value == 1
        ]

        pred_labels = [
            label for label, value in zip(mlb.classes_, predicted) if value == 1
        ]

        print(f"문장: {text}")
        print(f"실제: {actual_labels}")
        print(f"예측: {pred_labels}\n")

        error_count += 1

    print("오분류 문장 수:", error_count)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "pipeline": model,
        "mlb": mlb,
        "threshold": best_threshold,
    }

    joblib.dump(artifact, MODEL_PATH)

    print(f"\n모델 저장 완료: {MODEL_PATH}")


if __name__ == "__main__":
    main()
