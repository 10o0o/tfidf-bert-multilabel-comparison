import argparse
import json
from pathlib import Path

import joblib

from train import MODEL_PATH

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "multilabel.joblib"


def main():
    parser = argparse.ArgumentParser(description="TF_IDF 복수 라벨 문장 분류기")

    parser.add_argument("text", type=str, help="분류할 한국어 문장")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="예측 임계값 (기본값: 학습 중 선택한 값)",
    )

    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("모델 파일이 없습니다. python train.py를 먼저 실행하세요")

    artifact = joblib.load(MODEL_PATH)

    model = artifact["pipeline"]
    mlb = artifact["mlb"]

    threshold = artifact["threshold"] if args.threshold is None else args.threshold

    if not 0 <= threshold <= 1:
        parser.error("threshold는 0~1 사이여야 합니다.")

    probabilities = model.predict_proba([args.text])[0]

    scores = {
        label: round(float(prob), 4) for label, prob in zip(mlb.classes_, probabilities)
    }

    labels = [
        label for label, prob in zip(mlb.classes_, probabilities) if prob >= threshold
    ]

    result = {
        "text": args.text,
        "labels": labels,
        "scores": scores,
        "threshold": round(float(threshold), 4),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
