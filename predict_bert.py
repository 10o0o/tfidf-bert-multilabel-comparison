import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

MODEL_DIR = Path(__file__).resolve().parent / "models" / "bert_multilabel"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("text", type=str)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))

    threshold = metadata["threshold"] if args.threshold is None else args.threshold

    if not 0 <= threshold <= 1:
        parser.error("threshold는 0~1 사이여야 합니다")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

    model.eval()

    inputs = tokenizer(
        args.text,
        truncation=True,
        max_length=metadata["max_length"],
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.sigmoid(outputs.logits)[0].cpu().tolist()

    labels = metadata["labels"]

    result = {
        "text": args.text,
        "labels": [
            label for label, prob in zip(labels, probabilities) if prob >= threshold
        ],
        "scores": {label: round(prob, 4) for label, prob in zip(labels, probabilities)},
        "threshold": threshold,
    }

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
