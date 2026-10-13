# TF-IDF vs BERT 다중 라벨 분류 비교

한국어 문의 한 문장에 `배송`, `환불`, `계정` 라벨을 여러 개 붙이는 작은 예제 실험입니다. 같은 데이터 분할에서 TF-IDF와 BERT를 학습하고 테스트 성능을 비교합니다.

## 실험 설정

- `data/messages.csv`의 예제 문장 78개를 사용합니다. 시드 42로 학습 46개, 검증 16개, 테스트 16개로 나눕니다.
- TF-IDF는 문자 n-그램(2~4)과 One-vs-Rest 로지스틱 회귀를 사용합니다.
- BERT는 `klue/bert-base`를 다중 라벨 분류로 미세 조정합니다.
- 두 모델 모두 검증 세트에서 임계값을 고르고, 같은 테스트 분할과 라벨 순서로 평가합니다.

| 모델 | Exact Match ↑ | Micro F1 ↑ | Macro F1 ↑ | Hamming Loss ↓ |
|---|---:|---:|---:|---:|
| TF-IDF | 0.3750 | 0.7500 | 0.7552 | 0.2500 |
| BERT | 0.2500 | 0.6667 | 0.6741 | 0.3542 |

이 결과는 작은 예제 데이터에서 한 번 수행한 비교입니다. 다른 데이터에서도 같은 성능 순서가 나온다는 뜻은 아닙니다.

## 실행

Python 3.14 이상과 [uv](https://docs.astral.sh/uv/)가 필요합니다. PyTorch는 Linux/Windows에서 CUDA 12.8 빌드를 사용하도록 설정되어 있습니다. BERT 학습 시 사전 학습 모델을 내려받습니다.

```bash
uv sync --locked
uv run python train.py
uv run python train_bert.py
uv run python compare_models.py
```

예측은 학습한 모델별로 실행합니다.

```bash
uv run python predict.py "배송이 늦어서 환불하고 싶어요"
uv run python predict_bert.py "배송이 늦어서 환불하고 싶어요"
```

TF-IDF 모델은 `models/multilabel.joblib`, BERT 모델은 `models/bert_multilabel/`에 저장됩니다. 학습 체크포인트와 모델 파일은 Git에서 제외됩니다. 데이터가 필요하면 `uv run python make_data.py`로 `data/messages.csv`를 다시 만들 수 있습니다.
