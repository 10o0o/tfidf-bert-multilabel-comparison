# TF-IDF 다중 라벨 분류기

한국어 문의 문장을 배송, 환불, 계정으로 분류하는 간단한 프로젝트입니다.
TF-IDF와 로지스틱 회귀를 사용하며, 한 문장에 여러 라벨을 예측할 수 있습니다.

Python 3.14 이상과 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
uv sync --locked
uv run python train.py
uv run python predict.py "배송이 늦어서 환불하고 싶어요"
```

예제 데이터는 `data/messages.csv`에 포함되어 있습니다.
학습하면 검증 데이터로 예측 임계값을 선택하고 모델을 `models/multilabel.joblib`에 저장합니다.
