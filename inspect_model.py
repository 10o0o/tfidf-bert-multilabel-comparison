import joblib
import numpy as np

artifact = joblib.load("models/multilabel.joblib")

model = artifact["pipeline"]
mlb = artifact["mlb"]

vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["clf"]

features = vectorizer.get_feature_names_out()

print("TF-IDF 특징 수:", len(features))

for label, estimator in zip(
    mlb.classes_,
    classifier.estimators_,
):
    # 각 클래스에 대한 양의 가중치가 큰 특징
    coefficients = estimator.coef_[0]

    indices = np.argsort(coefficients)[-10:][::-1]

    print(f"\n[{label}] 주요 특징")

    for index in indices:
        print(
            repr(features[index]),
            round(coefficients[index], 4),
        )
