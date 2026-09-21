"""Train and evaluate a transparent logistic-regression baseline."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .evaluation import classification_metrics, subgroup_metrics

TARGET = "persisted_next_term"
SENSITIVE_COLUMNS = ["first_generation", "financial_aid", "online_student", "age_band"]


def train_baseline(data: pd.DataFrame, seed: int = 42) -> tuple[Pipeline, pd.DataFrame, dict[str, float]]:
    """Fit the baseline and return it, test predictions, and metrics."""
    features = data.drop(columns=[TARGET, "student_id"])
    X_train, X_test, y_train, y_test = train_test_split(
        features, data[TARGET], test_size=0.25, random_state=seed, stratify=data[TARGET]
    )
    categorical = ["age_band"]
    numeric = [column for column in features.columns if column not in categorical]
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    results = X_test.copy()
    results[TARGET] = y_test
    results["prediction"] = predictions
    results["probability"] = probabilities
    return model, results, classification_metrics(y_test, predictions, probabilities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    _, results, metrics = train_baseline(pd.read_csv(args.data))
    print(pd.Series(metrics).to_string())
    for column in SENSITIVE_COLUMNS:
        print(f"\n{column}\n{subgroup_metrics(results, column, TARGET, 'prediction').to_string(index=False)}")


if __name__ == "__main__":
    main()
