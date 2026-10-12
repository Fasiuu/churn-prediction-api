"""Train the churn model and save it with its metadata.

    python train.py

Logistic regression was the strongest model in cross-validation in my churn analysis notebook
(github.com/Fasiuu/telco-churn-prediction), and it's small, fast and explainable, which suits an
API: each prediction can come with the features that pushed it up or down.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.features import CATEGORICAL, NUMERIC, ChurnFeatures

ROOT = Path(__file__).parent
SEED = 42
MODEL_VERSION = "1.0.0"


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("features", ChurnFeatures()), ("prep", pre),
                     ("model", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))])


def main():
    df = pd.read_csv(ROOT / "data" / "Telco-Customer-Churn.csv")
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn", "customerID"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    pipe = build_pipeline()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, oof)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    threshold = float(thr[np.argmax(f1[:-1])])

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "cv_roc_auc": round(roc_auc_score(y_train, oof), 4),
        "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
        "test_pr_auc": round(average_precision_score(y_test, proba), 4),
        "test_f1": round(f1_score(y_test, pred), 4),
        "test_recall": round(float(pred[y_test.values == 1].mean()), 4),
        "test_precision": round(float(y_test.values[pred == 1].mean()), 4),
    }

    # refit on all data for the served model
    final = build_pipeline().fit(X, y)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(final, ROOT / "models" / "churn_pipeline.joblib")
    meta = {
        "model_version": MODEL_VERSION, "algorithm": "LogisticRegression (class_weight=balanced, C=0.5)",
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "training_rows": int(len(X)), "threshold": round(threshold, 4),
        "metrics_holdout": metrics, "sklearn_version": sklearn.__version__,
    }
    (ROOT / "models" / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
