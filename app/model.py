"""Loads the trained pipeline and turns raw customers into explained predictions."""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import CATEGORICAL, NUMERIC

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[1] / "models"))


class ChurnModel:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.pipeline = joblib.load(model_dir / "churn_pipeline.joblib")
        self.meta = json.loads((model_dir / "metadata.json").read_text())
        self.threshold = float(self.meta["threshold"])
        self.version = self.meta["model_version"]
        prep = self.pipeline.named_steps["prep"]
        self.feature_names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
        self.coef = self.pipeline.named_steps["model"].coef_[0]
        self.numeric = set(NUMERIC)
        self.categorical = sorted(CATEGORICAL, key=len, reverse=True)   # longest first: "tenure_group" before "tenure"

    def _label(self, i: int, row: np.ndarray) -> str:
        """'tenure (low)' for numeric features, 'Contract = Month-to-month' for categories."""
        name = self.feature_names[i]
        if name in self.numeric:
            return f"{name} ({'high' if row[i] > 0 else 'low'})"
        for col in self.categorical:
            if name.startswith(col + "_"):
                return f"{col} = {name[len(col) + 1:]}"
        return name

    def predict(self, rows: list[dict], top_n: int = 3) -> list[dict]:
        df = pd.DataFrame(rows)
        # transform once, then reuse the matrix for both the probability and the explanation
        X = self.pipeline.named_steps["prep"].transform(self.pipeline.named_steps["features"].transform(df))
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        proba = self.pipeline.named_steps["model"].predict_proba(X)[:, 1]
        # per-customer contribution = scaled feature value x coefficient (exact for a linear model)
        contrib = X * self.coef
        out = []
        for p, c, X_row in zip(proba, contrib, X, strict=True):
            order = np.argsort(-np.abs(c))[:top_n]
            out.append({
                "churn_probability": round(float(p), 4),
                "will_churn": bool(p >= self.threshold),
                "risk_band": "high" if p >= 0.6 else "medium" if p >= 0.3 else "low",
                "top_factors": [{"feature": self._label(i, X_row),
                                 "effect": "increases risk" if c[i] > 0 else "decreases risk",
                                 "weight": round(float(c[i]), 3)} for i in order],
                "model_version": self.version,
            })
        return out
