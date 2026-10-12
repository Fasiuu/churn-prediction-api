"""Feature engineering as a scikit-learn transformer, so it travels inside the saved pipeline.

Keeping this in an importable module (instead of a notebook cell) means the pickled model
can be loaded by the API without copying functions around.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "num_addons", "avg_monthly_spend", "charge_increase"]
CATEGORICAL = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
               "Contract", "PaperlessBilling", "PaymentMethod", "tenure_group", "auto_pay"]


class ChurnFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        d = X.copy()
        d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0.0)
        d["SeniorCitizen"] = d["SeniorCitizen"].map({0: "No", 1: "Yes", "0": "No", "1": "Yes"}).fillna(d["SeniorCitizen"])
        d["num_addons"] = (d[ADDON_COLS] == "Yes").sum(axis=1)
        d["tenure_group"] = pd.cut(d["tenure"], bins=[-1, 6, 12, 24, 48, 10_000],
                                   labels=["0-6m", "7-12m", "1-2y", "2-4y", "4y+"]).astype(str)
        d["avg_monthly_spend"] = np.where(d["tenure"] > 0, d["TotalCharges"] / d["tenure"].clip(lower=1), d["MonthlyCharges"])
        d["charge_increase"] = d["MonthlyCharges"] - d["avg_monthly_spend"]
        d["auto_pay"] = np.where(d["PaymentMethod"].str.contains("automatic"), "Yes", "No")
        return d[NUMERIC + CATEGORICAL]
