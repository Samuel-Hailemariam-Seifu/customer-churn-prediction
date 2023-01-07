import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "tenure" in out.columns:
        out["tenure_group"] = pd.cut(
            out["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0_12", "13_24", "25_48", "49_72"],
        ).astype(str)

    if {"MonthlyCharges", "TotalCharges"}.issubset(out.columns):
        monthly = out["MonthlyCharges"].replace(0, np.nan)
        out["charges_ratio"] = out["TotalCharges"] / monthly
        out["charges_ratio"] = out["charges_ratio"].replace([np.inf, -np.inf], np.nan)

    service_cols = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    available_service_cols = [c for c in service_cols if c in out.columns]
    if available_service_cols:
        normalized = out[available_service_cols].astype(str).applymap(
            lambda x: 0
            if x.strip().lower() in {"no", "no internet service", "no phone service"}
            else 1
        )
        out["service_count"] = normalized.sum(axis=1)

    if "Contract" in out.columns:
        contract_map = {"Month-to-month": "high_risk", "One year": "medium_risk", "Two year": "low_risk"}
        out["contract_risk"] = out["Contract"].map(contract_map).fillna("unknown")

    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_engineered_features(X)
