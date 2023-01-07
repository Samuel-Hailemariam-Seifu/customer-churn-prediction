from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from src.config.settings import get_config
from src.features.engineer import add_engineered_features
from src.utils.io import load_joblib


class ChurnPredictor:
    def __init__(self):
        config = get_config()
        self.preprocessor = load_joblib(config.artifacts.preprocessor_path)
        self.model = load_joblib(config.artifacts.model_path)
        self.threshold = load_joblib(config.artifacts.threshold_path)

    def predict_proba(self, payload: Dict) -> float:
        X = pd.DataFrame([payload])
        X = add_engineered_features(X)
        X_transformed = self.preprocessor.transform(X)
        proba = self.model.predict_proba(X_transformed)[:, 1][0]
        return float(proba)

    def predict(self, payload: Dict) -> Tuple[int, float, float]:
        proba = self.predict_proba(payload)
        prediction = int(proba >= self.threshold)
        return prediction, proba, float(self.threshold)
