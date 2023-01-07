from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config.settings import get_config
from src.data.loader import load_dataset, standardize_dataframe
from src.features.engineer import FeatureEngineer, add_engineered_features
from src.features.preprocess import build_preprocessor, infer_feature_types
from src.models.metrics import evaluate_threshold, tune_threshold
from src.utils.io import save_joblib, save_json
from src.utils.logger import get_logger


def split_data(df: pd.DataFrame, target_column: str, random_state: int, val_size: float, test_size: float):
    y = (df[target_column] == "Yes").astype(int)
    X = df.drop(columns=[target_column])

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    adjusted_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_models() -> Dict[str, object]:
    config = get_config()
    return {
        "logistic_regression": LogisticRegression(
            random_state=config.model.random_state,
            class_weight=config.class_weight_strategy,
            **config.model_candidates["logistic_regression"],
        ),
        "random_forest": RandomForestClassifier(
            random_state=config.model.random_state,
            class_weight=config.class_weight_strategy,
            n_jobs=-1,
            **config.model_candidates["random_forest"],
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=config.model.random_state,
            **config.model_candidates["gradient_boosting"],
        ),
    }


def _train_single_model(
    model_name: str,
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
):
    X_train_fe = add_engineered_features(X_train)
    numeric_features, categorical_features = infer_feature_types(
        X_train_fe.assign(target=y_train), target_column="target"
    )
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    pipeline = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    pipeline.fit(X_train, y_train)
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    return model_name, pipeline, val_auc, val_proba


def run_training() -> Dict:
    config = get_config()
    logger = get_logger(config.logger_name)
    logger.info("Starting training pipeline")

    raw_df = load_dataset()
    df = standardize_dataframe(raw_df)
    df = df.dropna(subset=[config.data.target_column])

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df=df,
        target_column=config.data.target_column,
        random_state=config.model.random_state,
        val_size=config.model.val_size,
        test_size=config.model.test_size,
    )

    models = build_models()
    fitted = []
    for name, estimator in models.items():
        fitted.append(_train_single_model(name, estimator, X_train, y_train, X_val, y_val))

    best_name, best_pipeline, _, best_val_proba = max(fitted, key=lambda x: x[2])
    threshold_results = tune_threshold(
        y_val.to_numpy(),
        best_val_proba,
        config.model.threshold_grid,
    )
    best_threshold = threshold_results["best"]["threshold"]

    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_threshold(y_test.to_numpy(), test_proba, best_threshold)

    feature_importance = explain_model(best_pipeline, X_train)

    save_joblib(best_pipeline.named_steps["preprocessor"], config.artifacts.preprocessor_path)
    save_joblib(best_pipeline.named_steps["model"], config.artifacts.model_path)
    save_joblib(best_threshold, config.artifacts.threshold_path)

    payload = {
        "best_model": best_name,
        "validation_threshold_search": threshold_results,
        "test_metrics": test_metrics,
        "class_distribution": {
            "non_churn_ratio": float((y_train == 0).mean()),
            "churn_ratio": float((y_train == 1).mean()),
        },
        "feature_importance_top10": feature_importance[:10],
    }
    save_json(payload, config.artifacts.metrics_path)

    logger.info("Training complete. Best model: %s", best_name)
    return payload


def explain_model(pipeline: Pipeline, X_train: pd.DataFrame) -> list[Tuple[str, float]]:
    config = get_config()
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_train_fe = add_engineered_features(X_train)
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        coefs = model.coef_[0]
        importances = np.abs(coefs)

    paired = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    top = paired[:20]

    plot_features = [x[0] for x in top][::-1]
    plot_values = [x[1] for x in top][::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(plot_features, plot_values)
    plt.title("Top Feature Importances")
    plt.tight_layout()
    config.artifacts.feature_importance_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.artifacts.feature_importance_plot_path)
    plt.close()

    return [(k, float(v)) for k, v in paired]
