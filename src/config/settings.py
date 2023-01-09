from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "src" / "artifacts"


@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 42
    val_size: float = 0.2
    test_size: float = 0.2
    cv_folds: int = 5
    scoring: str = "roc_auc"
    threshold_grid: List[float] = field(
        default_factory=lambda: [i / 100 for i in range(20, 81, 5)]
    )


@dataclass(frozen=True)
class DataConfig:
    dataset_urls: List[str] = field(
        default_factory=lambda: [
            "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
            "https://raw.githubusercontent.com/sonarsushant/IBM-Telco-Customer-Churn-Analysis/master/Telco-Customer-Churn.csv",
        ]
    )
    local_dataset_path: Path = DATA_DIR / "telco_churn.csv"
    target_column: str = "Churn"
    id_column: str = "customerID"


@dataclass(frozen=True)
class ArtifactConfig:
    preprocessor_path: Path = ARTIFACTS_DIR / "preprocessor.joblib"
    model_path: Path = ARTIFACTS_DIR / "model.joblib"
    threshold_path: Path = ARTIFACTS_DIR / "threshold.joblib"
    metrics_path: Path = ARTIFACTS_DIR / "metrics.json"
    feature_importance_plot_path: Path = ARTIFACTS_DIR / "feature_importance.png"


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    artifacts: ArtifactConfig = ArtifactConfig()
    logger_name: str = "customer_churn_system"
    positive_label: str = "Yes"
    negative_label: str = "No"
    numeric_coercion_cols: List[str] = field(default_factory=lambda: ["TotalCharges"])
    default_threshold: float = 0.5
    class_weight_strategy: str = "balanced"
    model_candidates: Dict[str, Dict] = field(
        default_factory=lambda: {
            "logistic_regression": {
                "C": 1.0,
                "max_iter": 2000,
                "solver": "lbfgs",
            },
            "random_forest": {
                "n_estimators": 400,
                "min_samples_leaf": 2,
                "max_depth": None,
            },
            "gradient_boosting": {
                "n_estimators": 220,
                "learning_rate": 0.05,
                "max_depth": 3,
            },
        }
    )


def get_config() -> AppConfig:
    return AppConfig()
