import pandas as pd

from src.features.preprocess import build_preprocessor, infer_feature_types


def test_preprocessor_transforms_mixed_types():
    df = pd.DataFrame(
        {
            "tenure": [1, 5, 10],
            "MonthlyCharges": [50.0, 70.0, 90.0],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "target": [0, 1, 0],
        }
    )
    numeric, categorical = infer_feature_types(df, target_column="target")
    preprocessor = build_preprocessor(numeric, categorical)
    transformed = preprocessor.fit_transform(df.drop(columns=["target"]))
    assert transformed.shape[0] == 3
    assert transformed.shape[1] > 0
