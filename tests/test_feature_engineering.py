import pandas as pd

from src.features.engineer import add_engineered_features


def test_add_engineered_features_creates_expected_columns():
    df = pd.DataFrame(
        [
            {
                "tenure": 10,
                "MonthlyCharges": 50.0,
                "TotalCharges": 500.0,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
            }
        ]
    )

    out = add_engineered_features(df)
    assert "tenure_group" in out.columns
    assert "charges_ratio" in out.columns
    assert "service_count" in out.columns
    assert "contract_risk" in out.columns
