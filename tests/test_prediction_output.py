from src.api.schemas import PredictionResponse


def test_prediction_response_schema():
    payload = PredictionResponse(
        churn_prediction=1,
        churn_probability=0.83,
        threshold_used=0.45,
    )
    assert payload.churn_prediction in [0, 1]
    assert 0.0 <= payload.churn_probability <= 1.0
    assert 0.0 <= payload.threshold_used <= 1.0
