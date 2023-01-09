from fastapi.testclient import TestClient

import src.api.app as app_module


class DummyPredictor:
    threshold = 0.4

    def predict(self, payload):
        return 1, 0.72, self.threshold

    def predict_proba(self, payload):
        return 0.72


def test_predict_endpoint():
    app_module.predictor = DummyPredictor()
    client = TestClient(app_module.app)
    body = {"tenure": 12, "MonthlyCharges": 80.0, "TotalCharges": 960.0, "Contract": "Month-to-month"}
    response = client.post("/predict", json=body)
    assert response.status_code == 200
    out = response.json()
    assert set(out.keys()) == {"churn_prediction", "churn_probability", "threshold_used"}


def test_predict_proba_endpoint():
    app_module.predictor = DummyPredictor()
    client = TestClient(app_module.app)
    body = {"tenure": 12, "MonthlyCharges": 80.0, "TotalCharges": 960.0, "Contract": "Month-to-month"}
    response = client.post("/predict_proba", json=body)
    assert response.status_code == 200
    out = response.json()
    assert out["churn_prediction"] in [0, 1]
