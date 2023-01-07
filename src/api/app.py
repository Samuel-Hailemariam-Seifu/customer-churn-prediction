from fastapi import FastAPI, HTTPException

from src.api.schemas import CustomerRequest, PredictionResponse
from src.config.settings import get_config
from src.models.predict import ChurnPredictor
from src.utils.logger import get_logger

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")
logger = get_logger(get_config().logger_name)

try:
    predictor = ChurnPredictor()
except Exception as exc:
    predictor = None
    logger.warning("Predictor not loaded at startup: %s", exc)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict_proba", response_model=PredictionResponse)
def predict_proba(req: CustomerRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model artifacts not found. Run training first.")
    payload = req.model_dump()
    try:
        proba = predictor.predict_proba(payload)
    except Exception as exc:
        logger.exception("Inference error in /predict_proba")
        raise HTTPException(status_code=400, detail=f"Malformed input: {exc}") from exc
    logger.info("predict_proba request served")
    return PredictionResponse(churn_prediction=int(proba >= predictor.threshold), churn_probability=proba, threshold_used=float(predictor.threshold))


@app.post("/predict", response_model=PredictionResponse)
def predict(req: CustomerRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model artifacts not found. Run training first.")
    payload = req.model_dump()
    try:
        pred, proba, threshold = predictor.predict(payload)
    except Exception as exc:
        logger.exception("Inference error in /predict")
        raise HTTPException(status_code=400, detail=f"Malformed input: {exc}") from exc
    logger.info("predict request served")
    return PredictionResponse(churn_prediction=pred, churn_probability=proba, threshold_used=threshold)
