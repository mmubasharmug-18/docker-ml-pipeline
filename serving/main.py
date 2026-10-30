import logging
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from serving.config import Settings, load_settings
from serving.handlers import validation_error_handler
from serving.logs import setup_logging
from serving.model_store import ModelNotAvailable, current_version, list_versions, load_model
from serving.schemas import PredictIn, PredictionOut

logger = logging.getLogger(__name__)

DESCRIPTION = (
    "Predicts whether a patient is likely to have diabetes from 8 measurements. "
    "A value of 0 for glucose, blood_pressure, skin_thickness, insulin or bmi means "
    "'not measured' and is filled in with the training median. "
    "This is a learning project, not medical advice."
)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or load_settings()
    setup_logging(settings.log_level)
    state = {"model": None, "problem": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            loaded = load_model(settings.model_dir, settings.model_version)
            state["model"] = loaded
            state["problem"] = None
            logger.info(
                "Loaded model %s: %s (%s=%s)",
                loaded.version,
                loaded.name,
                loaded.metadata["selection_metric"],
                loaded.metadata["selection_score"],
            )
        except ModelNotAvailable as error:
            state["model"] = None
            state["problem"] = str(error)
            logger.error("No model loaded: %s", error)
        yield

    app = FastAPI(title="Diabetes Prediction API", version="1.0.0", description=DESCRIPTION, lifespan=lifespan)
    app.add_exception_handler(RequestValidationError, validation_error_handler)

    def require_model():
        if state["model"] is None:
            raise HTTPException(status_code=503, detail=state["problem"] or "No model is loaded")
        return state["model"]

    @app.get("/health", summary="Is the service ready to predict?")
    def health():
        loaded = state["model"]
        if loaded is None:
            return JSONResponse(status_code=503, content={"status": "no_model", "detail": state["problem"]})
        return {"status": "ok", "model_version": loaded.version}

    @app.get("/model", summary="Which model is serving, and how good is it?")
    def model_info():
        loaded = require_model()
        meta = loaded.metadata
        return {
            "version": loaded.version,
            "model_name": loaded.name,
            "created_at": meta["created_at"],
            "selection_metric": meta["selection_metric"],
            "selection_score": meta["selection_score"],
            "test_metrics": meta["test_metrics"],
            "threshold": loaded.threshold,
            "features": loaded.features,
            "dataset": meta["dataset"],
            "trained_with": meta.get("libraries", {}),
            "running_scikit_learn": sklearn.__version__,
            "sklearn_matches": loaded.sklearn_matches,
            "pinned_version": settings.model_version,
            "current_version": current_version(settings.model_dir),
            "available_versions": list_versions(settings.model_dir),
        }

    @app.post("/predict", response_model=PredictionOut, summary="Predict diabetes from 8 measurements")
    def predict(data: PredictIn):
        loaded = require_model()
        row = pd.DataFrame([data.model_dump()], columns=loaded.features)
        ready = loaded.preprocessor.transform(row)
        probability = float(loaded.model.predict_proba(ready)[0, 1])
        prediction = int(probability >= loaded.threshold)
        logger.info("Predicted class %d (probability %.3f) with model %s", prediction, probability, loaded.version)
        return PredictionOut(
            prediction=prediction,
            label=loaded.labels[str(prediction)],
            probability=round(probability, 4),
            threshold=loaded.threshold,
            model_version=loaded.version,
            model_name=loaded.name,
        )

    return app


app = create_app()
