"""
FastAPI Sentiment Analysis API
-------------------------------
Endpoints:
  POST /predict-ml  → classical ML model prediction
  POST /predict-dl  → transformer DL model prediction
  GET  /healthcheck → service status

All predictions logged to SQLite database.
"""

import sys
import os
import time
import logging
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

# Ensure src is on path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.text_preprocessor import TextPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str

    @validator("text")
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        if len(v) > 5000:
            raise ValueError("text too long (max 5000 chars)")
        return v.strip()


class PredictResponse(BaseModel):
    label: int
    label_name: str
    probabilities: dict | None = None
    inference_time_ms: float
    model: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    uptime_seconds: float
    timestamp: str


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------
DB_PATH = str(ROOT / "logs" / "predictions.db")
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            input_text  TEXT NOT NULL,
            prediction  TEXT NOT NULL,
            label_id    INTEGER NOT NULL,
            model_used  TEXT NOT NULL,
            inference_ms REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"SQLite DB initialized at {DB_PATH}")


def log_prediction(input_text: str, prediction: str, label_id: int,
                   model_used: str, inference_ms: float):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO predictions (timestamp, input_text, prediction, label_id, model_used, inference_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), input_text, prediction, label_id, model_used, inference_ms),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


# ---------------------------------------------------------------------------
# Model registry (loaded at startup)
# ---------------------------------------------------------------------------
_models = {}
_start_time = time.time()

def load_models():
    global _models

    ml_path = str(ROOT / "models" / "saved" / "ml_model.pkl")
    dl_path = str(ROOT / "models" / "saved" / "dl_model")
    ml_cfg = str(ROOT / "src" / "config" / "ml_config.yaml")
    dl_cfg = str(ROOT / "src" / "config" / "dl_config.yaml")

    # ML Model
    try:
        from src.models.ml_model import SentimentMLModel
        if Path(ml_path).exists():
            _models["ml"] = SentimentMLModel.load(ml_path, config_path=ml_cfg)
            logger.info("ML model loaded.")
        else:
            # Train on-the-fly if no saved model
            logger.info("ML model not found. Training now...")
            from src.models.ml_model import run_training_pipeline
            data_path = str(ROOT / "data" / "processed" / "combined_dataset.csv")
            if not Path(data_path).exists():
                from src.data.dataset_builder import DatasetBuilder
                builder = DatasetBuilder(output_dir=str(ROOT / "data" / "processed"))
                builder.build_dataset()
            out = run_training_pipeline(data_path, ml_cfg, ml_path)
            _models["ml"] = out["model"]
    except Exception as e:
        logger.error(f"Could not load ML model: {e}")
        _models["ml"] = None

    # DL Model
    try:
        from src.models.dl_model import SentimentDLModel
        _models["dl"] = SentimentDLModel(config_path=dl_cfg)
        logger.info("DL model initialized.")
    except Exception as e:
        logger.error(f"Could not load DL model: {e}")
        _models["dl"] = None

    # Preprocessors
    _models["prep_ml"] = TextPreprocessor(mode="ml")
    _models["prep_dl"] = TextPreprocessor(mode="dl")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_models()
    yield

app = FastAPI(
    title="Sentiment Analysis API",
    description="ML & DL sentiment classification with logging",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/healthcheck", response_model=HealthResponse, tags=["Health"])
async def healthcheck():
    """Returns API health status and model availability."""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "ml": _models.get("ml") is not None,
            "dl": _models.get("dl") is not None,
        },
        uptime_seconds=round(time.time() - _start_time, 2),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict-ml", response_model=PredictResponse, tags=["Prediction"])
async def predict_ml(request: PredictRequest):
    """
    Predict sentiment using the classical ML model (TF-IDF + Logistic Regression).

    Returns:
        label (int): 0=negative, 1=neutral, 2=positive
        label_name (str): human-readable class name
        probabilities (dict): class probability distribution
        inference_time_ms (float): latency in milliseconds
    """
    if _models.get("ml") is None:
        raise HTTPException(status_code=503, detail="ML model not available")

    cleaned = _models["prep_ml"].clean(request.text)
    if not cleaned:
        raise HTTPException(status_code=422, detail="Text is empty after preprocessing")

    result = _models["ml"].predict_single(cleaned)
    result["timestamp"] = datetime.utcnow().isoformat()

    log_prediction(
        input_text=request.text,
        prediction=result["label_name"],
        label_id=result["label"],
        model_used="classical_ml",
        inference_ms=result["inference_time_ms"],
    )
    return result


@app.post("/predict-dl", response_model=PredictResponse, tags=["Prediction"])
async def predict_dl(request: PredictRequest):
    """
    Predict sentiment using the transformer DL model (DistilBERT).

    Returns:
        label (int): 0=negative, 1=neutral, 2=positive
        label_name (str): human-readable class name
        inference_time_ms (float): latency in milliseconds
    """
    if _models.get("dl") is None:
        raise HTTPException(status_code=503, detail="DL model not available")

    cleaned = _models["prep_dl"].clean(request.text)
    if not cleaned:
        raise HTTPException(status_code=422, detail="Text is empty after preprocessing")

    result = _models["dl"].predict_single(cleaned)
    result["timestamp"] = datetime.utcnow().isoformat()

    log_prediction(
        input_text=request.text,
        prediction=result["label_name"],
        label_id=result["label"],
        model_used="transformer_dl",
        inference_ms=result["inference_time_ms"],
    )
    return result


@app.get("/predictions/history", tags=["Logs"])
async def get_history(limit: int = 50):
    """Return recent prediction logs from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    cols = ["id", "timestamp", "input_text", "prediction", "label_id", "model_used", "inference_ms"]
    return [dict(zip(cols, row)) for row in rows]


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
