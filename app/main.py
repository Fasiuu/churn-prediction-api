"""FastAPI service for churn predictions.

    uvicorn app.main:app --reload
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .model import ChurnModel
from .schemas import BatchRequest, BatchResponse, Customer, Prediction

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("churn-api")

REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("http_request_duration_seconds", "Request latency", ["path"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("churn_predictions_total", "Predictions served", ["risk_band"])

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = ChurnModel()
    log.info(json.dumps({"event": "model_loaded", "version": state["model"].version}))
    yield
    state.clear()


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan,
              description="Predicts the probability that a telecom customer will churn, with the top factors behind it.")


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    path = request.url.path
    REQUESTS.labels(request.method, path, response.status_code).inc()
    LATENCY.labels(path).observe(elapsed)
    response.headers["x-request-id"] = rid
    if path != "/metrics":
        log.info(json.dumps({"rid": rid, "method": request.method, "path": path,
                             "status": response.status_code, "ms": round(elapsed * 1000, 2)}))
    return response


def _model() -> ChurnModel:
    model = state.get("model")
    if model is None:
        raise HTTPException(503, "Model not loaded")
    return model


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model/info")
def model_info():
    return _model().meta


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    result = _model().predict([customer.model_dump()])[0]
    PREDICTIONS.labels(result["risk_band"]).inc()
    return result


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    results = _model().predict([c.model_dump() for c in req.customers])
    for r in results:
        PREDICTIONS.labels(r["risk_band"]).inc()
    return {"predictions": results, "count": len(results)}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
