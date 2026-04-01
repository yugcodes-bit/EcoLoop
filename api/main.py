# ============================================================
# ECO-LOOP — FastAPI REST Server
# api/main.py
#
# Connects sensor readings → normalization → ML models
# → compost engine → JSON response
#
# Run with: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Access at: http://localhost:8000
# API docs at: http://localhost:8000/docs
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import sys
import os
from datetime import datetime

# Add project root to path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.compost_engine import get_compost_prescription
from utils.normalize import (normalize_reading,
                              validate_calibration,
                              reading_to_list,
                              CHANNEL_NAMES)

# ── Initialize FastAPI app ─────────────────────────────────────────────────
app = FastAPI(
    title="ECO-LOOP API",
    description="""
    ECO-LOOP — Edge-AI Vis-NIR Spectral System
    for Non-Destructive Apple Rot Classification,
    pH Prediction, and Compost Valorization.
    
    Built on Raspberry Pi 4 | AS7265x 18-channel sensor
    2nd Year B.E. CSE Research Project
    """,
    version="1.0.0"
)

# ── Load ML models at startup ──────────────────────────────────────────────
print("\nLoading ECO-LOOP models...")

try:
    scaler     = joblib.load("models/scaler.pkl")
    classifier = joblib.load("models/best_classifier.pkl")
    regressor  = joblib.load("models/best_regressor.pkl")
    print("✅ Scaler loaded")
    print("✅ Classifier loaded")
    print("✅ Regressor loaded")
except FileNotFoundError as e:
    print(f"❌ Model file not found: {e}")
    print("   Run models/train_models.py first")

# ── Default calibration references ────────────────────────────────────────
# These are updated by calling /calibrate endpoint
# before each scanning session
calibration_store = {
    "dark_ref":  [10] * 18,
    "white_ref": [900] * 18,
    "calibrated": False,
    "last_calibrated": None
}

# ── Request and Response models ────────────────────────────────────────────

class RawScanRequest(BaseModel):
    """18 raw integer readings directly from AS7265x sensor"""
    raw_readings: list = Field(
        ...,
        description="List of 18 raw integer values from AS7265x sensor",
        example=[280, 310, 360, 400, 440, 490,
                 510, 530, 540, 510, 260, 490,
                 570, 620, 750, 680, 620, 560]
    )

class NormalizedScanRequest(BaseModel):
    """Pre-normalized reflectance values (0.0 to 1.0)"""
    ch_410nm: float = Field(..., ge=0.0, le=1.0)
    ch_435nm: float = Field(..., ge=0.0, le=1.0)
    ch_460nm: float = Field(..., ge=0.0, le=1.0)
    ch_485nm: float = Field(..., ge=0.0, le=1.0)
    ch_510nm: float = Field(..., ge=0.0, le=1.0)
    ch_535nm: float = Field(..., ge=0.0, le=1.0)
    ch_560nm: float = Field(..., ge=0.0, le=1.0)
    ch_585nm: float = Field(..., ge=0.0, le=1.0)
    ch_610nm: float = Field(..., ge=0.0, le=1.0)
    ch_645nm: float = Field(..., ge=0.0, le=1.0)
    ch_680nm: float = Field(..., ge=0.0, le=1.0)
    ch_705nm: float = Field(..., ge=0.0, le=1.0)
    ch_730nm: float = Field(..., ge=0.0, le=1.0)
    ch_760nm: float = Field(..., ge=0.0, le=1.0)
    ch_810nm: float = Field(..., ge=0.0, le=1.0)
    ch_860nm: float = Field(..., ge=0.0, le=1.0)
    ch_900nm: float = Field(..., ge=0.0, le=1.0)
    ch_940nm: float = Field(..., ge=0.0, le=1.0)

class CalibrationRequest(BaseModel):
    """Dark and white reference readings for calibration"""
    dark_readings:  list = Field(...,
        description="18 dark reference values — box sealed, LEDs off")
    white_readings: list = Field(...,
        description="18 white reference values — PTFE tile, LEDs on")


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    """Health check — confirms server is running"""
    return {
        "system": "ECO-LOOP",
        "status": "running",
        "version": "1.0.0",
        "message": "Edge-AI Apple Rot Detection System is online",
        "endpoints": {
            "POST /calibrate":       "Set dark and white reference readings",
            "POST /predict/raw":     "Predict from raw AS7265x sensor values",
            "POST /predict/normalized": "Predict from pre-normalized values",
            "GET  /calibration/status": "Check current calibration status",
            "GET  /docs":            "Interactive API documentation"
        }
    }


@app.post("/calibrate")
def calibrate(request: CalibrationRequest):
    """
    Store dark and white reference readings.
    Call this before every scanning session.
    """

    if len(request.dark_readings) != 18:
        raise HTTPException(
            status_code=400,
            detail=f"dark_readings must have 18 values, got {len(request.dark_readings)}"
        )

    if len(request.white_readings) != 18:
        raise HTTPException(
            status_code=400,
            detail=f"white_readings must have 18 values, got {len(request.white_readings)}"
        )

    # Validate calibration quality
    from utils.normalize import validate_calibration
    validation = validate_calibration(
        request.dark_readings,
        request.white_readings
    )

    if not validation["calibration_valid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Calibration failed: {validation['details']}"
        )

    # Store references
    calibration_store["dark_ref"]       = request.dark_readings
    calibration_store["white_ref"]      = request.white_readings
    calibration_store["calibrated"]     = True
    calibration_store["last_calibrated"] = datetime.now().isoformat()

    return {
        "status": "Calibration successful",
        "calibrated_at": calibration_store["last_calibrated"],
        "channels_calibrated": 18,
        "message": "System ready for apple scanning"
    }


@app.get("/calibration/status")
def calibration_status():
    """Check if system is calibrated and ready"""
    return {
        "calibrated": calibration_store["calibrated"],
        "last_calibrated": calibration_store["last_calibrated"],
        "ready_to_scan": calibration_store["calibrated"],
        "message": "Ready to scan" if calibration_store["calibrated"]
                   else "Please run /calibrate before scanning"
    }


@app.post("/predict/raw")
def predict_from_raw(request: RawScanRequest):
    """
    Main prediction endpoint — takes raw AS7265x readings.

    Steps:
    1. Normalize raw readings using stored calibration
    2. Scale using StandardScaler
    3. Run classifier — get rot class
    4. Run regressor — get pH prediction
    5. Run compost engine — get prescription
    6. Return complete JSON response
    """

    start_time = datetime.now()

    # Validate input
    if len(request.raw_readings) != 18:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 18 raw readings, got {len(request.raw_readings)}"
        )

    # Step 1 — Normalize
    normalized_dict = normalize_reading(
        request.raw_readings,
        calibration_store["dark_ref"],
        calibration_store["white_ref"]
    )

    # Step 2 — Convert to ordered list then scale
    normalized_list = reading_to_list(normalized_dict)
    X = np.array(normalized_list).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # Step 3 — Classify
    rot_class = int(classifier.predict(X_scaled)[0])

    # Step 4 — Predict pH
    ph_pred = regressor.predict(X_scaled)
    if hasattr(ph_pred, "ravel"):
        ph_pred = float(ph_pred.ravel()[0])
    else:
        ph_pred = float(ph_pred[0])

    # Step 5 — Compost prescription
    prescription = get_compost_prescription(rot_class, ph_pred)

    # Calculate total processing time
    end_time   = datetime.now()
    duration   = (end_time - start_time).total_seconds()

    # Add scan duration to response
    prescription["scan_duration_seconds"] = round(duration, 3)
    prescription["normalized_readings"]   = normalized_dict

    return prescription


@app.post("/predict/normalized")
def predict_from_normalized(request: NormalizedScanRequest):
    """
    Prediction endpoint — takes pre-normalized values.
    Useful for testing without physical sensor.
    """

    start_time = datetime.now()

    # Build ordered list from request
    normalized_list = [
        request.ch_410nm, request.ch_435nm, request.ch_460nm,
        request.ch_485nm, request.ch_510nm, request.ch_535nm,
        request.ch_560nm, request.ch_585nm, request.ch_610nm,
        request.ch_645nm, request.ch_680nm, request.ch_705nm,
        request.ch_730nm, request.ch_760nm, request.ch_810nm,
        request.ch_860nm, request.ch_900nm, request.ch_940nm
    ]

    # Scale
    X = np.array(normalized_list).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # Classify
    rot_class = int(classifier.predict(X_scaled)[0])

    # Predict pH
    ph_pred = regressor.predict(X_scaled)
    if hasattr(ph_pred, "ravel"):
        ph_pred = float(ph_pred.ravel()[0])
    else:
        ph_pred = float(ph_pred[0])

    # Compost prescription
    prescription = get_compost_prescription(rot_class, ph_pred)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    prescription["scan_duration_seconds"] = round(duration, 3)

    return prescription






# ```

# {"system":"ECO-LOOP","status":"running","version":"1.0.0","message":"Edge-AI Apple Rot Detection System is online","endpoints":{"POST /calibrate":"Set dark and white reference readings","POST /predict/raw":"Predict from raw AS7265x sensor values","POST /predict/normalized":"Predict from pre-normalized values","GET  /calibration/status":"Check current calibration status","GET  /docs":"Interactive API documentation"}}

# ```



# this i can only see on chrome


# this is i the vscode terminal

# Loading ECO-LOOP models...

# ✅ Scaler loaded

# ✅ Classifier loaded

# ✅ Regressor loaded

# INFO:     Started server process [7888]

# INFO:     Waiting for application startup.

# INFO:     Application startup complete.

# INFO:     127.0.0.1:59880 - "GET / HTTP/1.1" 200 OK

# INFO:     127.0.0.1:59880 - "GET /favicon.ico HTTP/1.1" 404 Not Found