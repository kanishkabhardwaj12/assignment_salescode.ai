"""Tiny live demo: webcam -> live real/screen score in the browser.

Usage:
    python demo/server.py        # then open http://localhost:5001

The page grabs a camera frame every ~700 ms and POSTs it to /predict.
"""

import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, request, send_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from features import extract_features  # noqa: E402

app = Flask(__name__)
MODEL_PATH = ROOT / "model.pkl"
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@app.get("/")
def index():
    return send_file(Path(__file__).parent / "index.html")


@app.post("/predict")
def predict():
    buf = np.frombuffer(request.get_data(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="bad image"), 400
    vec, _ = extract_features(img)
    if model is None:
        return jsonify(error="model.pkl missing - run train.py first"), 503
    score = float(model.predict_proba(vec.reshape(1, -1))[0, 1])
    return jsonify(score=score)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
