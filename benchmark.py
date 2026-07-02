"""Measure per-image latency of the predictor.

Usage:
    python benchmark.py data/real        # or any folder of images

Reports the warm per-image time (decode + features + inference), which is
what matters in a serving loop, plus the one-off model-load cost.
"""

import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

from features import extract_features

MODEL_PATH = Path(__file__).parent / "model.pkl"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/real")
    files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in EXTS]
    if not files:
        sys.exit(f"no images in {folder}")

    t0 = time.perf_counter()
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    load_ms = (time.perf_counter() - t0) * 1000

    decode_ms, feat_ms, infer_ms = [], [], []
    for f in files:
        t0 = time.perf_counter()
        img = cv2.imread(str(f))
        t1 = time.perf_counter()
        vec, _ = extract_features(img)
        t2 = time.perf_counter()
        if model is not None:
            model.predict_proba(vec.reshape(1, -1))
        t3 = time.perf_counter()
        decode_ms.append((t1 - t0) * 1000)
        feat_ms.append((t2 - t1) * 1000)
        infer_ms.append((t3 - t2) * 1000)

    d, ft, inf = map(np.array, (decode_ms, feat_ms, infer_ms))
    total = d + ft + inf
    print(f"images: {len(files)}   (model load, one-off: {load_ms:.0f} ms)")
    print(f"decode   : {d.mean():7.1f} ms  (median {np.median(d):.1f})")
    print(f"features : {ft.mean():7.1f} ms  (median {np.median(ft):.1f})")
    print(f"inference: {inf.mean():7.1f} ms  (median {np.median(inf):.1f})")
    print(f"TOTAL    : {total.mean():7.1f} ms/image  "
          f"(median {np.median(total):.1f}, p95 {np.percentile(total, 95):.1f})")


if __name__ == "__main__":
    main()
