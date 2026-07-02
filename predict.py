#!/usr/bin/env python3
"""Spot the fake photo: is this a real photo or a photo of a screen?

Usage:
    python predict.py some_image.jpg
Prints one number in [0, 1]:  0 = real photo, 1 = photo of a screen.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from features import extract_features

MODEL_PATH = Path(__file__).parent / "model.pkl"


def _heuristic_score(vec, names):
    """Untrained fallback used only when model.pkl is missing.

    Combines the strongest physical cues (spectral moire peaks, banding,
    backlight desaturation, dark border) through a hand-set sigmoid.
    """
    f = dict(zip(names, vec))
    z = (
        1.0 * f["fft_c_n_peaks"]
        + 0.6 * (f["fft_c_max_peak"] - 4.0)
        + 0.4 * (f["band_row_peak"] - 4.0)
        + 3.0 * (f["low_sat_frac"] - 0.2)
        + 2.5 * (f["center_minus_border"] - 0.2)
    )
    return 1.0 / (1.0 + np.exp(-z))


def predict(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        sys.exit(f"error: could not read image '{image_path}'")
    vec, names = extract_features(img)

    if MODEL_PATH.exists():
        import joblib
        model = joblib.load(MODEL_PATH)
        return float(model.predict_proba(vec.reshape(1, -1))[0, 1])

    print("warning: model.pkl not found, using untrained heuristic "
          "(run train.py for full accuracy)", file=sys.stderr)
    return float(_heuristic_score(vec, names))


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: python {Path(sys.argv[0]).name} some_image.jpg")
    score = predict(sys.argv[1])
    print(f"{score:.2f}")


if __name__ == "__main__":
    main()
