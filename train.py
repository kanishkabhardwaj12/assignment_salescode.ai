"""Train the screen-recapture classifier.

Expects:
    data/real/    ~50 normal photos
    data/screen/  ~50 photos of a screen (or printout) showing a picture

Usage:
    python train.py

Reports honest 5-fold cross-validated accuracy, then fits the best model on
all data and saves it to model.pkl for predict.py.
"""

import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import extract_features

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"
CACHE_PATH = Path(__file__).parent / "features_cache.npz"
EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp"}


def load_dataset():
    files, labels = [], []
    for label, sub in ((0, "real"), (1, "screen")):
        folder = DATA_DIR / sub
        if not folder.is_dir():
            sys.exit(f"Missing folder: {folder} - put your photos there first.")
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in EXTS:
                files.append(f)
                labels.append(label)
    if not files:
        sys.exit(f"No images found under {DATA_DIR}/real and {DATA_DIR}/screen.")
    return files, np.array(labels)


def extract_all(files):
    """Extract features for every file, with a cache keyed by path+mtime."""
    cache = {}
    if CACHE_PATH.exists():
        z = np.load(CACHE_PATH, allow_pickle=True)
        cache = dict(zip(z["keys"], z["vecs"]))

    X, names = [], None
    t0 = time.time()
    for i, f in enumerate(files, 1):
        key = f"{f}:{f.stat().st_mtime_ns}"
        if key in cache:
            vec = cache[key]
        else:
            img = cv2.imread(str(f))
            if img is None:
                print(f"  ! could not read {f.name}, skipping")
                vec = None
            else:
                vec, names = extract_features(img)
            cache[key] = vec
        X.append(vec)
        if i % 10 == 0 or i == len(files):
            print(f"  features {i}/{len(files)} ({time.time() - t0:.1f}s)")

    np.savez_compressed(
        CACHE_PATH,
        keys=np.array(list(cache.keys())),
        vecs=np.array(list(cache.values()), dtype=object),
    )
    keep = [i for i, v in enumerate(X) if v is not None]
    return np.array([X[i] for i in keep], dtype=np.float32), keep


def main():
    files, y = load_dataset()
    print(f"Dataset: {int((y == 0).sum())} real, {int((y == 1).sum())} screen")

    X, keep = extract_all(files)
    y = y[keep]

    candidates = {
        "logreg": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=0, n_jobs=-1),
        "gbm": GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.06, random_state=0),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    best_name, best_acc, best_pred = None, -1.0, None
    print("\n5-fold cross-validation (honest accuracy):")
    for name, model in candidates.items():
        pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
        acc = float((pred == y).mean())
        print(f"  {name:7s} accuracy = {acc:.3f}")
        if acc > best_acc:
            best_name, best_acc, best_pred = name, acc, pred

    tn, fp, fn, tp = confusion_matrix(y, best_pred).ravel()
    print(f"\nBest: {best_name}  CV accuracy = {best_acc:.3f}")
    print(f"Confusion (best): real->screen {fp}/{tn + fp}, "
          f"screen->real {fn}/{fn + tp}")

    final = candidates[best_name].fit(X, y)
    joblib.dump(final, MODEL_PATH, compress=3)
    print(f"\nSaved {MODEL_PATH.name} "
          f"({MODEL_PATH.stat().st_size / 1024:.0f} KB) - ready for predict.py")


if __name__ == "__main__":
    main()
