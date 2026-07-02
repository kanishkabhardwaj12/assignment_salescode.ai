"""Hyperparameter tuning and ensemble to try improving accuracy.

Usage:
    python train_tune.py

Runs randomized search for LogisticRegression, RandomForest, and GradientBoosting,
then evaluates a soft Voting ensemble of the best-found estimators.
"""

from pathlib import Path
import time

import joblib
import numpy as np
from scipy.stats import randint, loguniform
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train


def main():
    files, y = train.load_dataset()
    print(f"Dataset: {int((y == 0).sum())} real, {int((y == 1).sum())} screen")

    X, keep = train.extract_all(files)
    y = y[keep]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # Logistic regression pipeline
    logpipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="liblinear"))
    log_dist = {"logisticregression__C": loguniform(1e-3, 1e3)}
    rs_log = RandomizedSearchCV(logpipe, log_dist, n_iter=30, cv=cv, scoring="accuracy", n_jobs=-1, random_state=0)

    # Random forest
    rf = RandomForestClassifier(random_state=0)
    rf_dist = {
        "n_estimators": randint(100, 601),
        "max_depth": [3, 6, 12, None],
        "min_samples_split": randint(2, 6),
    }
    rs_rf = RandomizedSearchCV(rf, rf_dist, n_iter=30, cv=cv, scoring="accuracy", n_jobs=-1, random_state=0)

    # Gradient boosting
    gb = GradientBoostingClassifier(random_state=0)
    gb_dist = {
        "n_estimators": randint(100, 501),
        "learning_rate": loguniform(1e-3, 1.0),
        "max_depth": randint(1, 4),
    }
    rs_gb = RandomizedSearchCV(gb, gb_dist, n_iter=30, cv=cv, scoring="accuracy", n_jobs=-1, random_state=0)

    searches = [("logreg", rs_log), ("rf", rs_rf), ("gbm", rs_gb)]

    best_estimators = {}
    start = time.time()
    for name, rs in searches:
        print(f"\nRunning randomized search for {name}...")
        rs.fit(X, y)
        print(f"Best {name} CV accuracy = {rs.best_score_:.3f}")
        print(f"Best params: {rs.best_params_}")
        best_estimators[name] = rs.best_estimator_

    # Try a soft-voting ensemble of the three best estimators
    print("\nEvaluating VotingClassifier (soft) via 5-fold CV...")
    estimators = [(k, v) for k, v in best_estimators.items()]
    # Ensure estimator names are valid identifiers
    estimators = [(k, v) for k, v in estimators]
    # VotingClassifier requires predict_proba for soft voting; wrap if necessary
    vot = VotingClassifier(estimators=estimators, voting="soft")
    pred = cross_val_predict(vot, X, y, cv=cv, n_jobs=-1)
    acc = float((pred == y).mean())
    print(f"Voting ensemble CV accuracy = {acc:.3f}")

    # Save the best single estimator and the ensemble
    best_name, best_acc = max(((n, e.score(X, y) if hasattr(e, 'score') else 0) for n, e in best_estimators.items()), key=lambda t: t[1])
    print(f"\nBest single estimator (train score): {best_name}")

    MODEL_PATH = Path(__file__).parent / "model_tuned.pkl"
    joblib.dump({"best_estimators": best_estimators, "voting": vot}, MODEL_PATH, compress=3)
    print(f"Saved tuned artifacts to {MODEL_PATH}")
    print(f"Elapsed: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
