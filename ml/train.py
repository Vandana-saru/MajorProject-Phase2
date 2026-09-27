"""
PRAHARI — train the detector (Layer 1) and the severity regressors (Layer 2).

Layer 1 reproduces the Phase 1 Random Forest exactly: 50 trees, max_depth 10,
balanced class weights, seed 42, 70/15/15 stratified split, StandardScaler fit
on train only. If this does not land near 86.4% / 0.819 on BIDMC, something in
the pipeline has drifted from Phase 1 and you should stop and find out what.

Layer 2 is new: one RandomForestRegressor per fault class, predicting the
continuous severity. At inference the classifier picks which regressor runs.
Clean predictions get severity 0 by definition.

Usage
-----
    python -m ml.train                 # train, evaluate, save
    python -m ml.train --cv            # add 5-fold CV (slower)
"""

import argparse
import json
import os
import sys

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, mean_absolute_error, r2_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hub.contracts import CLASSES, FEATURE_NAMES
from hub.features import extract_batch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data", "synthetic", "corpus.npz")
MODELS = os.path.join(ROOT, "models")

RF_KW = dict(n_estimators=50, max_depth=10, random_state=42,
             class_weight="balanced", n_jobs=-1)


def load_features(corpus_path=CORPUS, cache=True):
    """Extract (or load cached) 13 features for every window in the corpus."""
    d = np.load(corpus_path, allow_pickle=True)
    cache_path = corpus_path.replace(".npz", "_features.npz")

    if cache and os.path.exists(cache_path):
        c = np.load(cache_path, allow_pickle=True)
        if len(c["F"]) == len(d["y"]):
            print(f"  using cached features {cache_path}")
            return c["F"], d["y"], d["severity"], d

    print(f"  extracting features for {len(d['y'])} windows ...", flush=True)
    F = extract_batch(d["X"])
    if cache:
        np.savez_compressed(cache_path, F=F)
    return F, d["y"], d["severity"], d


def train_detector(F, y, seed=42):
    """Layer 1. Returns (scaler, clf, metrics, split indices)."""
    idx = np.arange(len(y))
    i_tr, i_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    i_val, i_te = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=seed)

    scaler = StandardScaler().fit(F[i_tr])
    clf = RandomForestClassifier(**RF_KW).fit(scaler.transform(F[i_tr]), y[i_tr])

    pred = clf.predict(scaler.transform(F[i_te]))
    m = dict(
        accuracy=float(accuracy_score(y[i_te], pred)),
        macro_f1=float(f1_score(y[i_te], pred, average="macro")),
        n_train=len(i_tr), n_val=len(i_val), n_test=len(i_te),
    )
    print(f"\n  Layer 1 — detector")
    print(f"    test accuracy : {m['accuracy']*100:.2f}%   (Phase 1: 86.42%)")
    print(f"    test macro F1 : {m['macro_f1']:.4f}      (Phase 1: 0.8189)")
    print("\n" + classification_report(y[i_te], pred, digits=3, zero_division=0))
    print("  confusion matrix (rows=true, cols=pred), class order:",
          list(clf.classes_))
    print(confusion_matrix(y[i_te], pred, labels=clf.classes_))

    imp = sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                 key=lambda t: -t[1])
    print("\n  top features by Gini importance:")
    for n, v in imp[:5]:
        print(f"    {n:<16} {v:.4f}")
    m["feature_importance"] = {n: float(v) for n, v in imp}

    return scaler, clf, m, (i_tr, i_val, i_te)


def train_severity(F, y, sev, scaler, i_tr, i_te):
    """Layer 2. One regressor per fault class, trained on that class only."""
    regs, metrics = {}, {}
    print("\n  Layer 2 — severity regressors")
    for c in CLASSES:
        if c == "Clean":
            continue
        tr = i_tr[y[i_tr] == c]
        te = i_te[y[i_te] == c]
        if len(tr) < 20 or len(te) < 5:
            print(f"    {c:<12} too few samples, skipped")
            continue
        r = RandomForestRegressor(n_estimators=50, max_depth=10,
                                  random_state=42, n_jobs=-1)
        r.fit(scaler.transform(F[tr]), sev[tr])
        p = r.predict(scaler.transform(F[te]))
        mae = float(mean_absolute_error(sev[te], p))
        r2 = float(r2_score(sev[te], p))
        regs[c] = r
        metrics[c] = dict(mae=mae, r2=r2, n_train=len(tr), n_test=len(te))
        print(f"    {c:<12} MAE {mae:.4f}   R2 {r2:+.3f}   (n_test={len(te)})")
    return regs, metrics


def cross_validate(F, y, folds=5, seed=42):
    print(f"\n  {folds}-fold stratified CV ...", flush=True)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for k, (tr, te) in enumerate(skf.split(F, y), 1):
        sc = StandardScaler().fit(F[tr])
        c = RandomForestClassifier(**RF_KW).fit(sc.transform(F[tr]), y[tr])
        p = c.predict(sc.transform(F[te]))
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
        print(f"    fold {k}: acc {accs[-1]*100:.2f}%  f1 {f1s[-1]:.4f}")
    print(f"    mean:   acc {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%"
          f"   f1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    return dict(acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                f1_mean=float(np.mean(f1s)), f1_std=float(np.std(f1s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--onnx", action="store_true", help="also export ONNX")
    args = ap.parse_args()

    os.makedirs(MODELS, exist_ok=True)
    F, y, sev, _ = load_features(args.corpus)

    scaler, clf, m, (i_tr, i_val, i_te) = train_detector(F, y)
    regs, sev_metrics = train_severity(F, y, sev, scaler, i_tr, i_te)

    if args.cv:
        m["cv"] = cross_validate(F, y)

    joblib.dump({"scaler": scaler, "clf": clf, "severity": regs,
                 "classes": list(clf.classes_), "features": FEATURE_NAMES},
                os.path.join(MODELS, "prahari_model.joblib"))

    report = {"detector": m, "severity": sev_metrics}
    with open(os.path.join(MODELS, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    if args.onnx:
        try:
            from skl2onnx import to_onnx
            onx = to_onnx(clf, scaler.transform(F[i_tr])[:1].astype(np.float32))
            with open(os.path.join(MODELS, "detector.onnx"), "wb") as f:
                f.write(onx.SerializeToString())
            print("\n  wrote models/detector.onnx")
        except ImportError:
            print("\n  skl2onnx not installed — skipping ONNX export")

    print(f"\n  saved models/prahari_model.joblib and models/metrics.json")


if __name__ == "__main__":
    main()
