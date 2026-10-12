"""Train churn models on the TRAIN partition, select on VALIDATION. The test partition is never loaded here.

    python scripts/train_model.py                       # primary experiment: customers active in the last 90 days
    python scripts/train_model.py --population all_buyers --tag all_buyers   # secondary experiment
    python scripts/train_model.py --quick               # tiny grids / row limit, for smoke tests

Outputs (git-ignored): models/<tag>/champion.joblib, models/<tag>/candidates/*.joblib, models/<tag>/metadata.json
and an appended line in reports/experiments.jsonl.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402

from src import config as C  # noqa: E402
from src.database import DatabaseError, check_connection, read_sql_df  # noqa: E402
from src.evaluation import choose_threshold, probability_metrics, ranking_metrics  # noqa: E402
from src.feature_extraction import FEATURES, TARGET, load_labelled_snapshots  # noqa: E402
from src.modeling import (HAVE_XGBOOST, ChurnModel, baseline_scores, build_pipeline, partition_snapshots,  # noqa: E402
                          select_columns, validate_temporal_split)

log = logging.getLogger("train_model")


def grids(quick: bool) -> dict[str, list[dict]]:
    if quick:
        return {"logreg": [{"C": 0.1}], "random_forest": [{"min_samples_leaf": 50, "n_estimators": 60}],
                "xgboost": [{"max_depth": 3, "min_child_weight": 10, "n_estimators": 100, "learning_rate": 0.1}]}
    return {
        "logreg": [{"C": c} for c in (0.01, 0.1, 1.0)],
        "random_forest": [{"min_samples_leaf": m, "max_features": "sqrt"} for m in (20, 50, 100)],
        "xgboost": [dict(max_depth=d, min_child_weight=w, learning_rate=0.05, n_estimators=1000, subsample=0.8,
                         colsample_bytree=0.8, early_stopping_rounds=50)
                    for d, w in itertools.product((3, 5), (5, 20))],
    }


def fit_one(kind: str, params: dict, seed: int, tr, va):
    """Fit one candidate. XGBoost uses the validation set only for early stopping (number of trees)."""
    pipe = build_pipeline(kind, seed, **dict(params))
    Xtr, ytr, Xva, yva = select_columns(tr), tr[TARGET], select_columns(va), va[TARGET]
    if kind == "xgboost" and HAVE_XGBOOST:
        derive, prep, clf = pipe["derive"], pipe["prep"], pipe["clf"]
        Xt = prep.fit_transform(derive.fit_transform(Xtr))
        clf.fit(Xt, ytr, eval_set=[(prep.transform(derive.transform(Xva)), yva)], verbose=False)
        params = {**params, "best_iteration": int(getattr(clf, "best_iteration", clf.n_estimators))}
    else:
        pipe.fit(Xtr, ytr)
    return pipe, params


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--population", default="active_90d", choices=["active_90d", "all_buyers"])
    p.add_argument("--tag", default=None, help="output sub-folder under models/ (default: the population name)")
    p.add_argument("--seed", type=int, default=C.get_settings().random_seed)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="max rows per partition (development)")
    args = p.parse_args(argv)
    C.configure_logging()
    tag = args.tag or args.population
    out_dir = C.PROJECT_ROOT / "models" / tag
    (out_dir / "candidates").mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    try:
        log.info("connected: %s", check_connection())
    except DatabaseError as exc:
        log.error("%s", exc)
        return 2

    t0 = time.perf_counter()
    train = load_labelled_snapshots(partition_snapshots("train"), args.population, args.limit)
    val = load_labelled_snapshots(partition_snapshots("validation"), args.population, args.limit)
    validate_temporal_split(train, val, None)            # chronological + purge gap, or refuse to train
    log.info("train: %s rows over %d snapshots, churn rate %.3f | validation: %s rows over %d snapshots, churn rate %.3f",
             f"{len(train):,}", train.snapshot_ts.nunique(), train[TARGET].mean(), f"{len(val):,}",
             val.snapshot_ts.nunique(), val[TARGET].mean())
    if train[TARGET].nunique() < 2 or val[TARGET].nunique() < 2:
        log.error("a partition contains a single class; nothing to learn")
        return 1
    yva = val[TARGET].to_numpy()

    results, fitted = [], {}
    # ---- baselines (no training; scored on validation for the record) ----
    for name, s in baseline_scores(val).items():
        m = ranking_metrics(yva, s)
        results.append({"model": name, "params": {}, **m})
        log.info("%-28s val ROC-AUC %.4f  PR-AUC %.4f  recall@10%% %.4f", name, m["roc_auc"], m["pr_auc"], m["recall_at_top10pct"])

    # ---- candidates ----
    for kind, grid in grids(args.quick).items():
        for params in grid:
            ts = time.perf_counter()
            pipe, used = fit_one(kind, params, args.seed, train, val)
            proba = pipe.predict_proba(select_columns(val))[:, 1]
            m = ranking_metrics(yva, proba)
            results.append({"model": kind, "params": used, "fit_seconds": round(time.perf_counter() - ts, 1), **m})
            fitted[(kind, json.dumps(used, sort_keys=True))] = (pipe, proba)
            log.info("%-14s %-70s val ROC-AUC %.4f PR-AUC %.4f (%.1fs)", kind, json.dumps(used)[:70], m["roc_auc"], m["pr_auc"],
                     time.perf_counter() - ts)

    # ---- best configuration per family (by validation PR-AUC), then the champion across families ----
    best = {}
    for r in results:
        if r["model"].startswith("baseline"):
            continue
        if r["model"] not in best or r["pr_auc"] > best[r["model"]]["pr_auc"]:
            best[r["model"]] = r
    champion_kind = max(best, key=lambda k: best[k]["pr_auc"])

    models = {}
    for kind, r in best.items():
        pipe, proba = fitted[(kind, json.dumps(r["params"], sort_keys=True))]
        thr = choose_threshold(yva, proba, C.BUSINESS)                      # threshold from validation outcomes only
        # Calibration decision made WITHOUT test data: fit isotonic on the earlier validation snapshots and
        # keep it only if it improves Brier score on the later ones (a time-ordered check inside validation).
        snaps = np.sort(val.snapshot_ts.unique())
        if len(snaps) >= 2:
            later = (val.snapshot_ts >= snaps[len(snaps) // 2]).to_numpy()
            probe = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(proba[~later], yva[~later])
            brier_raw, brier_cal = brier_score_loss(yva[later], proba[later]), brier_score_loss(yva[later], probe.predict(proba[later]))
            use_cal = brier_cal < 0.99 * brier_raw                            # require a >=1% improvement to bother
        else:                                                                 # cannot do a time-ordered check -> do not calibrate
            brier_raw = brier_cal = float("nan")
            use_cal = False
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(proba, yva) if use_cal else None
        log.info("%-14s calibration check on later validation snapshots: Brier raw %.5f vs isotonic %.5f -> %s", kind, brier_raw,
                 brier_cal, "ATTACH isotonic" if use_cal else "keep raw probabilities")
        meta = {"family": kind, "params": r["params"], "validation": {k: r[k] for k in r if k not in ("model", "params")},
                "validation_probability_metrics": probability_metrics(yva, proba), "threshold_selection": thr,
                "calibration_decision": {"brier_raw_later_snapshots": brier_raw, "brier_isotonic_later_snapshots": brier_cal,
                                         "isotonic_attached": bool(use_cal)}}
        models[kind] = ChurnModel(kind, pipe, cal, thr["threshold"], meta)
        joblib.dump(models[kind], out_dir / "candidates" / f"{kind}.joblib")
        log.info("%-14s best val PR-AUC %.4f | business threshold %.3f (contact-all value $%.0f -> tuned $%.0f on validation)",
                 kind, r["pr_auc"], thr["threshold"], thr["contact_all_value"], thr["validation_value"])
    joblib.dump(models[champion_kind], out_dir / "champion.joblib")

    try:
        fingerprint = read_sql_df("SELECT TOP 1 fingerprint, load_id FROM analytics.data_load_log ORDER BY load_id DESC").iloc[0].to_dict()
    except Exception:   # noqa: BLE001 - metadata only
        fingerprint = None
    metadata = {
        "tag": tag, "population": args.population, "champion": champion_kind, "selection_metric": "validation PR-AUC",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": args.seed, "quick": args.quick,
        "partitions": {
            "train": {"snapshots": sorted(train.snapshot_ts.dt.strftime("%Y-%m-%d").unique()), "rows": len(train), "churn_rate": float(train[TARGET].mean())},
            "validation": {"snapshots": sorted(val.snapshot_ts.dt.strftime("%Y-%m-%d").unique()), "rows": len(val), "churn_rate": float(val[TARGET].mean())},
            "test": "not loaded by this script"},
        "features": FEATURES, "business_assumptions": C.BUSINESS, "data_fingerprint": fingerprint,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
                     "xgboost": __import__("xgboost").__version__ if HAVE_XGBOOST else "NOT INSTALLED (HistGradientBoosting fallback)"},
        "all_runs": results, "best_per_family": {k: {"params": v["params"], "pr_auc": v["pr_auc"], "roc_auc": v["roc_auc"]} for k, v in best.items()},
        "total_seconds": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (C.PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    if not args.quick:     # smoke/test runs must not pollute the experiment log
        with open(C.PROJECT_ROOT / "reports" / "experiments.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({k: metadata[k] for k in ("tag", "population", "champion", "trained_at_utc", "seed", "quick", "best_per_family", "data_fingerprint")}, default=str) + "\n")
    log.info("champion: %s (validation PR-AUC %.4f, ROC-AUC %.4f) | saved to %s | %.0fs", champion_kind,
             best[champion_kind]["pr_auc"], best[champion_kind]["roc_auc"], out_dir, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
