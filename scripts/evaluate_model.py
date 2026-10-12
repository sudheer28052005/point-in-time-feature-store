"""Final evaluation on the untouched TEST partition + figures + predictions written to SQL Server.

    python scripts/evaluate_model.py [--tag active_90d] [--no-db-write]

The model was selected on validation data only. All candidates are reported on test for transparency, but
the champion was fixed BEFORE this script ran; test results never feed back into selection or thresholds.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from src import config as C  # noqa: E402
from src.database import DatabaseError, check_connection, get_engine  # noqa: E402
from src.evaluation import (calibration_table, campaign_value, probability_metrics, ranking_metrics,  # noqa: E402
                            threshold_metrics)
from src.feature_extraction import FEATURES, TARGET, load_labelled_snapshots, load_scoring_snapshot  # noqa: E402
from src.modeling import baseline_scores, partition_snapshots, select_columns, validate_temporal_split  # noqa: E402

log = logging.getLogger("evaluate_model")
PALETTE = {"xgboost": "#1f77b4", "random_forest": "#2ca02c", "logreg": "#ff7f0e", "baseline_recency_rule": "#7f7f7f",
           "baseline_no_order_30d_rule": "#bcbd22", "baseline_constant": "#d62728"}


def decile_table(y, p) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p})
    d["risk_decile"] = np.ceil(d.p.rank(method="first", pct=True) * 10).astype(int)       # 10 = highest risk
    t = d.groupby("risk_decile").agg(customers=("y", "size"), churners=("y", "sum"), churn_rate=("y", "mean"), mean_p=("p", "mean"))
    t = t.sort_index(ascending=False)
    t["cumulative_recall"] = t.churners.cumsum() / t.churners.sum()
    t["lift"] = t.churn_rate / d.y.mean()
    return t.reset_index()


def fig_roc_pr(y, scores: dict, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, s in scores.items():
        if np.ptp(s) == 0:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        pr, rc, _ = precision_recall_curve(y, s)
        lw = 2.4 if name == "xgboost" else 1.3
        ax[0].plot(fpr, tpr, label=name, color=PALETTE.get(name), lw=lw)
        ax[1].plot(rc, pr, label=name, color=PALETTE.get(name), lw=lw)
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[1].axhline(np.mean(y), color="k", ls=":", lw=1, label="prevalence")
    ax[0].set(title="ROC (test)", xlabel="false positive rate", ylabel="true positive rate")
    ax[1].set(title="Precision-recall (test)", xlabel="recall", ylabel="precision")
    ax[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_calibration(y, p_raw, p_cal, path, calibrated: bool):
    """Reliability curve; the isotonic curve is drawn only when a calibrator is actually attached to the model."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect")
    curves = [("raw (model output used as is)", p_raw, "#1f77b4")]
    if calibrated:
        curves.append(("isotonic (fit on validation)", p_cal, "#d62728"))
    for label, p, c in curves:
        t = calibration_table(y, p, 10)
        ax.plot(t.mean_predicted, t.observed_rate, "o-", label=label, color=c)
    ax.set(title="Calibration on test (decile bins)", xlabel="mean predicted churn probability", ylabel="observed churn rate")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_deciles(t: pd.DataFrame, prevalence: float, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(t.risk_decile.astype(str), t.churn_rate, color="#1f77b4")
    ax.axhline(prevalence, color="k", ls=":", label=f"overall churn rate {prevalence:.2f}")
    ax.set(title="Observed churn rate by predicted-risk decile (10 = highest), test", xlabel="risk decile", ylabel="churn rate")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_importance(imp: pd.DataFrame, path, top=15):
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(t.feature, t.importance_mean, xerr=t.importance_std, color="#1f77b4")
    ax.set(title=f"Permutation importance (drop in PR-AUC), validation, top {top}", xlabel="mean decrease in average precision")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_monthly(per_snap: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(per_snap.snapshot, per_snap.roc_auc, "o-", label="champion ROC-AUC")
    ax.plot(per_snap.snapshot, per_snap.recency_rule_roc_auc, "s--", label="recency rule ROC-AUC", color="#7f7f7f")
    ax.set(title="Stability across test snapshots", ylabel="ROC-AUC", ylim=(0.5, 0.85)); ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="active_90d")
    ap.add_argument("--no-db-write", action="store_true")
    args = ap.parse_args(argv)
    C.configure_logging()
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    mdir = C.PROJECT_ROOT / "models" / args.tag
    meta = json.loads((mdir / "metadata.json").read_text())
    pop = meta["population"]
    try:
        check_connection()
    except DatabaseError as exc:
        log.error("%s", exc)
        return 2
    champion_name = meta["champion"]
    models = {k.stem: joblib.load(k) for k in sorted((mdir / "candidates").glob("*.joblib"))}
    champ = joblib.load(mdir / "champion.joblib")
    figs = C.PROJECT_ROOT / "reports" / "figures" / args.tag
    figs.mkdir(parents=True, exist_ok=True)

    train_stub = pd.DataFrame({"snapshot_ts": pd.to_datetime(meta["partitions"]["train"]["snapshots"])})
    val = load_labelled_snapshots(partition_snapshots("validation"), pop)
    test = load_labelled_snapshots(partition_snapshots("test"), pop)           # <- the one and only load of the test partition
    validate_temporal_split(train_stub, val, test)
    y, yv = test[TARGET].to_numpy(), val[TARGET].to_numpy()
    log.info("test: %s rows over %d snapshots, churn rate %.3f", f"{len(test):,}", test.snapshot_ts.nunique(), y.mean())

    # ---------------- every model + baselines on test ----------------
    table, scores = {}, {}
    for name, s in baseline_scores(test).items():
        scores[name] = s
        table[name] = {"ranking": ranking_metrics(y, s)}
    table["baseline_no_order_30d_rule"]["at_threshold"] = threshold_metrics(y, scores["baseline_no_order_30d_rule"], 0.5)
    for name, m in models.items():
        raw = m.raw_proba(test); cal = m.predict_proba(test)
        scores[name] = raw
        table[name] = {"ranking": ranking_metrics(y, raw), "at_threshold": threshold_metrics(y, raw, m.threshold),
                       "probability_raw": probability_metrics(y, raw), "probability_calibrated": probability_metrics(y, cal),
                       "campaign_value": campaign_value(y, raw, m.threshold, C.BUSINESS),
                       "campaign_value_contact_all": campaign_value(y, raw, -np.inf, C.BUSINESS), "params": m.metadata["params"],
                       "validation": m.metadata["validation"]}
        r = table[name]["ranking"]
        log.info("%-14s test ROC-AUC %.4f PR-AUC %.4f recall@10%% %.4f | validation ROC-AUC %.4f PR-AUC %.4f", name,
                 r["roc_auc"], r["pr_auc"], r["recall_at_top10pct"], m.metadata["validation"]["roc_auc"], m.metadata["validation"]["pr_auc"])
    for name in ("baseline_constant", "baseline_recency_rule", "baseline_no_order_30d_rule"):
        r = table[name]["ranking"]
        log.info("%-28s test ROC-AUC %.4f PR-AUC %.4f recall@10%% %.4f", name, r["roc_auc"], r["pr_auc"], r["recall_at_top10pct"])

    raw, cal = champ.raw_proba(test), champ.predict_proba(test)
    deciles = decile_table(y, cal)

    # stability by snapshot and on customers never seen in training
    per_snap = []
    for ts, g in test.assign(p=raw, rec=baseline_scores(test)["baseline_recency_rule"]).groupby("snapshot_ts"):
        from sklearn.metrics import average_precision_score, roc_auc_score
        per_snap.append({"snapshot": ts.strftime("%Y-%m-%d"), "rows": len(g), "churn_rate": float(g[TARGET].mean()),
                         "roc_auc": float(roc_auc_score(g[TARGET], g.p)), "pr_auc": float(average_precision_score(g[TARGET], g.p)),
                         "recency_rule_roc_auc": float(roc_auc_score(g[TARGET], g.rec))})
    per_snap = pd.DataFrame(per_snap)
    train_users = set(load_labelled_snapshots(meta["partitions"]["train"]["snapshots"], pop)["user_id"])
    unseen = ~test.user_id.isin(train_users).to_numpy()
    unseen_metrics = ranking_metrics(y[unseen], raw[unseen]) if unseen.sum() > 100 and len(set(y[unseen])) == 2 else None

    # importances: permutation on validation (not test) + model-native gain for XGBoost
    rng = np.random.default_rng(meta["seed"])
    idx = rng.choice(len(val), size=min(15000, len(val)), replace=False)
    pi = permutation_importance(champ.pipeline, select_columns(val).iloc[idx], yv[idx], scoring="average_precision",
                                n_repeats=3, random_state=meta["seed"], n_jobs=1)
    imp = pd.DataFrame({"feature": FEATURES, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std}) \
        .sort_values("importance_mean", ascending=False).reset_index(drop=True)
    imp.to_csv(C.PROJECT_ROOT / "reports" / f"feature_importance_{args.tag}.csv", index=False)

    fig_roc_pr(y, scores, figs / "roc_pr.png")
    fig_calibration(y, raw, cal, figs / "calibration.png", calibrated=champ.calibrator is not None)
    fig_deciles(deciles, float(y.mean()), figs / "deciles.png")
    fig_importance(imp, figs / "feature_importance.png")
    fig_monthly(per_snap, figs / "stability.png")

    ch = table[champion_name]
    c_thr, c_tbl = ch["at_threshold"], ch["ranking"]
    metrics = {
        "tag": args.tag, "population": pop, "champion": champion_name, "test_rows": int(len(test)), "test_churn_rate": float(y.mean()),
        "test_snapshots": sorted(test.snapshot_ts.dt.strftime("%Y-%m-%d").unique()), "models": table,
        "champion_deciles": deciles.to_dict("records"), "per_snapshot": per_snap.to_dict("records"),
        "unseen_customers_only": unseen_metrics, "unseen_customer_rows": int(unseen.sum()),
        "permutation_importance_validation_top10": imp.head(10).to_dict("records"),
        "business": {"assumptions": C.BUSINESS, "threshold_raw_probability": champ.threshold,
                     "test_campaign_value_at_threshold": ch["campaign_value"], "test_value_contact_everyone": ch["campaign_value_contact_all"],
                     "test_value_contact_nobody": 0.0, "note": "illustrative parameters on synthetic data, not measured business results"},
        "calibration": {"raw": ch["probability_raw"], "isotonic": ch["probability_calibrated"]},
        "trained_at_utc": meta["trained_at_utc"], "data_fingerprint": meta["data_fingerprint"],
    }
    (C.PROJECT_ROOT / "reports" / f"metrics_{args.tag}.json").write_text(json.dumps(metrics, indent=2, default=str))

    rec = table["baseline_recency_rule"]["ranking"]
    log.info("CHAMPION %s | test ROC-AUC %.4f (recency rule %.4f) | PR-AUC %.4f (rule %.4f, prevalence %.4f) | recall@10%% %.4f | lift@10%% %.2f",
             champion_name, c_tbl["roc_auc"], rec["roc_auc"], c_tbl["pr_auc"], rec["pr_auc"], c_tbl["prevalence"],
             c_tbl["recall_at_top10pct"], c_tbl["lift_at_top10pct"])
    log.info("at the validation-chosen threshold %.3f: precision %.3f recall %.3f F1 %.3f, flags %.1f%% of customers | confusion %s",
             c_thr["threshold"], c_thr["precision"], c_thr["recall"], c_thr["f1"], 100 * c_thr["flagged_share"], c_thr["confusion_matrix"])
    log.info("calibration (Brier / ECE): raw %.4f / %.4f -> isotonic %.4f / %.4f", ch["probability_raw"]["brier"],
             ch["probability_raw"]["expected_calibration_error"], ch["probability_calibrated"]["brier"], ch["probability_calibrated"]["expected_calibration_error"])
    log.info("campaign value (illustrative $): tuned threshold %.0f | contact everyone %.0f", ch["campaign_value"], ch["campaign_value_contact_all"])
    if unseen_metrics:
        log.info("customers never seen in training (%s rows): ROC-AUC %.4f PR-AUC %.4f", f"{unseen.sum():,}", unseen_metrics["roc_auc"], unseen_metrics["pr_auc"])

    # ---------------- predictions -> SQL Server (dashboard input) ----------------
    if not args.no_db_write:
        version = f"{args.tag}-{champion_name}-{meta['trained_at_utc'][:10].replace('-', '')}"
        scoring = load_scoring_snapshot(C.SCORING_SNAPSHOT, pop)
        frames = []
        for part, df, probs, actual in (("test", test, cal, y), ("scoring", scoring, champ.predict_proba(scoring), None)):
            f = df[["snapshot_ts", "user_id"]].copy()
            f["p"] = probs
            f["decile"] = f.groupby("snapshot_ts")["p"].transform(lambda s: np.ceil(s.rank(method="first", pct=True) * 10)).astype(int)
            f["pred"] = (champ.raw_proba(df) >= champ.threshold).astype(int)
            f["actual"] = actual if actual is not None else None
            f["partition"] = part
            frames.append(f)
        allp = pd.concat(frames, ignore_index=True)
        rows = [(version, r.partition, r.snapshot_ts.to_pydatetime(), int(r.user_id), round(float(r.p), 6), int(r.decile), int(r.pred),
                 None if pd.isna(r.actual) else int(r.actual)) for r in allp.itertuples()]
        with get_engine().begin() as c:
            c.exec_driver_sql("DELETE FROM analytics.churn_predictions WHERE model_version = ?", (version,))
            c.exec_driver_sql("INSERT INTO analytics.churn_predictions (model_version, partition, snapshot_ts, user_id, churn_probability, "
                              "risk_decile, predicted_churn, actual_churn) VALUES (?,?,?,?,?,?,?,?)", rows)
        log.info("wrote %s predictions for model_version %s (test + scoring snapshot %s)", f"{len(rows):,}", version, C.SCORING_SNAPSHOT)
        metrics["model_version"] = version
        (C.PROJECT_ROOT / "reports" / f"metrics_{args.tag}.json").write_text(json.dumps(metrics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
