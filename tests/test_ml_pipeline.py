"""ML pipeline tests: leakage guards, split rules, metric correctness, model plumbing, and DB integration."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.evaluation import (calibration_table, campaign_value, choose_threshold, expected_calibration_error,
                            expected_value_per_contact, lift_at_top, precision_at_top, recall_at_top, threshold_metrics)
from src.feature_extraction import (CATEGORICAL_FEATURES, FEATURES, FORBIDDEN_AS_FEATURES, NUMERIC_FEATURES,
                                    assert_no_leakage)
from src.modeling import (ChurnModel, add_derived_features, baseline_scores, build_pipeline, partition_snapshots,
                          validate_temporal_split)


# ------------------------------- leakage guards -------------------------------
def test_feature_list_contains_no_forbidden_or_label_like_columns():
    assert not set(FEATURES) & FORBIDDEN_AS_FEATURES
    assert len(FEATURES) == len(set(FEATURES)) and set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert_no_leakage(FEATURES)


@pytest.mark.parametrize("bad", ["churn_label", "qualifying_orders_in_window", "outcome_window_end", "user_id", "snapshot_ts",
                                 "last_purchase_ts", "feature_generated_at", "future_churn_flag", "orders_label"])
def test_leakage_guard_rejects_dangerous_columns(bad):
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(FEATURES + [bad])


def test_training_script_never_loads_the_test_partition():
    """Static guard: only evaluate_model.py may read the test partition."""
    train_src = (C.PROJECT_ROOT / "scripts" / "train_model.py").read_text(encoding="utf-8")
    eval_src = (C.PROJECT_ROOT / "scripts" / "evaluate_model.py").read_text(encoding="utf-8")
    assert 'partition_snapshots("test")' not in train_src and "SCORING_SNAPSHOT" not in train_src
    assert 'partition_snapshots("test")' in eval_src


# ------------------------------- temporal split -------------------------------
def frame(*dates):
    return pd.DataFrame({"snapshot_ts": pd.to_datetime(list(dates))})


def test_configured_splits_are_chronological_and_purged():
    tr, va, te = (frame(*partition_snapshots(n)) for n in ("train", "validation", "test"))
    validate_temporal_split(tr, va, te)                                   # must not raise
    assert tr.snapshot_ts.max() + pd.Timedelta(days=30) <= va.snapshot_ts.min()
    assert va.snapshot_ts.max() + pd.Timedelta(days=30) <= te.snapshot_ts.min()
    assert len(tr) == 15 and len(va) == 4 and len(te) == 6


def test_split_with_exactly_one_outcome_window_gap_is_accepted():
    validate_temporal_split(frame("2024-01-01"), frame("2024-01-31"), None)


def test_split_overlapping_an_outcome_window_is_rejected():
    with pytest.raises(ValueError, match="temporal leakage"):
        validate_temporal_split(frame("2024-01-01"), frame("2024-01-30"), None)      # 1 day short of the 30-day label window


def test_split_in_wrong_order_or_with_shared_snapshot_is_rejected():
    with pytest.raises(ValueError):
        validate_temporal_split(frame("2024-06-01"), frame("2024-01-01"), None)
    with pytest.raises(ValueError):
        validate_temporal_split(frame("2024-01-01", "2024-03-01"), frame("2024-03-01"), None)


# ------------------------------- metrics -------------------------------
def test_recall_precision_lift_at_top_k_hand_computed():
    y = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 0])
    s = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .0])
    assert recall_at_top(y, s, 0.2) == pytest.approx(2 / 4)       # top 2 rows hold 2 of the 4 churners
    assert precision_at_top(y, s, 0.2) == pytest.approx(1.0)
    assert lift_at_top(y, s, 0.2) == pytest.approx(1.0 / 0.4)
    assert recall_at_top(y, s, 1.0) == 1.0


def test_top_k_is_deterministic_under_ties():
    y = np.array([1, 0, 1, 0]); s = np.zeros(4)
    assert recall_at_top(y, s, 0.5) == recall_at_top(y, s, 0.5) == pytest.approx(0.5)   # stable order -> first two rows


def test_calibration_error_is_zero_when_perfectly_calibrated_and_large_when_not():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 100_000); y = rng.binomial(1, p)
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, np.clip(p * 0.5, 0, 1)) > 0.15
    assert len(calibration_table(y, p, 10)) == 10


def test_threshold_metrics_confusion_matrix():
    y = np.array([1, 1, 0, 0, 1]); p = np.array([.9, .4, .6, .1, .8])
    m = threshold_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3) and m["flagged_share"] == pytest.approx(0.6)


def test_business_threshold_recovers_the_analytic_break_even_probability():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 300_000); y = rng.binomial(1, p)               # perfectly calibrated scores
    res = choose_threshold(y, p, C.BUSINESS)
    assert res["threshold"] == pytest.approx(res["breakeven_probability"], abs=0.03)
    assert expected_value_per_contact(np.array([res["breakeven_probability"]]), C.BUSINESS)[0] == pytest.approx(0, abs=1e-9)
    assert res["validation_value"] >= max(res["contact_all_value"], res["contact_none_value"])
    assert campaign_value(y, p, 2.0, C.BUSINESS) == 0.0                   # nobody contacted -> $0


# ------------------------------- model plumbing -------------------------------
def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(5, 2, n).clip(0) for c in NUMERIC_FEATURES})
    df["country"] = rng.choice(["US", "GB", "DE"], n); df["acquisition_channel"] = rng.choice(["organic", "email"], n)
    df["customer_segment"] = rng.choice(["budget", "premium"], n); df["rfm_segment"] = rng.choice(["Champions", "At Risk"], n)
    df.loc[rng.random(n) < 0.2, ["mean_purchase_interval_days", "std_order_value"]] = np.nan          # undefined stats
    df["churn_label"] = (rng.random(n) < 1 / (1 + np.exp(-(df.days_since_last_purchase - 5)))).astype(int)
    df["snapshot_ts"] = pd.Timestamp("2024-01-01"); df["user_id"] = np.arange(n)
    return df


def test_derived_overdue_ratio_is_undefined_not_zero_when_interval_missing_or_zero():
    X = pd.DataFrame({"days_since_last_purchase": [10.0, 10.0, 10.0, np.nan], "mean_purchase_interval_days": [5.0, np.nan, 0.0, 5.0]})
    r = add_derived_features(X)["overdue_ratio"]
    assert r.iloc[0] == 2.0 and r.iloc[1:].isna().all()


@pytest.mark.parametrize("kind,params", [("logreg", {"C": 0.1}), ("random_forest", {"n_estimators": 20, "min_samples_leaf": 5}),
                                         ("xgboost", {"n_estimators": 20, "max_depth": 3})])
def test_each_model_fits_handles_missing_values_and_unseen_categories(kind, params):
    df = synthetic()
    from src.modeling import select_columns
    pipe = build_pipeline(kind, seed=0, **params).fit(select_columns(df), df.churn_label)
    new = synthetic(50, seed=9)
    new.loc[:5, "country"] = "ZZ"                                           # category never seen in training
    new.loc[:5, "days_since_last_purchase"] = np.nan                         # missing value at prediction time
    p = pipe.predict_proba(select_columns(new))[:, 1]
    assert p.shape == (50,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_churn_model_bundle_applies_threshold_to_raw_probability():
    df = synthetic()
    from src.modeling import select_columns
    pipe = build_pipeline("logreg", seed=0, C=0.1).fit(select_columns(df), df.churn_label)
    m = ChurnModel("t", pipe, None, 0.5, {})
    assert ((m.raw_proba(df) >= 0.5).astype(int) == m.predict(df)).all()
    assert np.allclose(m.predict_proba(df), m.raw_proba(df))                  # no calibrator -> raw


def test_baselines_are_scores_where_higher_means_more_churn():
    df = synthetic(20)
    b = baseline_scores(df)
    assert set(b) == {"baseline_constant", "baseline_recency_rule", "baseline_no_order_30d_rule"}
    assert np.ptp(b["baseline_constant"]) == 0
    assert b["baseline_recency_rule"].argmax() == df.days_since_last_purchase.to_numpy().argmax()


# ------------------------------- database integration -------------------------------
@pytest.mark.db
def test_extraction_returns_only_requested_labelled_active_snapshots(db_ready):
    from src.feature_extraction import load_labelled_snapshots
    snaps = ["2024-05-01", "2024-06-01"]
    df = load_labelled_snapshots(snaps)
    assert set(df.snapshot_ts.dt.strftime("%Y-%m-%d")) == set(snaps)
    assert (df.orders_90d > 0).all()                                          # the 90-day training population
    assert (FORBIDDEN_AS_FEATURES - {"user_id", "snapshot_ts", "churn_label"}).isdisjoint(df.columns)   # nothing outcome-like is extracted
    assert all(str(df[c].dtype) == "float64" for c in NUMERIC_FEATURES)       # DECIMAL/BIT coerced explicitly
    assert df.churn_label.isin([0, 1]).all() and not df.duplicated(["snapshot_ts", "user_id"]).any()
    assert len(load_labelled_snapshots(snaps, limit=100)) == 100
    with pytest.raises(ValueError):
        load_labelled_snapshots(snaps, population="everyone")


@pytest.mark.db
def test_unlabelled_snapshot_is_not_extractable_as_training_data(db_ready):
    from src.feature_extraction import load_labelled_snapshots, load_scoring_snapshot
    assert load_labelled_snapshots([C.SCORING_SNAPSHOT]).empty                # no complete outcome window -> no rows
    s = load_scoring_snapshot(C.SCORING_SNAPSHOT)
    assert len(s) > 1000 and "churn_label" not in s.columns and (s.orders_90d > 0).all()


@pytest.mark.db
def test_training_is_reproducible_for_a_fixed_seed(db_ready):
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_model", C.PROJECT_ROOT / "scripts" / "train_model.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    out = []
    for tag in ("pytest_repro_a", "pytest_repro_b"):
        assert mod.main(["--quick", "--tag", tag, "--limit", "4000", "--seed", "7"]) == 0
        out.append(json.loads((C.PROJECT_ROOT / "models" / tag / "metadata.json").read_text())["best_per_family"])
        shutil.rmtree(C.PROJECT_ROOT / "models" / tag)
    assert out[0] == out[1]


@pytest.fixture(scope="module")
def champion_meta():
    p = C.PROJECT_ROOT / "models" / "active_90d" / "metadata.json"
    if not p.exists():
        pytest.skip("run scripts/train_model.py first")
    return json.loads(p.read_text())


def test_trained_model_metadata_documents_a_leak_free_experiment(champion_meta):
    m = champion_meta
    assert m["partitions"]["test"] == "not loaded by this script"
    tr, va = m["partitions"]["train"]["snapshots"], m["partitions"]["validation"]["snapshots"]
    validate_temporal_split(frame(*tr), frame(*va), None)
    assert set(m["features"]) == set(FEATURES) and m["selection_metric"] == "validation PR-AUC"
    assert m["champion"] in m["best_per_family"]
    assert m["best_per_family"][m["champion"]]["pr_auc"] == max(v["pr_auc"] for v in m["best_per_family"].values())


@pytest.mark.db
def test_stored_predictions_are_consistent_with_the_label_store(db_ready, champion_meta):
    from src.database import read_sql_df
    version = f"{champion_meta['tag']}-{champion_meta['champion']}-{champion_meta['trained_at_utc'][:10].replace('-', '')}"
    p = read_sql_df("SELECT partition, COUNT(*) n, MIN(CAST(churn_probability AS FLOAT)) lo, MAX(CAST(churn_probability AS FLOAT)) hi, "
                    "MIN(risk_decile) dmin, MAX(risk_decile) dmax, SUM(CASE WHEN actual_churn IS NULL THEN 1 ELSE 0 END) null_actual "
                    "FROM analytics.churn_predictions WHERE model_version = :v GROUP BY partition", {"v": version}).set_index("partition")
    if p.empty:
        pytest.skip("run scripts/evaluate_model.py first")
    assert p.loc["test", "null_actual"] == 0 and p.loc["scoring", "null_actual"] == p.loc["scoring", "n"]
    assert (p.lo >= 0).all() and (p.hi <= 1).all() and (p.dmin == 1).all() and (p.dmax == 10).all()
    bad = read_sql_df("SELECT COUNT(*) n FROM analytics.churn_predictions c JOIN analytics.churn_labels l "
                      "ON l.snapshot_ts = c.snapshot_ts AND l.user_id = c.user_id "
                      "WHERE c.model_version = :v AND c.partition = 'test' AND CAST(c.actual_churn AS INT) <> CAST(l.churn_label AS INT)",
                      {"v": version}).n[0]
    assert bad == 0
    scoring_ts = read_sql_df("SELECT DISTINCT snapshot_ts FROM analytics.churn_predictions WHERE model_version = :v AND partition = 'scoring'",
                             {"v": version}).snapshot_ts
    assert list(scoring_ts) == [pd.Timestamp(C.SCORING_SNAPSHOT)]
