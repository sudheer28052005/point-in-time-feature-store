"""Churn-label contract: churn_label = 1 <=> no qualifying purchase in [T0, T0 + 30 days).

Labels exist only for COMPLETE outcome windows (T0 + 30d <= observation_end).
"""
import pandas as pd
import pytest
from sqlalchemy.exc import DBAPIError

pytestmark = pytest.mark.db
T0 = "2024-06-01 00:00:00"
OBS_END = "2025-01-01"          # far later: every window below is complete


def label_of(scn, **order_kwargs):
    scn.reset()
    scn.add_user(1)
    scn.add_order(1, **order_kwargs)
    return scn.build_labels(T0, OBS_END).loc[1]


# ---------------------------- window boundaries ----------------------------
def test_purchase_exactly_at_cutoff_counts_as_retained(scn):           # window start is inclusive
    assert label_of(scn, ts="2024-06-01 00:00:00.000").churn_label == 0


def test_purchase_one_ms_before_cutoff_is_history_not_outcome(scn):
    r = label_of(scn, ts="2024-05-31 23:59:59.999")
    assert (r.churn_label, r.qualifying_orders_in_window, r.has_prior_purchase) == (1, 0, 1)


def test_purchase_last_ms_of_window_counts(scn):
    assert label_of(scn, ts="2024-06-30 23:59:59.999").churn_label == 0


def test_purchase_exactly_30_days_after_cutoff_is_outside_the_window(scn):   # window end is exclusive
    r = label_of(scn, ts="2024-07-01 00:00:00.000")
    assert r.churn_label == 1 and r.outcome_window_end == pd.Timestamp("2024-07-01")


def test_window_length_is_30_days_exactly(scn):
    scn.add_user(1)
    r = scn.build_labels("2024-02-01", OBS_END).loc[1]        # Feb 2024 start + 30d = Mar 2 (leap year)
    assert (r.outcome_window_end - r.outcome_window_start) == pd.Timedelta(days=30)


# ---------------------------- what qualifies ----------------------------
def test_cancelled_order_in_window_does_not_prevent_churn(scn):
    assert label_of(scn, ts="2024-06-10", status="cancelled").churn_label == 1


def test_refunded_inside_window_does_not_count(scn):
    r = label_of(scn, ts="2024-06-10", status="refunded", refunded_at="2024-06-20")
    assert (r.churn_label, r.qualifying_orders_in_window) == (1, 0)


def test_refunded_after_window_closes_still_counts(scn):
    r = label_of(scn, ts="2024-06-10", status="refunded", refunded_at="2024-07-15")
    assert (r.churn_label, r.qualifying_orders_in_window) == (0, 1)


def test_refunded_exactly_when_window_closes_counts(scn):
    assert label_of(scn, ts="2024-06-10", status="refunded", refunded_at="2024-07-01 00:00:00").churn_label == 0


def test_multiple_orders_in_window_count_once_as_retained(scn):
    scn.add_user(1)
    for d in ("2024-06-02", "2024-06-03", "2024-06-04"):
        scn.add_order(1, d)
    r = scn.build_labels(T0, OBS_END).loc[1]
    assert (r.churn_label, r.qualifying_orders_in_window) == (0, 3)


# ---------------------------- population ----------------------------
def test_new_customers_and_non_purchasers_are_labelled_and_flagged(scn):
    scn.add_users([1, 2, 3])
    scn.add_order(2, "2024-05-01")               # prior buyer, silent in window
    scn.add_order(3, "2024-06-15")               # first-ever purchase falls in the window
    lab = scn.build_labels(T0, OBS_END)
    assert lab.loc[1].to_dict()["churn_label"] == 1 and lab.loc[1].has_prior_purchase == 0   # never bought
    assert lab.loc[2].churn_label == 1 and lab.loc[2].has_prior_purchase == 1
    assert lab.loc[3].churn_label == 0 and lab.loc[3].has_prior_purchase == 0                # converted in window


def test_users_registered_at_or_after_cutoff_get_no_label(scn):
    scn.add_user(1, "2024-05-31 23:59:59.999")
    scn.add_user(2, T0)
    scn.add_user(3, "2024-06-10")
    scn.add_order(3, "2024-06-15")
    assert scn.build_labels(T0, OBS_END).index.tolist() == [1]


# ---------------------------- complete windows only ----------------------------
def test_incomplete_outcome_window_is_rejected_and_writes_nothing(scn):
    scn.add_user(1)
    with pytest.raises(DBAPIError, match="incomplete"):
        scn.build_labels(T0, "2024-06-30 23:59:59.999")           # window needs data through 2024-07-01
    assert scn.labels(T0).empty


def test_window_that_ends_exactly_at_observation_end_is_complete(scn):
    scn.add_user(1)
    assert scn.build_labels(T0, "2024-07-01 00:00:00").loc[1].churn_label == 1


def test_orchestrator_builds_features_only_for_incomplete_window_and_clears_stale_labels(scn):
    scn.add_user(1)
    scn.add_order(1, "2024-05-10")
    scn.build_sessions()
    scn.exec("EXEC analytics.usp_build_feature_snapshot @T0=:t, @observation_end=:e", t=T0, e=OBS_END)
    assert len(scn.labels(T0)) == 1 and len(scn.features(T0)) == 1
    scn.exec("EXEC analytics.usp_build_feature_snapshot @T0=:t, @observation_end=:e", t=T0, e="2024-06-15")
    assert len(scn.features(T0)) == 1 and scn.labels(T0).empty        # no label from an incomplete window, stale one removed


def test_snapshot_grid_labels_only_complete_months(scn):
    scn.add_user(1, "2023-01-01")
    scn.add_order(1, "2024-01-20")
    scn.build_sessions()
    scn.exec("EXEC analytics.usp_build_feature_store @first_cutoff='2023-12-01', @last_cutoff='2024-03-01', @observation_end='2024-03-01'")
    snaps = scn.q("SELECT DISTINCT snapshot_ts FROM analytics.customer_features ORDER BY 1").snapshot_ts.dt.strftime("%Y-%m-%d").tolist()
    labelled = scn.q("SELECT DISTINCT snapshot_ts FROM analytics.churn_labels ORDER BY 1").snapshot_ts.dt.strftime("%Y-%m-%d").tolist()
    assert snaps == ["2023-12-01", "2024-01-01", "2024-02-01", "2024-03-01"]
    # 2024-01-01 + 30d = 2024-01-31 (complete). 2024-02-01 + 30d = 2024-03-02 (leap year) > obs end 2024-03-01
    # -> that snapshot is scoring-only, even though its calendar month looks "complete".
    assert labelled == ["2023-12-01", "2024-01-01"]


# ---------------------------- features and labels stay separate ----------------------------
def test_features_do_not_change_when_the_outcome_changes(scn):
    scn.add_user(1)
    scn.add_order(1, "2024-05-10")
    scn.build_sessions()
    scn.exec("EXEC analytics.usp_build_customer_features @T0=:t", t=T0)
    before = scn.features(T0).drop(columns="feature_generated_at")
    scn.add_order(1, "2024-06-12")                                       # the outcome: customer buys inside the window
    assert scn.build_labels(T0, OBS_END).loc[1].churn_label == 0
    scn.exec("EXEC analytics.usp_build_customer_features @T0=:t", t=T0)
    pd.testing.assert_frame_equal(scn.features(T0).drop(columns="feature_generated_at"), before)


def test_training_view_contains_only_labelled_snapshots_and_flags_the_90_day_population(scn):
    scn.add_users([1, 2, 3])
    scn.add_order(1, "2024-05-20")                # bought in last 90d      -> in population
    scn.add_order(2, "2023-12-01")                # bought, but > 90d ago   -> out
    scn.build_sessions()
    scn.exec("EXEC analytics.usp_build_feature_snapshot @T0=:t, @observation_end=:e", t=T0, e=OBS_END)
    scn.exec("EXEC analytics.usp_build_feature_snapshot @T0=:t, @observation_end=:e", t="2024-12-20", e=OBS_END)   # window incomplete
    v = scn.q("SELECT snapshot_ts, user_id, in_training_population, churn_label FROM analytics.vw_training_set ORDER BY user_id")
    assert set(v.snapshot_ts) == {pd.Timestamp(T0)}                       # unlabelled snapshot is invisible to training
    assert v.set_index("user_id").in_training_population.tolist() == [True, False, False]
