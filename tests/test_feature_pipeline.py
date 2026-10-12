"""Cross-check the T-SQL feature store against an independent pandas implementation.

The oracle below re-implements the temporal contract from scratch (no SQL reuse):
  history = ts < T0, outcome window = [T0, T0+30d), session gap rule = strictly > 30 minutes,
  order qualifies if completed, or refunded with refunded_at >= evaluation time.
It reads the RAW tables and compares to analytics.customer_features / churn_labels for every user.
"""
import numpy as np
import pandas as pd
import pytest

from src.database import read_sql_df

pytestmark = pytest.mark.db
SNAPSHOTS = ["2023-10-01", "2024-09-01", "2025-03-01"]


@pytest.fixture(scope="module")
def raw(db_ready):
    orders = read_sql_df("""SELECT order_id, user_id, order_ts, order_status, net_amount, refunded_at FROM dbo.orders""")
    events = read_sql_df("SELECT event_id, user_id, event_ts, event_type FROM dbo.raw_web_events")
    users = read_sql_df("SELECT user_id, signup_at FROM dbo.users")
    events = events.sort_values(["user_id", "event_ts", "event_id"], ignore_index=True)
    gap = events.groupby("user_id").event_ts.diff()
    events["session"] = ((gap.isna()) | (gap > pd.Timedelta(minutes=30))).groupby(events.user_id).cumsum()
    return orders, events, users


def qualifying(orders, lo, hi, asof):
    m = (orders.order_ts >= lo) & (orders.order_ts < hi) & (
        (orders.order_status == "completed") | ((orders.order_status == "refunded") & (orders.refunded_at >= asof)))
    return orders[m]


def oracle_features(raw, t0):
    orders, events, users = raw
    t0 = pd.Timestamp(t0)
    q = qualifying(orders, pd.Timestamp("1900-01-01"), t0, t0).sort_values(["user_id", "order_ts", "order_id"])
    q = q.assign(interval=q.groupby("user_id").order_ts.diff().dt.total_seconds() / 86400)
    out = pd.DataFrame(index=users.loc[users.signup_at < t0, "user_id"].rename("user_id"))
    g = q.groupby("user_id")
    out["lifetime_orders"] = g.size()
    out["lifetime_spend"] = g.net_amount.sum()
    out["last_purchase_ts"] = g.order_ts.max()
    for d in (30, 90, 365):
        w = q[q.order_ts >= t0 - pd.Timedelta(days=d)].groupby("user_id")
        out[f"orders_{d}d"], out[f"spend_{d}d"] = w.size(), w.net_amount.sum()
    prev = q[(q.order_ts >= t0 - pd.Timedelta(days=60)) & (q.order_ts < t0 - pd.Timedelta(days=30))]
    out["orders_prev_30d"] = prev.groupby("user_id").size()
    out["max_order_value"] = g.net_amount.max()
    out["std_order_value"] = g.net_amount.std(ddof=1)
    out["mean_interval"] = g.interval.mean()
    out["std_interval"] = g.interval.std(ddof=1)
    out["median_interval"] = g.interval.median()
    out["distinct_purchase_dates"] = g.order_ts.agg(lambda s: s.dt.date.nunique())
    e = events[(events.event_ts < t0) & (events.event_ts >= t0 - pd.Timedelta(days=90))]
    for d in (7, 30, 90):
        w = e[e.event_ts >= t0 - pd.Timedelta(days=d)].groupby("user_id")
        out[f"events_{d}d"], out[f"sessions_{d}d"] = w.size(), w.session.nunique()
    e30 = e[e.event_ts >= t0 - pd.Timedelta(days=30)]
    out["product_views_30d"] = e30[e30.event_type == "product_view"].groupby("user_id").size()
    out["purchase_sessions_30d"] = e30[e30.event_type == "purchase"].groupby("user_id").session.nunique()
    zero = ["lifetime_orders", "lifetime_spend", "orders_30d", "orders_90d", "orders_365d", "spend_30d", "spend_90d",
            "spend_365d", "orders_prev_30d", "distinct_purchase_dates", "events_7d", "events_30d", "events_90d",
            "sessions_7d", "sessions_30d", "sessions_90d", "product_views_30d", "purchase_sessions_30d"]
    out[zero] = out[zero].fillna(0)
    out["conversion_rate_30d"] = np.where(out.sessions_30d > 0, out.purchase_sessions_30d / out.sessions_30d.where(out.sessions_30d > 0), np.nan)
    return out


@pytest.mark.parametrize("t0", SNAPSHOTS)
def test_sql_features_match_independent_pandas_oracle(raw, t0):
    exp = oracle_features(raw, t0)
    got = read_sql_df("SELECT * FROM analytics.customer_features WHERE snapshot_ts = :t", {"t": t0}).set_index("user_id")
    assert len(got) == len(exp) and set(got.index) == set(exp.index)
    exp = exp.reindex(got.index)
    for sql_col, ora_col, tol in [
        ("lifetime_orders", "lifetime_orders", 0), ("orders_30d", "orders_30d", 0), ("orders_90d", "orders_90d", 0),
        ("orders_365d", "orders_365d", 0), ("orders_prev_30d", "orders_prev_30d", 0),
        ("lifetime_spend", "lifetime_spend", 0.005), ("spend_30d", "spend_30d", 0.005), ("spend_90d", "spend_90d", 0.005),
        ("spend_365d", "spend_365d", 0.005), ("max_order_value", "max_order_value", 0.005),
        ("std_order_value", "std_order_value", 0.011), ("distinct_purchase_dates", "distinct_purchase_dates", 0),
        ("mean_purchase_interval_days", "mean_interval", 0.011), ("std_purchase_interval_days", "std_interval", 0.011),
        ("median_purchase_interval_days", "median_interval", 0.011),
        ("events_7d", "events_7d", 0), ("events_30d", "events_30d", 0), ("events_90d", "events_90d", 0),
        ("sessions_7d", "sessions_7d", 0), ("sessions_30d", "sessions_30d", 0), ("sessions_90d", "sessions_90d", 0),
        ("product_views_30d", "product_views_30d", 0), ("conversion_rate_30d", "conversion_rate_30d", 0.0001),
    ]:
        a, b = got[sql_col].astype(float), exp[ora_col].astype(float)
        assert (a.isna() == b.isna()).all(), f"{sql_col}: NULL pattern differs"
        assert np.nanmax(np.abs(a - b)) <= tol + 1e-9, f"{sql_col}: max diff {np.nanmax(np.abs(a - b))}"
    # no feature may reference a purchase at or after the cutoff
    assert (got.last_purchase_ts.dropna() < pd.Timestamp(t0)).all()
    assert (got.last_purchase_ts.dropna() == exp.last_purchase_ts.dropna()).all()


@pytest.mark.parametrize("t0", SNAPSHOTS)
def test_sql_labels_match_oracle(raw, t0):
    orders, _, users = raw
    t0 = pd.Timestamp(t0)
    end = t0 + pd.Timedelta(days=30)
    buyers = set(qualifying(orders, pd.Timestamp("1900-01-01"), t0, t0).user_id)
    bought = set(qualifying(orders, t0, end, end).user_id)
    got = read_sql_df("SELECT user_id, churn_label, has_prior_purchase FROM analytics.churn_labels WHERE snapshot_ts = :t",
                      {"t": t0}).set_index("user_id")
    pop = users.loc[users.signup_at < t0, "user_id"]
    assert set(got.index) == set(pop)
    assert (got.churn_label.astype(bool) == ~got.index.isin(bought)).all()
    assert (got.has_prior_purchase.astype(bool) == got.index.isin(buyers)).all()


def test_scoring_snapshot_has_features_but_no_labels(db_ready):
    """The 2025-07-01 snapshot has no complete 30-day outcome window -> no labels may exist."""
    feats = read_sql_df("SELECT COUNT(*) n FROM analytics.customer_features WHERE snapshot_ts = '2025-07-01'").n[0]
    labels = read_sql_df("SELECT COUNT(*) n FROM analytics.churn_labels WHERE snapshot_ts > '2025-06-01'").n[0]
    assert feats > 0 and labels == 0


def test_label_procedure_refuses_incomplete_window(db_ready):
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError
    from src.database import get_engine
    with pytest.raises(DBAPIError, match="incomplete"):
        with get_engine().begin() as c:
            c.execute(text("EXEC analytics.usp_build_churn_labels @T0 = '2025-06-15', @observation_end = '2025-07-01'"))
    assert read_sql_df("SELECT COUNT(*) n FROM analytics.churn_labels WHERE snapshot_ts = '2025-06-15'").n[0] == 0


def test_derived_data_quality_checks_all_zero(db_ready):
    from src.validation import run_data_quality_checks
    df = run_data_quality_checks(derived=True)
    assert len(df) >= 20 and df[df.violations != 0].empty, df.to_string()
