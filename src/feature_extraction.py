"""Read point-in-time feature snapshots from SQL Server into pandas.

All relational work (sessionization, windows, joins, quintiles, labels) already happened inside SQL Server;
Python only SELECTs the finished feature rows for the requested snapshots - never raw events/orders.

Large extractions: `chunksize` streams the result set in pieces, so a full-history pull never needs more than
one chunk in flight plus the final frame. For tables far beyond RAM, extract per snapshot (one query per
snapshot_ts, which is the clustered-key prefix) and write each to Parquet/CSV, or use bcp/BULK export.
"""
from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd
from sqlalchemy import bindparam, text

from .config import Settings
from .database import get_engine

log = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "account_age_days", "days_since_last_purchase",
    "orders_30d", "orders_90d", "orders_365d", "lifetime_orders", "orders_prev_30d", "order_count_change_30d",
    "spend_30d", "spend_90d", "spend_365d", "lifetime_spend",
    "avg_order_value", "max_order_value", "std_order_value",
    "mean_purchase_interval_days", "std_purchase_interval_days", "median_purchase_interval_days", "distinct_purchase_dates",
    "sessions_7d", "sessions_30d", "sessions_90d", "events_7d", "events_30d", "events_90d",
    "product_views_30d", "cart_additions_30d", "checkout_starts_30d", "conversion_rate_30d",
    "recency_quintile", "frequency_quintile", "spend_quintile",
    "is_marketing_opted_in", "has_web_events",
]
CATEGORICAL_FEATURES = ["country", "acquisition_channel", "customer_segment", "rfm_segment"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
KEY_COLUMNS = ["snapshot_ts", "user_id"]
TARGET = "churn_label"

# Columns that must NEVER be model inputs: identifiers, the label, outcome-window details, and raw timestamps
# (the model sees days_since_last_purchase instead) / lineage columns that identify the build, not the customer.
FORBIDDEN_AS_FEATURES = {
    "user_id", "snapshot_ts", "churn_label", "qualifying_orders_in_window", "outcome_window_end", "outcome_window_start",
    "has_prior_purchase", "in_training_population", "last_purchase_ts", "feature_generated_at", "label_generated_at",
    "observation_end",
}


def assert_no_leakage(columns: Sequence[str]) -> None:
    bad = sorted(set(columns) & FORBIDDEN_AS_FEATURES)
    suspicious = [c for c in columns if any(w in c.lower() for w in ("churn", "outcome", "label", "window_end"))]
    if bad or suspicious:
        raise ValueError(f"leakage-prone columns in feature list: {bad + suspicious}")


assert_no_leakage(FEATURES)   # import-time guard: the feature list itself can never regress silently

_POPULATIONS = {
    "active_90d": "in_training_population = 1",          # primary: bought in the 90 days before the cutoff
    "all_buyers": "has_prior_purchase = 1",              # secondary experiment
}


def load_labelled_snapshots(
    snapshots: Sequence[str],
    population: str = "active_90d",
    limit: int | None = None,
    chunksize: int = 50_000,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Features + label for the given snapshot dates (labelled snapshots only, via analytics.vw_training_set)."""
    if population not in _POPULATIONS:
        raise ValueError(f"population must be one of {sorted(_POPULATIONS)}")
    cols = ", ".join(KEY_COLUMNS + FEATURES + [TARGET])           # explicit column list from constants, never SELECT *
    top = "TOP (:limit) " if limit else ""
    sql = text(f"SELECT {top}{cols} FROM analytics.vw_training_set "
               f"WHERE snapshot_ts IN :snaps AND {_POPULATIONS[population]} "
               f"ORDER BY snapshot_ts, user_id").bindparams(bindparam("snaps", expanding=True))
    params = {"snaps": list(snapshots)} | ({"limit": limit} if limit else {})
    with get_engine(settings).connect() as conn:
        parts = list(pd.read_sql_query(sql, conn, params=params, chunksize=chunksize))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEY_COLUMNS + FEATURES + [TARGET])
    return _coerce_types(df)


def load_scoring_snapshot(snapshot_ts: str, population: str = "active_90d", settings: Settings | None = None) -> pd.DataFrame:
    """Features only (no label) for a snapshot whose outcome window is not complete yet."""
    cond = {"active_90d": "orders_90d > 0", "all_buyers": "has_purchased = 1"}[population]
    cols = ", ".join(KEY_COLUMNS + FEATURES)
    sql = text(f"SELECT {cols} FROM analytics.customer_features WHERE snapshot_ts = :t AND {cond} ORDER BY user_id")
    with get_engine(settings).connect() as conn:
        df = pd.read_sql_query(sql, conn, params={"t": snapshot_ts})
    return _coerce_types(df)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """DECIMAL columns arrive as Decimal objects and BIT as bool; make everything float/int explicitly."""
    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype("string")
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype("int8")
    return df


def snapshots_in_range(lo: str, hi: str) -> list[str]:
    """First-of-month cutoffs from lo to hi inclusive (the project's snapshot grid)."""
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(lo, hi, freq="MS")]
