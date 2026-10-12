"""Data access for the dashboard: read-only, parameterised, Streamlit-free (so it can be unit-tested).

Read-only is enforced twice: (1) every statement must pass `assert_select_only` here, and (2) the optional
`dashboard_reader` role (sql/15) denies writes at the database. Set DASHBOARD_SQL_USERNAME / DASHBOARD_SQL_PASSWORD
to connect as a restricted SQL login instead of your Windows account. Only synthetic customer ids are shown;
the queries never touch date_of_birth or raw clickstream.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src import config as C
from src.database import read_sql_df

_FORBIDDEN = re.compile(r"\b(insert|update|delete|merge|drop|alter|create|truncate|exec|execute|grant|revoke|deny|into|backup|restore|shutdown|dbcc)\b", re.I)


def assert_select_only(sql: str) -> None:
    """Reject anything that is not a single read-only SELECT/WITH statement."""
    body = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.S).strip().rstrip(";").strip()
    if not re.match(r"(?is)^(select|with)\b", body):
        raise PermissionError("dashboard queries must start with SELECT or WITH")
    if ";" in body:
        raise PermissionError("multiple statements are not allowed")
    hit = _FORBIDDEN.search(body)
    if hit:
        raise PermissionError(f"forbidden keyword in dashboard query: {hit.group(0)}")


def _settings() -> C.Settings:
    s = C.get_settings()
    user = os.getenv("DASHBOARD_SQL_USERNAME", "").strip()
    if user:
        s = dataclasses.replace(s, username=user, password=os.getenv("DASHBOARD_SQL_PASSWORD", ""))
    return s


def q(sql: str, params: Mapping[str, Any] | None = None) -> pd.DataFrame:
    assert_select_only(sql)
    return read_sql_df(sql, params, settings=_settings())


def _dt(d: date | datetime) -> datetime:
    return datetime(d.year, d.month, d.day)


# --------------------------------------------------------------------------- overview
def data_bounds() -> dict:
    r = q("SELECT (SELECT CAST(MIN(signup_at) AS DATE) FROM dbo.users) AS first_signup, "
          "(SELECT CAST(MAX(order_ts) AS DATE) FROM dbo.orders) AS last_order").iloc[0]
    return {"min": r.first_signup, "max": r.last_order}


def kpis(date_from: date, date_to_inclusive: date) -> dict:
    """Customers registered by the end date + qualifying orders/revenue inside [from, to].

    'Qualifying' uses the project's single rule (analytics.fn_qualifying_orders): completed orders, plus refunded
    orders only while the refund had not happened yet at the end date.
    """
    end = _dt(date_to_inclusive) + pd.Timedelta(days=1)          # half-open upper bound
    r = q("SELECT (SELECT COUNT(*) FROM dbo.users WHERE signup_at < :e) AS customers, "
          "COUNT_BIG(*) AS orders, ISNULL(SUM(net_amount), 0) AS revenue "
          "FROM analytics.fn_qualifying_orders(:f, :e, :e)", {"f": _dt(date_from), "e": end}).iloc[0]
    orders, revenue = int(r.orders), float(r.revenue)
    return {"customers": int(r.customers), "orders": orders, "revenue": revenue, "aov": revenue / orders if orders else float("nan")}


def monthly_activity(date_from: date, date_to_inclusive: date) -> pd.DataFrame:
    end = _dt(date_to_inclusive) + pd.Timedelta(days=1)
    df = q("SELECT DATEFROMPARTS(YEAR(order_ts), MONTH(order_ts), 1) AS month, COUNT_BIG(*) AS orders, "
           "COUNT(DISTINCT user_id) AS active_customers, SUM(net_amount) AS revenue "
           "FROM analytics.fn_qualifying_orders(:f, :e, :e) GROUP BY DATEFROMPARTS(YEAR(order_ts), MONTH(order_ts), 1) ORDER BY month",
           {"f": _dt(date_from), "e": end})
    signups = q("SELECT DATEFROMPARTS(YEAR(signup_at), MONTH(signup_at), 1) AS month, COUNT(*) AS new_customers FROM dbo.users "
                "WHERE signup_at >= :f AND signup_at < :e GROUP BY DATEFROMPARTS(YEAR(signup_at), MONTH(signup_at), 1)",
                {"f": _dt(date_from), "e": end})
    out = df.merge(signups, on="month", how="outer").sort_values("month")
    out["month"] = pd.to_datetime(out["month"])
    out["revenue"] = out["revenue"].astype(float)
    return out.fillna({"orders": 0, "active_customers": 0, "revenue": 0.0, "new_customers": 0})


def customers_by_channel() -> pd.DataFrame:
    return q("SELECT acquisition_channel, COUNT(*) AS customers FROM dbo.users GROUP BY acquisition_channel ORDER BY customers DESC")


# --------------------------------------------------------------------------- retention
def cohort_long(definition: str = "purchase") -> pd.DataFrame:
    df = q("SELECT cohort_month, months_since_acquisition, cohort_size, active_customers, retention_pct, is_observed "
           "FROM analytics.cohort_retention_long WHERE activity_definition = :d", {"d": definition})
    df["cohort_month"] = pd.to_datetime(df["cohort_month"])
    df["retention_pct"] = df["retention_pct"].astype(float)
    return df


def cohort_matrix(definition: str = "purchase") -> pd.DataFrame:
    """Triangular matrix: NaN = not yet observed; 0.0 = observed with no activity."""
    long = cohort_long(definition)
    return long.pivot(index="cohort_month", columns="months_since_acquisition", values="retention_pct").sort_index()


# --------------------------------------------------------------------------- segments and risk
def snapshots() -> list[str]:
    df = q("SELECT DISTINCT snapshot_ts FROM analytics.customer_features ORDER BY snapshot_ts")
    return [pd.Timestamp(t).strftime("%Y-%m-%d") for t in df.snapshot_ts]


def rfm_distribution(snapshot: str) -> pd.DataFrame:
    df = q("SELECT rfm_segment, COUNT(*) AS customers, AVG(CAST(lifetime_spend AS FLOAT)) AS avg_lifetime_spend, "
           "AVG(CAST(lifetime_orders AS FLOAT)) AS avg_orders FROM analytics.customer_features WHERE snapshot_ts = :s "
           "GROUP BY rfm_segment ORDER BY customers DESC", {"s": snapshot})
    df["share_pct"] = 100 * df.customers / df.customers.sum()
    return df


def model_versions() -> list[str]:
    return list(q("SELECT DISTINCT model_version FROM analytics.churn_predictions ORDER BY model_version DESC").model_version)


def risk_distribution(model_version: str, partition: str = "scoring") -> pd.DataFrame:
    """Customers per 5%-wide churn-probability band."""
    df = q("SELECT CAST(FLOOR(CAST(churn_probability AS FLOAT) * 20) / 20.0 AS DECIMAL(4,2)) AS band_start, COUNT(*) AS customers, "
           "SUM(CAST(predicted_churn AS INT)) AS flagged FROM analytics.churn_predictions "
           "WHERE model_version = :v AND partition = :p GROUP BY CAST(FLOOR(CAST(churn_probability AS FLOAT) * 20) / 20.0 AS DECIMAL(4,2)) "
           "ORDER BY band_start", {"v": model_version, "p": partition})
    df["band_start"] = df["band_start"].astype(float)
    return df


def top_risk_customers(model_version: str, n: int = 25) -> pd.DataFrame:
    """Highest-risk customers in the scoring snapshot (synthetic ids only, no personal attributes)."""
    return q("SELECT TOP (:n) p.user_id AS customer_id, CAST(p.churn_probability AS FLOAT) AS churn_probability, p.risk_decile, "
             "f.days_since_last_purchase, f.orders_90d, f.lifetime_orders, CAST(f.spend_90d AS FLOAT) AS spend_90d, f.rfm_segment "
             "FROM analytics.churn_predictions p JOIN analytics.customer_features f ON f.snapshot_ts = p.snapshot_ts AND f.user_id = p.user_id "
             "WHERE p.model_version = :v AND p.partition = 'scoring' ORDER BY p.churn_probability DESC, p.user_id",
             {"n": int(n), "v": model_version})


def decile_outcomes(model_version: str) -> pd.DataFrame:
    """Test partition: observed churn rate per predicted-risk decile (10 = highest risk)."""
    df = q("SELECT risk_decile, COUNT(*) AS customers, AVG(CAST(actual_churn AS FLOAT)) AS observed_churn_rate, "
           "AVG(CAST(churn_probability AS FLOAT)) AS mean_predicted FROM analytics.churn_predictions "
           "WHERE model_version = :v AND partition = 'test' GROUP BY risk_decile ORDER BY risk_decile", {"v": model_version})
    return df


# --------------------------------------------------------------------------- model + benchmarks
def model_metrics(tag: str = "active_90d") -> dict | None:
    """Metrics written by scripts/evaluate_model.py - read from file, never invented or hard-coded."""
    p = C.PROJECT_ROOT / "reports" / f"metrics_{tag}.json"
    return json.loads(Path(p).read_text()) if p.exists() else None


def benchmark_table(cache_state: str = "warm") -> pd.DataFrame:
    """Median per (benchmark, design) from valid benchmark runs (latest valid run wins)."""
    df = q("SELECT r.benchmark_id, r.workload, r.variant, r.design, r.run_id, r.cache_state, "
           "CAST(r.logical_reads AS FLOAT) AS logical_reads, CAST(r.elapsed_ms AS FLOAT) AS elapsed_ms "
           "FROM analytics.benchmark_results r JOIN analytics.benchmark_runs b ON b.run_id = r.run_id "
           "WHERE r.cache_state = :c AND ISNULL(b.notes, '') NOT LIKE 'INVALID%'", {"c": cache_state})
    if df.empty:
        return df
    df = df[df.run_id == df.groupby(["benchmark_id", "design"]).run_id.transform("max")]
    out = df.groupby(["benchmark_id", "workload", "variant", "design"]).agg(
        elapsed_ms=("elapsed_ms", "median"), logical_reads=("logical_reads", "median")).reset_index()
    out["design_letter"] = out.design.str[0]
    return out


def benchmark_environment() -> dict | None:
    df = q("SELECT TOP 1 environment FROM analytics.benchmark_runs WHERE ISNULL(notes, '') NOT LIKE 'INVALID%' ORDER BY run_id DESC")
    return json.loads(df.environment[0]) if len(df) else None
