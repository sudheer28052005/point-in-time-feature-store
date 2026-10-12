"""Dashboard data layer + a headless run of the Streamlit app."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from dashboard import data as D
from src import config as C
from src.database import read_sql_df


# ------------------------------- read-only guard (no DB) -------------------------------
@pytest.mark.parametrize("sql", ["SELECT 1", "  select * from analytics.sessions", "WITH x AS (SELECT 1 a) SELECT a FROM x;",
                                 "-- note\nSELECT created_at FROM dbo.users", "SELECT /* c */ 1"])
def test_guard_accepts_plain_selects(sql):
    D.assert_select_only(sql)


@pytest.mark.parametrize("sql", ["DELETE FROM dbo.orders", "UPDATE dbo.users SET country='XX'", "SELECT 1; DROP TABLE dbo.users",
                                 "SELECT * INTO dbo.copy FROM dbo.users", "EXEC analytics.usp_build_sessions", "INSERT dbo.users DEFAULT VALUES",
                                 "WITH x AS (SELECT 1 a) DELETE FROM dbo.orders", "TRUNCATE TABLE analytics.sessions", "CREATE TABLE t (i int)",
                                 "SELECT 1 -- ok\n; DELETE FROM dbo.orders", "  /* sneaky */ DROP TABLE x", ""])
def test_guard_rejects_anything_that_could_write(sql):
    with pytest.raises(PermissionError):
        D.assert_select_only(sql)


def test_every_query_in_the_data_layer_passes_the_guard():
    """Static check: all SQL literals in dashboard/data.py are read-only (they are also checked at run time by q())."""
    import ast
    tree = ast.parse((C.PROJECT_ROOT / "dashboard" / "data.py").read_text(encoding="utf-8"))
    n = 0
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "q" and call.args:
            try:
                sql = ast.literal_eval(call.args[0])
            except ValueError:      # implicit string concatenation of literals is folded by ast; anything else is skipped
                continue
            D.assert_select_only(sql)
            n += 1
    assert n >= 10


# ------------------------------- values against an independent recomputation -------------------------------
pytestmark_db = pytest.mark.db


@pytest.mark.db
def test_kpis_match_independent_pandas_calculation(db_ready):
    f, t = date(2024, 3, 1), date(2024, 9, 30)
    k = D.kpis(f, t)
    end = pd.Timestamp(t) + pd.Timedelta(days=1)
    o = read_sql_df("SELECT order_ts, order_status, refunded_at, CAST(gross_amount - discount_amount AS FLOAT) AS net FROM dbo.orders "
                    "WHERE order_ts >= :f AND order_ts < :e", {"f": pd.Timestamp(f), "e": end})
    ok = (o.order_status == "completed") | ((o.order_status == "refunded") & (o.refunded_at >= end))
    assert k["orders"] == int(ok.sum())
    assert k["revenue"] == pytest.approx(float(o.net[ok].sum()), abs=0.01)
    assert k["aov"] == pytest.approx(k["revenue"] / k["orders"])
    n_users = int(read_sql_df("SELECT COUNT(*) n FROM dbo.users WHERE signup_at < :e", {"e": end}).n[0])
    assert k["customers"] == n_users


@pytest.mark.db
def test_monthly_activity_adds_up_to_the_kpis_and_does_not_double_count(db_ready):
    f, t = date(2024, 1, 1), date(2024, 12, 31)
    m, k = D.monthly_activity(f, t), D.kpis(f, t)
    assert int(m.orders.sum()) == k["orders"] and m.revenue.sum() == pytest.approx(k["revenue"], abs=0.01)
    assert m.month.is_monotonic_increasing and m.month.is_unique
    assert (m.active_customers <= m.orders).all()


@pytest.mark.db
def test_empty_period_returns_zeros_not_errors(db_ready):
    k = D.kpis(date(2000, 1, 1), date(2000, 1, 31))
    assert k["orders"] == 0 and k["revenue"] == 0 and np.isnan(k["aov"])
    assert D.monthly_activity(date(2000, 1, 1), date(2000, 1, 31)).empty


@pytest.mark.db
@pytest.mark.parametrize("definition", ["purchase", "web"])
def test_cohort_matrix_keeps_unobserved_cells_as_nan_not_zero(db_ready, definition):
    long, mat = D.cohort_long(definition), D.cohort_matrix(definition)
    assert int(np.isnan(mat.to_numpy()).sum()) == int((~long.is_observed.astype(bool)).sum())
    obs_zero = long[long.is_observed.astype(bool) & (long.active_customers == 0)]
    for r in obs_zero.itertuples():
        assert mat.loc[r.cohort_month, r.months_since_acquisition] == 0.0           # real zeros stay zeros


@pytest.mark.db
def test_risk_views_are_consistent_and_expose_no_personal_columns(db_ready):
    v = D.model_versions()
    if not v:
        pytest.skip("run evaluate_model.py")
    ver = v[0]
    top = D.top_risk_customers(ver, 20)
    assert len(top) == 20 and top.churn_probability.is_monotonic_decreasing and top.customer_id.is_unique
    assert not {"date_of_birth", "country", "email", "name"} & set(top.columns)
    rd = D.risk_distribution(ver, "scoring")
    n = int(read_sql_df("SELECT COUNT(*) n FROM analytics.churn_predictions WHERE model_version = :v AND partition = 'scoring'", {"v": ver}).n[0])
    assert int(rd.customers.sum()) == n and rd.band_start.between(0, 0.95).all()
    dd = D.decile_outcomes(ver)
    assert dd.observed_churn_rate.is_monotonic_increasing and len(dd) == 10          # higher risk decile -> higher real churn


@pytest.mark.db
def test_rfm_distribution_shares_sum_to_100(db_ready):
    r = D.rfm_distribution("2025-01-01")
    assert r.share_pct.sum() == pytest.approx(100) and r.customers.sum() == int(
        read_sql_df("SELECT COUNT(*) n FROM analytics.customer_features WHERE snapshot_ts = '2025-01-01'").n[0])


@pytest.mark.db
def test_benchmark_table_ignores_runs_marked_invalid(db_ready):
    bt = D.benchmark_table("warm")
    assert not bt.empty
    bad = read_sql_df("SELECT COUNT(*) n FROM analytics.benchmark_runs WHERE notes LIKE 'INVALID%'").n[0]
    assert bad >= 1                                                                      # the invalid run exists but is excluded...
    # ...and it was the ONLY source of design-D rows for the feature-build benchmarks (D mis-parsed as bare heaps there)
    assert not (bt.benchmark_id.str.startswith("B11") & (bt.design_letter == "D")).any()
    assert set(bt.design_letter) <= set("ABCDEF")


def test_model_metrics_come_from_file_and_match_champion():
    mm = D.model_metrics("active_90d")
    if mm is None:
        pytest.skip("run evaluate_model.py")
    assert mm["champion"] in mm["models"] and 0.5 < mm["models"][mm["champion"]]["ranking"]["roc_auc"] < 1
    assert D.model_metrics("no_such_tag") is None


# ------------------------------- the whole app, headless -------------------------------
@pytest.mark.db
def test_app_renders_every_tab_without_exceptions_and_reacts_to_filters(db_ready):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(C.PROJECT_ROOT / "dashboard" / "app.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 5 and len(at.metric) >= 9
    assert any("Synthetic data" in w.value for w in at.warning)
    labels = {m.label for m in at.metric}
    assert {"Qualifying orders in period", "Net revenue in period", "Average order value"} <= labels
    # interaction: switch the retention definition and the performance metric; nothing may break
    for r in at.radio:
        if "web" in r.options:
            r.set_value("web")
        if "elapsed_ms" in r.options:
            r.set_value("elapsed_ms")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
