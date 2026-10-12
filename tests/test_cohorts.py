"""Cohort retention: denominators, zero-vs-unobserved, partial months, activity definitions."""
import pandas as pd
import pytest

pytestmark = pytest.mark.db
OBS_END = "2025-04-15"          # April 2025 is PARTIAL -> last observed month is March 2025


@pytest.fixture()
def cohorts(scn):
    # January 2025 cohort: 4 users (denominator must include inactive ones)
    for uid, d in [(1, "2025-01-05"), (2, "2025-01-10"), (3, "2025-01-15"), (4, "2025-01-20")]:
        scn.add_user(uid, d)
    # February 2025 cohort: 2 users
    scn.add_user(5, "2025-02-03")
    scn.add_user(6, "2025-02-10")
    # April 2025 (partial month): must not form a cohort
    scn.add_user(7, "2025-04-02")
    scn.add_order(1, "2025-01-06")
    scn.add_order(1, "2025-01-20")                                           # 2nd order same month: user counted once
    scn.add_order(1, "2025-02-03")
    scn.add_order(2, "2025-03-03")                                           # skipped Feb -> active in M2 only
    scn.add_order(3, "2025-01-25", status="cancelled")                       # cancelled never counts
    scn.add_order(4, "2025-02-05", status="refunded", refunded_at="2025-02-20")   # refunded before obs end -> not active
    scn.add_order(5, "2025-02-20")
    scn.add_order(6, "2025-03-05", status="refunded", refunded_at="2025-05-01")   # refund AFTER obs end -> still active
    scn.add_order(7, "2025-04-05")
    scn.exec("EXEC analytics.usp_build_cohort_retention @observation_end=:e", e=OBS_END)
    df = scn.q("SELECT * FROM analytics.cohort_retention_long WHERE activity_definition='purchase'")
    return df.set_index([df.cohort_month.astype(str), "months_since_acquisition"])


def long_df(scn):
    d = scn.q("SELECT * FROM analytics.cohort_retention_long")
    d["cohort_month"] = pd.to_datetime(d.cohort_month)
    return d.set_index(["cohort_month", "months_since_acquisition"])


def cell(df, cohort, m):
    return df.loc[(cohort, m)]


def test_cohort_size_includes_customers_who_never_purchased(cohorts):
    assert cell(cohorts, "2025-01-01", 0).cohort_size == 4
    assert cell(cohorts, "2025-02-01", 0).cohort_size == 2


def test_partial_month_is_not_a_cohort(cohorts):
    assert "2025-04-01" not in set(cohorts.index.get_level_values(0))


def test_active_customers_and_percentages(cohorts):
    c = cohorts
    assert (cell(c, "2025-01-01", 0).active_customers, float(cell(c, "2025-01-01", 0).retention_pct)) == (1, 25.00)  # user 1 (once)
    assert (cell(c, "2025-01-01", 1).active_customers, float(cell(c, "2025-01-01", 1).retention_pct)) == (1, 25.00)  # user 1
    assert cell(c, "2025-01-01", 2).active_customers == 1                                                            # user 2 only
    assert (cell(c, "2025-02-01", 0).active_customers, float(cell(c, "2025-02-01", 0).retention_pct)) == (1, 50.00)  # user 5


def test_zero_retention_is_distinct_from_unobserved_future(cohorts):
    zero = cell(cohorts, "2025-02-01", 1)          # March 2025: observed
    # user 6 ordered 2025-03-05, refunded AFTER the observation end -> counts -> 1 active (not zero)
    assert zero.is_observed and zero.active_customers == 1 and float(zero.retention_pct) == 50.00
    # February cohort, M2 = April 2025 = partial month -> UNOBSERVED (NULL), even though nobody is active in it
    future = cell(cohorts, "2025-02-01", 2)
    assert (not future.is_observed) and pd.isna(future.active_customers) and pd.isna(future.retention_pct)
    # the same calendar month for the Jan cohort (M3) is outside the triangle: no row at all
    assert ("2025-01-01", 3) not in cohorts.index


def test_observed_month_with_no_activity_is_zero_not_null(scn):
    scn.add_user(1, "2025-01-05")
    scn.add_user(2, "2025-01-06")
    scn.add_user(3, "2025-02-03")                  # a later cohort, so the triangle has a NULL region
    scn.add_order(1, "2025-01-10")
    scn.exec("EXEC analytics.usp_build_cohort_retention @observation_end=:e", e="2025-04-15")
    df = long_df(scn)
    jan, feb = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")
    assert df.loc[(jan, 1)].is_observed and df.loc[(jan, 1)].active_customers == 0 and float(df.loc[(jan, 1)].retention_pct) == 0.0  # TRUE zero
    assert df.loc[(jan, 2)].is_observed and df.loc[(jan, 2)].active_customers == 0                                                   # March
    assert (not df.loc[(feb, 2)].is_observed) and pd.isna(df.loc[(feb, 2)].retention_pct)                                           # April = future/partial


def test_observation_end_exactly_on_a_month_start_makes_prior_month_observed(scn):
    scn.add_user(1, "2025-01-05")
    scn.add_user(2, "2025-02-10")
    scn.exec("EXEC analytics.usp_build_cohort_retention @observation_end=:e", e="2025-03-01")
    df = long_df(scn)
    jan, feb = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")
    assert df.loc[(jan, 1)].is_observed                                # February is complete (obs end = Mar 1 00:00)
    assert df.loc[(feb, 0)].is_observed and not df.loc[(feb, 1)].is_observed   # March has not started
    assert (jan, 2) not in df.index                                    # March is beyond the triangle


def test_refund_before_observation_end_removes_activity(cohorts):
    assert cell(cohorts, "2025-01-01", 1).active_customers == 1       # user 4's refunded Feb order is NOT counted (user 1 only)


def test_triangular_matrix_shape(scn, cohorts):
    m = scn.q("EXEC analytics.usp_get_cohort_retention_matrix").assign(cohort_month=lambda d: pd.to_datetime(d.cohort_month)).set_index("cohort_month")
    assert list(m.columns) == ["cohort_size", "M0", "M1", "M2"]
    assert pd.isna(m.loc[pd.Timestamp("2025-02-01"), "M2"])             # unobserved -> NULL in the matrix
    assert not pd.isna(m.loc[pd.Timestamp("2025-01-01"), "M2"])         # observed
    assert float(m.loc[pd.Timestamp("2025-02-01"), "M1"]) == 50.0       # observed non-zero is shown
    assert float(m.loc[pd.Timestamp("2025-01-01"), "M0"]) == 25.0
    assert m.loc[pd.Timestamp("2025-02-01"), "cohort_size"] == 2


def test_web_definition_ignores_bare_page_views(scn):
    scn.add_user(1, "2025-01-05")
    scn.add_user(2, "2025-01-06")
    scn.add_event(1, "2025-02-10 10:00:00", "page_view")        # not meaningful
    scn.add_event(2, "2025-02-10 10:00:00", "product_view")     # meaningful
    scn.exec("EXEC analytics.usp_build_cohort_retention @observation_end=:e, @activity_definition='web'", e="2025-04-15")
    df = scn.q("SELECT * FROM analytics.cohort_retention_long WHERE activity_definition='web'").set_index("months_since_acquisition")
    assert df.loc[1].active_customers == 1 and df.loc[0].active_customers == 0


def test_rebuild_replaces_rather_than_duplicates(scn):
    scn.add_user(1, "2025-01-05")
    for _ in range(2):
        scn.exec("EXEC analytics.usp_build_cohort_retention @observation_end=:e", e="2025-04-15")
    assert len(scn.q("SELECT 1 x FROM analytics.cohort_retention_long")) == 3     # M0..M2 for one cohort
