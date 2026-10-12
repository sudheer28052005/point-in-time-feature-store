"""Point-in-time correctness: no feature may use information at or after the cutoff.

Contract: history = ts < T0 (cutoff EXCLUDED); window N days = [T0 - N days, T0).
"""
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.db
T0 = "2024-06-01 00:00:00"


# ---------------------------- cutoff boundary ----------------------------
def test_order_at_cutoff_and_after_are_excluded_order_just_before_is_included(scn):
    scn.add_user(1)
    scn.add_order(1, "2024-05-31 23:59:59.999", gross=10)     # included
    scn.add_order(1, "2024-06-01 00:00:00.000", gross=1000)   # == T0 -> excluded
    scn.add_order(1, "2024-06-01 00:00:00.001", gross=1000)   # after -> excluded
    scn.add_order(1, "2024-09-01 12:00:00", gross=1000)       # far future -> excluded
    f = scn.build_features(T0).loc[1]
    assert (f.lifetime_orders, f.orders_30d, f.orders_90d, f.orders_365d) == (1, 1, 1, 1)
    assert float(f.lifetime_spend) == 10 and f.last_purchase_ts == pd.Timestamp("2024-05-31 23:59:59.999")


def test_event_at_cutoff_and_after_are_excluded(scn):
    scn.add_user(1)
    scn.add_event(1, "2024-05-31 23:59:59.999", "product_view")   # included
    scn.add_event(1, "2024-06-01 00:00:00.000", "product_view")   # == T0 -> excluded
    scn.add_event(1, "2024-06-02 10:00:00", "product_view")       # future
    f = scn.build_features(T0).loc[1]
    assert (f.events_7d, f.events_30d, f.events_90d, f.product_views_30d) == (1, 1, 1, 1)
    assert (f.sessions_7d, f.sessions_30d, f.sessions_90d) == (1, 1, 1)


def test_window_lower_bounds_are_inclusive(scn):
    """[T0-30d, T0): an order exactly 30 days before T0 is IN the 30d window; 1 ms earlier is not."""
    scn.add_user(1)
    scn.add_order(1, "2024-05-02 00:00:00.000", gross=10)   # exactly T0 - 30d
    scn.add_order(1, "2024-05-01 23:59:59.999", gross=20)   # 1 ms earlier
    scn.add_event(1, "2024-05-25 00:00:00.000")             # exactly T0 - 7d
    scn.add_event(1, "2024-05-24 23:59:59.999")             # 1 ms earlier
    f = scn.build_features(T0).loc[1]
    assert (f.orders_30d, f.orders_90d, f.lifetime_orders) == (1, 2, 2)
    assert (float(f.spend_30d), float(f.spend_90d)) == (10, 30)
    assert (f.events_7d, f.events_30d) == (1, 2)


def test_users_registered_at_or_after_cutoff_are_excluded(scn):
    scn.add_user(1, "2024-05-31 23:59:59.999")
    scn.add_user(2, T0)
    scn.add_user(3, "2024-06-15")
    f = scn.build_features(T0)
    assert f.index.tolist() == [1]


def test_refund_policy_is_evaluated_at_the_cutoff(scn):
    scn.add_user(1)
    scn.add_order(1, "2024-05-01", status="completed", gross=10)
    scn.add_order(1, "2024-05-02", status="cancelled", gross=100)                                    # never counts
    scn.add_order(1, "2024-05-03", status="refunded", gross=20, refunded_at="2024-06-10")            # refund AFTER T0 -> counts
    scn.add_order(1, "2024-05-04", status="refunded", gross=40, refunded_at="2024-06-01 00:00:00")   # refund == T0 -> counts
    scn.add_order(1, "2024-05-05", status="refunded", gross=80, refunded_at="2024-05-31 23:59:59.999")  # refund before T0 -> no
    f = scn.build_features(T0).loc[1]
    assert f.lifetime_orders == 3 and float(f.lifetime_spend) == 70
    # at a LATER cutoff (2024-07-01) both the 06-01 and 06-10 refunds are known -> only the completed order is left
    f2 = scn.build_features("2024-07-01").loc[1]
    assert f2.lifetime_orders == 1 and float(f2.lifetime_spend) == 10
    # and at 2024-06-05 only the 06-01 refund is known
    f3 = scn.build_features("2024-06-05").loc[1]
    assert f3.lifetime_orders == 2 and float(f3.lifetime_spend) == 30


# ---------------------------- no leakage from the future ----------------------------
def test_adding_future_data_does_not_change_the_snapshot(scn):
    scn.add_users([1, 2, 3])
    scn.add_order(1, "2024-05-10", gross=50)
    scn.add_order(1, "2024-04-10", gross=70)
    scn.add_order(2, "2024-03-01", gross=30)
    scn.add_event(1, "2024-05-30 10:00:00", "purchase")
    scn.add_event(2, "2024-05-20 10:00:00", "product_view")
    before = scn.build_features(T0).drop(columns="feature_generated_at")
    # now add lots of future activity (and an order whose refund lies in the future)
    scn.add_order(1, "2024-06-01", gross=999)
    scn.add_order(2, "2024-06-20", gross=999)
    scn.add_order(3, "2024-07-04", gross=999, status="refunded", refunded_at="2024-07-05")
    scn.add_event(1, "2024-06-01 00:00:00", "purchase")
    scn.add_event(3, "2024-06-03 08:00:00", "add_to_cart")
    scn.add_event(2, "2024-05-20 10:10:00", "product_view")   # past event is allowed to change things...
    after = scn.build_features(T0).drop(columns="feature_generated_at")
    # ...so compare only after removing that deliberate past addition from the expectation:
    assert after.loc[[1, 3]].equals(before.loc[[1, 3]])
    assert after.loc[2].lifetime_orders == before.loc[2].lifetime_orders
    assert after.loc[2].events_30d == before.loc[2].events_30d + 1


def test_snapshot_is_deterministic_and_rebuildable(scn):
    scn.add_users([1, 2])
    scn.add_order(1, "2024-05-10")
    scn.add_event(2, "2024-05-11 10:00:00")
    a = scn.build_features(T0).drop(columns="feature_generated_at")
    b = scn.build_features(T0).drop(columns="feature_generated_at")
    pd.testing.assert_frame_equal(a, b)
    assert len(scn.q("SELECT 1 x FROM analytics.customer_features WHERE snapshot_ts = :t", t=T0)) == 2   # replaced, not duplicated


def test_multiple_snapshots_coexist(scn):
    scn.add_user(1, "2024-01-01")
    scn.add_order(1, "2024-03-15")
    scn.build_sessions()
    for t in ("2024-03-01", "2024-04-01"):
        scn.exec("EXEC analytics.usp_build_customer_features @T0 = :t", t=t)
    assert scn.features("2024-03-01").loc[1].lifetime_orders == 0
    assert scn.features("2024-04-01").loc[1].lifetime_orders == 1


# ---------------------------- RFM / aggregates ----------------------------
def test_order_statistics_for_a_known_customer(scn):
    scn.add_user(1)
    scn.add_order(1, "2024-05-01 12:00:00", gross=120, discount=20)   # net 100
    scn.add_order(1, "2024-05-11 12:00:00", gross=50)                 # +10 days
    scn.add_order(1, "2024-05-31 12:00:00", gross=150)                # +20 days
    f = scn.build_features(T0).loc[1]
    assert (f.lifetime_orders, f.distinct_purchase_dates) == (3, 3)
    assert float(f.lifetime_spend) == 300 and float(f.avg_order_value) == 100 and float(f.max_order_value) == 150
    assert float(f.std_order_value) == pytest.approx(50.0)                       # sample stdev of 100, 50, 150
    assert float(f.mean_purchase_interval_days) == pytest.approx(15.0)           # intervals 10, 20
    assert float(f.std_purchase_interval_days) == pytest.approx(7.07, abs=0.005)  # sample stdev of 10, 20
    assert float(f.median_purchase_interval_days) == pytest.approx(15.0)
    assert float(f.days_since_last_purchase) == pytest.approx(0.5)                # 12 h before T0


def test_median_interval_with_even_number_of_intervals(scn):
    scn.add_user(1)
    day = pd.Timestamp("2024-01-01")
    for off in (0, 10, 30, 60, 100):                                  # intervals 10, 20, 30, 40
        scn.add_order(1, (day + pd.Timedelta(days=off)).isoformat())
    f = scn.build_features(T0).loc[1]
    assert float(f.median_purchase_interval_days) == pytest.approx(25.0)
    assert float(f.mean_purchase_interval_days) == pytest.approx(25.0)


def test_single_order_has_no_interval_or_stdev(scn):
    scn.add_user(1)
    scn.add_order(1, "2024-05-10", gross=80)
    f = scn.build_features(T0).loc[1]
    assert f.lifetime_orders == 1 and float(f.avg_order_value) == 80
    for col in ("std_order_value", "mean_purchase_interval_days", "std_purchase_interval_days", "median_purchase_interval_days"):
        assert pd.isna(f[col]), col                                   # undefined, NOT fabricated as 0


def test_two_orders_have_interval_but_no_interval_stdev(scn):
    scn.add_user(1)
    scn.add_order(1, "2024-05-10 00:00:00")
    scn.add_order(1, "2024-05-12 00:00:00")
    f = scn.build_features(T0).loc[1]
    assert float(f.mean_purchase_interval_days) == 2.0 and pd.isna(f.std_purchase_interval_days)
    assert not pd.isna(f.std_order_value)


def test_same_timestamp_orders_have_zero_interval_and_one_purchase_date(scn):
    scn.add_user(1)
    scn.add_order(1, "2024-05-10 10:00:00", order_id=2)
    scn.add_order(1, "2024-05-10 10:00:00", order_id=1)
    f = scn.build_features(T0).loc[1]
    assert f.lifetime_orders == 2 and f.distinct_purchase_dates == 1
    assert float(f.mean_purchase_interval_days) == 0.0


def test_order_count_change_compares_last_30d_with_the_30d_before(scn):
    scn.add_user(1)
    for ts in ["2024-05-20", "2024-05-25", "2024-05-30"]:              # 3 in [T0-30d, T0)
        scn.add_order(1, ts)
    scn.add_order(1, "2024-04-10")                                     # 1 in [T0-60d, T0-30d)
    f = scn.build_features(T0).loc[1]
    assert (f.orders_30d, f.orders_prev_30d, f.order_count_change_30d) == (3, 1, 2)


def test_users_without_purchases_get_zeros_for_counts_and_nulls_for_undefined_values(scn):
    scn.add_user(1)
    f = scn.build_features(T0).loc[1]
    assert (f.has_purchased, f.lifetime_orders, f.orders_30d, f.orders_365d, f.distinct_purchase_dates) == (False, 0, 0, 0, 0)
    assert float(f.lifetime_spend) == 0 and float(f.spend_90d) == 0
    for col in ("last_purchase_ts", "days_since_last_purchase", "avg_order_value", "max_order_value", "std_order_value",
                "mean_purchase_interval_days", "median_purchase_interval_days",
                "recency_quintile", "frequency_quintile", "spend_quintile", "conversion_rate_30d"):
        assert pd.isna(f[col]), col
    assert f.rfm_segment == "No Purchase" and f.has_web_events == False  # noqa: E712


def test_conversion_rate_numerator_and_denominator(scn):
    scn.add_users([1, 2, 3])
    # user 1: 2 sessions in the window, one contains a purchase -> 0.5
    scn.add_event(1, "2024-05-20 10:00:00", "page_view")
    scn.add_event(1, "2024-05-20 10:05:00", "purchase")
    scn.add_event(1, "2024-05-22 10:00:00", "page_view")
    # user 2: sessions but no purchase -> 0 (a real zero)
    scn.add_event(2, "2024-05-20 10:00:00", "page_view")
    # user 3: no sessions in window -> NULL, never a division by zero
    f = scn.build_features(T0)
    assert float(f.loc[1].conversion_rate_30d) == 0.5 and f.loc[1].sessions_30d == 2
    assert float(f.loc[2].conversion_rate_30d) == 0.0
    assert pd.isna(f.loc[3].conversion_rate_30d) and f.loc[3].sessions_30d == 0


def test_conversion_ignores_a_purchase_that_happens_after_the_cutoff(scn):
    """A session that straddles T0 must not leak its post-cutoff purchase into the feature."""
    scn.add_user(1)
    scn.add_event(1, "2024-05-31 23:50:00", "product_view")
    scn.add_event(1, "2024-06-01 00:05:00", "purchase")     # same session, after T0
    f = scn.build_features(T0).loc[1]
    assert f.sessions_30d == 1 and float(f.conversion_rate_30d) == 0.0


def test_joining_orders_and_events_does_not_multiply_rows(scn):
    scn.add_user(1)
    for d in (10, 12, 14):
        scn.add_order(1, f"2024-05-{d}", gross=10)
    for m in range(5):
        scn.add_event(1, f"2024-05-20 10:0{m}:00")
    f = scn.build_features(T0).loc[1]
    assert (f.lifetime_orders, f.events_30d, f.sessions_30d) == (3, 5, 1)
    assert float(f.lifetime_spend) == 30


def test_quintiles_are_computed_only_over_customers_who_purchased(scn):
    buyers = list(range(1, 11))
    scn.add_users(buyers + list(range(101, 201)))                      # 100 extra non-buyers
    for i in buyers:   # buyer i: one order i days before T0, net spend 10*i
        scn.add_order(i, (pd.Timestamp(T0) - pd.Timedelta(days=i)).isoformat(), gross=10 * i)
    f = scn.build_features(T0)
    assert f.loc[buyers].spend_quintile.tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert f.loc[buyers].recency_quintile.tolist() == [5, 5, 4, 4, 3, 3, 2, 2, 1, 1]   # 5 = most recent
    assert f.loc[101:200].spend_quintile.isna().all() and (f.loc[101:200].rfm_segment == "No Purchase").all()
    assert f.loc[buyers].frequency_quintile.between(1, 5).all()


def test_rfm_segment_rules(scn):
    # 10 buyers with distinct frequency AND spend AND recency ordering -> predictable quintiles
    users = list(range(1, 11))
    scn.add_users(users)
    for i in users:   # buyer i: i orders; the latest one i*5 days before T0 ... recent = small i
        for k in range(i):
            scn.add_order(i, (pd.Timestamp(T0) - pd.Timedelta(days=5 * i + k)).isoformat(), gross=10 * i)
    f = scn.build_features(T0)
    assert f.loc[1].rfm_segment == "New Customers"                     # one order, R=5
    assert f.loc[2].rfm_segment in {"Potential Loyalists", "New Customers"} and f.loc[2].recency_quintile == 5
    assert f.loc[10].rfm_segment == "Hibernating" or f.loc[10].recency_quintile <= 2
    assert set(f.rfm_segment) <= {"No Purchase", "New Customers", "Champions", "Loyal Customers", "Potential Loyalists",
                                  "At Risk", "Hibernating", "Need Attention"}
    # high frequency + high spend but long silent => At Risk, not Champions
    assert f.loc[10].rfm_segment in {"At Risk", "Hibernating"}
