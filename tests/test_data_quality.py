"""Data-quality tests.

Part 1 checks the in-memory generator output (no database needed).
Part 2 runs the SQL checks in sql/13_validation_queries.sql against the loaded database.
"""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data_generation import GenerationConfig, dataset_fingerprint, generate_dataset
from tests.conftest import SMALL


# ------------------------------ generator (no DB) ------------------------------
def test_unique_identifiers(small_dataset):
    ds = small_dataset
    assert ds.users.user_id.is_unique
    assert ds.orders.order_id.is_unique
    assert ds.events.event_id.is_unique


def test_referential_integrity(small_dataset):
    ds = small_dataset
    assert ds.orders.user_id.isin(ds.users.user_id).all()
    assert ds.events.user_id.isin(ds.users.user_id).all()


def test_money_is_nonnegative_and_discount_bounded(small_dataset):
    o = small_dataset.orders
    assert (o.gross_amount >= 0).all() and (o.discount_amount >= 0).all()
    assert (o.discount_amount <= o.gross_amount).all()
    assert np.allclose(o.gross_amount, o.gross_amount.round(2)) and np.allclose(o.discount_amount, o.discount_amount.round(2))


def test_vocabularies_are_valid(small_dataset):
    ds = small_dataset
    assert set(ds.orders.order_status) <= set(C.ORDER_STATUSES)
    assert set(ds.orders.payment_method) <= set(C.PAYMENT_METHODS)
    assert set(ds.events.event_type) <= set(C.EVENT_TYPES)
    assert set(ds.events.device_type) <= set(C.DEVICE_TYPES)
    assert set(ds.events.referrer_type.dropna()) <= set(C.REFERRER_TYPES)
    assert set(ds.users.acquisition_channel) <= set(C.ACQUISITION_CHANNELS)
    assert set(ds.users.customer_segment) <= set(C.CUSTOMER_SEGMENTS)
    assert ds.users.country.str.fullmatch(r"[A-Z]{2}").all()


def test_refund_policy_columns(small_dataset):
    o = small_dataset.orders
    assert (o.order_status.eq("refunded") == o.refunded_at.notna()).all()
    r = o[o.refunded_at.notna()]
    assert (r.refunded_at >= r.order_ts).all()
    assert {"completed", "cancelled", "refunded"} <= set(o.order_status)  # all branches exercised


def test_no_activity_before_signup(small_dataset):
    ds = small_dataset
    signup = ds.users.set_index("user_id").signup_at
    assert (ds.orders.order_ts.values >= signup.loc[ds.orders.user_id].values).all()
    assert (ds.events.event_ts.values >= signup.loc[ds.events.user_id].values).all()


def test_timestamps_inside_observation_window(small_dataset):
    ds = small_dataset
    lo, hi = np.datetime64(C.OBS_START), np.datetime64(C.OBS_END)
    for col in (ds.users.signup_at, ds.orders.order_ts, ds.events.event_ts, ds.orders.refunded_at.dropna()):
        assert col.min() >= lo and col.max() < hi


def test_events_contain_gaps_over_30_minutes_and_source_ids_are_unreliable(small_dataset):
    """Sessionization must be needed: gaps >30 min exist inside one source session id."""
    e = small_dataset.events.sort_values(["user_id", "event_ts", "event_id"])
    same_user = e.user_id.eq(e.user_id.shift())
    same_src = e.source_session_id.eq(e.source_session_id.shift())
    gap_min = e.event_ts.diff().dt.total_seconds() / 60
    assert ((gap_min > 30) & same_user).any()
    assert ((gap_min > 30) & same_user & same_src).any()      # source id spans an idle gap
    assert e.groupby("user_id").source_session_id.nunique().mean() > 1


def test_volumes_close_to_targets(small_dataset):
    ds = small_dataset
    assert len(ds.users) == SMALL.num_users
    assert abs(len(ds.orders) - SMALL.num_orders) / SMALL.num_orders < 0.10
    assert abs(len(ds.events) - SMALL.num_events) / SMALL.num_events < 0.20


def test_some_customers_never_buy_and_some_repeat(small_dataset):
    per_user = small_dataset.orders.groupby("user_id").size().reindex(small_dataset.users.user_id, fill_value=0)
    assert (per_user == 0).mean() > 0.2
    assert (per_user >= 2).mean() > 0.1


# ------------------------------ determinism ------------------------------
def test_same_seed_gives_identical_data():
    a = generate_dataset(SMALL)
    b = generate_dataset(SMALL)
    assert dataset_fingerprint(a) == dataset_fingerprint(b)
    pd.testing.assert_frame_equal(a.events, b.events)


def test_different_seed_gives_different_data():
    other = GenerationConfig(SMALL.num_users, SMALL.num_orders, SMALL.num_events, seed=SMALL.seed + 1)
    assert dataset_fingerprint(generate_dataset(other)) != dataset_fingerprint(generate_dataset(SMALL))


def test_empty_volume_is_rejected():
    with pytest.raises(ValueError):
        generate_dataset(GenerationConfig(num_users=0))


# ------------------------------ database (SQL Server) ------------------------------
@pytest.mark.db
def test_sql_data_quality_checks_all_zero(db_ready):
    from src.validation import run_data_quality_checks
    df = run_data_quality_checks()
    assert len(df) >= 12
    assert df[df.violations != 0].empty, df.to_string()


@pytest.mark.db
def test_sql_checks_detect_a_planted_order_before_signup(db_ready):
    """The validator must have teeth: plant a violation, expect exactly that check to fail, clean up."""
    from sqlalchemy import text
    from src.database import get_engine
    from src.validation import run_data_quality_checks
    planted = 2_000_000_000
    try:
        with get_engine().begin() as conn:
            conn.execute(text("""INSERT dbo.orders (order_id, user_id, order_ts, order_status, gross_amount,
                                 discount_amount, payment_method)
                                 SELECT :oid, user_id, DATEADD(day, -1, signup_at), 'completed', 10, 0, 'card'
                                 FROM dbo.users WHERE user_id = (SELECT MIN(user_id) FROM dbo.users)"""),
                         {"oid": planted})
        df = run_data_quality_checks()
        assert set(df[df.violations != 0].check_name) == {"orders_before_signup"}
    finally:
        with get_engine().begin() as conn:
            conn.execute(text("DELETE FROM dbo.orders WHERE order_id = :oid"), {"oid": planted})
    assert run_data_quality_checks().violations.sum() == 0


@pytest.mark.db
def test_loaded_row_counts_match_audit_log(db_ready):
    from src.database import read_sql_df
    from src.validation import table_row_counts
    log = read_sql_df("SELECT TOP 1 num_users, num_orders, num_events FROM analytics.data_load_log ORDER BY load_id DESC")
    counts = table_row_counts()
    assert (log.num_users[0], log.num_orders[0], log.num_events[0]) == (
        counts["users"], counts["orders"], counts["raw_web_events"])


@pytest.mark.db
def test_database_refuses_invalid_rows(db_ready):
    """The constraints, not just the generator, protect the data (transaction is rolled back)."""
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError
    from src.database import get_engine
    with get_engine().connect() as conn:
        tx = conn.begin()
        with pytest.raises(IntegrityError):
            conn.execute(text("""INSERT dbo.orders (order_id, user_id, order_ts, order_status, gross_amount,
                                 discount_amount, payment_method) VALUES (-1, -1, SYSUTCDATETIME(), 'completed', 1, 0, 'card')"""))
        tx.rollback()
