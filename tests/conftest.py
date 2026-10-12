import dataclasses
import itertools
import os
import re
import uuid
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

from src.config import PROJECT_ROOT, get_settings
from src.data_generation import GenerationConfig, generate_dataset
from src.database import db_available, get_engine, read_sql_df

SMALL = GenerationConfig(num_users=1500, num_orders=6000, num_events=30000, seed=11)

TEST_DB = "CustomerFeatureStore_Test"   # throwaway database; the real one is never touched
_PROD_DB = "CustomerFeatureStore"
_BUILD_SCRIPTS = ["01_create_database.sql", "02_create_schema.sql", "03_create_tables.sql",
                  "04_create_constraints.sql", "06_sessionization.sql", "07_customer_features.sql",
                  "08_retention_cohorts.sql", "09_churn_labels.sql", "10_create_feature_store.sql",
                  "11_create_indexes.sql"]


@pytest.fixture(scope="session")
def small_dataset():
    return generate_dataset(SMALL)


@pytest.fixture(scope="session")
def db_ready():
    """Skip DB tests when SQL Server (or the loaded data) is unavailable."""
    if not db_available():
        pytest.skip("SQL Server not reachable")
    from src.validation import table_row_counts
    if table_row_counts()["users"] == 0:
        pytest.skip("CustomerFeatureStore is empty - run scripts/generate_data.py and load_data.py")
    return True


# --------------------------------------------------------------------------- #
# Throwaway database built from the REAL sql/ scripts (name substituted), so the
# edge-case tests exercise exactly the code that is deployed.
# --------------------------------------------------------------------------- #
def _run_script(engine, path, db_name):
    sql = path.read_text(encoding="utf-8").replace(_PROD_DB, db_name)
    batches = [b.strip() for b in re.split(r"(?im)^\s*GO\s*$", sql) if b.strip()]
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for b in batches:
            conn.exec_driver_sql(b)


def _drop_test_db(master_engine):
    assert TEST_DB.endswith("_Test")  # hard guard: never drop anything else
    with master_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.exec_driver_sql(f"IF DB_ID(N'{TEST_DB}') IS NOT NULL BEGIN "
                             f"ALTER DATABASE [{TEST_DB}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE; "
                             f"DROP DATABASE [{TEST_DB}]; END")


@pytest.fixture(scope="session")
def test_settings():
    if not db_available():
        pytest.skip("SQL Server not reachable")
    base = get_settings()
    master = dataclasses.replace(base, database="master")
    master_engine = get_engine(master)
    _drop_test_db(master_engine)
    # PITFS_SQL_DIR lets mutation testing run the suite against deliberately broken copies of the SQL.
    sql_dir = Path(os.getenv("PITFS_SQL_DIR") or PROJECT_ROOT / "sql")
    for name in _BUILD_SCRIPTS[:1]:
        _run_script(master_engine, sql_dir / name, TEST_DB)
    ts = dataclasses.replace(base, database=TEST_DB)
    test_engine = get_engine(ts)
    for name in _BUILD_SCRIPTS[1:]:
        _run_script(test_engine, sql_dir / name, TEST_DB)
    yield ts
    test_engine.dispose()
    _drop_test_db(master_engine)


class Scenario:
    """Tiny DSL for hand-built edge cases inside the throwaway database."""

    def __init__(self, settings):
        self.s = settings
        self.engine = get_engine(settings)
        self._order_ids = itertools.count(1)
        self._event_ids = itertools.count(1)

    # ---- setup ----
    def reset(self):
        with self.engine.begin() as c:
            for t in ("analytics.customer_features", "analytics.churn_labels", "analytics.cohort_retention_long",
                      "dbo.raw_web_events", "dbo.orders", "dbo.users"):
                c.execute(text(f"DELETE FROM {t}"))
            c.execute(text("TRUNCATE TABLE analytics.sessions"))
            c.execute(text("TRUNCATE TABLE analytics.session_events"))
        self._order_ids = itertools.count(1)
        self._event_ids = itertools.count(1)
        return self

    def add_user(self, user_id, signup_at="2020-01-01", **kw):
        row = dict(user_id=user_id, signup_at=signup_at, country="US", channel="organic",
                   segment="standard", opted=1)
        row.update(kw)
        with self.engine.begin() as c:
            c.execute(text("""INSERT dbo.users (user_id, signup_at, country, acquisition_channel, customer_segment,
                              is_marketing_opted_in) VALUES (:user_id, :signup_at, :country, :channel, :segment, :opted)"""), row)

    def add_users(self, ids, signup_at="2020-01-01"):
        for i in ids:
            self.add_user(i, signup_at)

    def add_order(self, user_id, ts, status="completed", gross=100, discount=0, refunded_at=None, order_id=None):
        with self.engine.begin() as c:
            c.execute(text("""INSERT dbo.orders (order_id, user_id, order_ts, order_status, gross_amount, discount_amount,
                              payment_method, refunded_at) VALUES (:id, :u, :ts, :st, :g, :d, 'card', :r)"""),
                      dict(id=order_id or next(self._order_ids), u=user_id, ts=ts, st=status, g=gross, d=discount, r=refunded_at))

    def add_event(self, user_id, ts, etype="page_view", device="desktop", event_id=None, source_session_id=None):
        with self.engine.begin() as c:
            c.execute(text("""INSERT dbo.raw_web_events (event_id, user_id, event_ts, source_session_id, event_type, device_type)
                              VALUES (:id, :u, :ts, :s, :t, :d)"""),
                      dict(id=event_id or next(self._event_ids), u=user_id, ts=ts, s=source_session_id or str(uuid.uuid4()),
                           t=etype, d=device))

    # ---- actions ----
    def exec(self, sql, **params):
        with self.engine.begin() as c:
            c.execute(text(sql), params)

    def build_sessions(self, minutes=30):
        self.exec("EXEC analytics.usp_build_sessions @inactivity_minutes = :m", m=minutes)

    def build_features(self, t0):
        self.build_sessions()
        self.exec("EXEC analytics.usp_build_customer_features @T0 = :t", t=t0)
        return self.features(t0)

    def build_labels(self, t0, observation_end, outcome_days=30):
        self.exec("EXEC analytics.usp_build_churn_labels @T0=:t, @observation_end=:e, @outcome_days=:d",
                  t=t0, e=observation_end, d=outcome_days)
        return self.labels(t0)

    # ---- reads ----
    def q(self, sql, **params) -> pd.DataFrame:
        return read_sql_df(sql, params, settings=self.s)

    def features(self, t0) -> pd.DataFrame:
        return self.q("SELECT * FROM analytics.customer_features WHERE snapshot_ts = :t ORDER BY user_id", t=t0).set_index("user_id")

    def labels(self, t0) -> pd.DataFrame:
        return self.q("SELECT * FROM analytics.churn_labels WHERE snapshot_ts = :t ORDER BY user_id", t=t0).set_index("user_id")

    def session_events(self) -> pd.DataFrame:
        return self.q("SELECT * FROM analytics.session_events ORDER BY user_id, event_ts, event_id")

    def sessions(self) -> pd.DataFrame:
        return self.q("SELECT * FROM analytics.sessions ORDER BY user_id, session_seq")


@pytest.fixture()
def scn(test_settings):
    return Scenario(test_settings).reset()
