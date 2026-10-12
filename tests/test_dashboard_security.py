"""The dashboard role really is read-only (verified with EXECUTE AS in the throwaway database)."""
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from src.config import PROJECT_ROOT
from tests.conftest import TEST_DB, _run_script

pytestmark = pytest.mark.db


@pytest.fixture()
def reader(scn):
    """A user WITHOUT a login in the test DB, member of dashboard_reader."""
    _run_script(scn.engine, PROJECT_ROOT / "sql" / "15_dashboard_readonly_role.sql", TEST_DB)
    with scn.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as c:
        c.exec_driver_sql("IF DATABASE_PRINCIPAL_ID('ro_user') IS NOT NULL DROP USER ro_user; "
                          "CREATE USER ro_user WITHOUT LOGIN; ALTER ROLE dashboard_reader ADD MEMBER ro_user;")
    scn.add_user(1, "2024-01-01")
    scn.add_order(1, "2024-02-01")
    yield scn
    with scn.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as c:
        c.exec_driver_sql("DROP USER IF EXISTS ro_user")


def as_reader(scn, sql):
    with scn.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as c:
        c.exec_driver_sql("EXECUTE AS USER = 'ro_user'")
        try:
            return c.execute(text(sql)).fetchall()
        finally:
            c.exec_driver_sql("REVERT")


def test_reader_can_select_analytics_and_allowed_columns(reader):
    assert as_reader(reader, "SELECT COUNT(*) FROM analytics.customer_features")[0][0] == 0
    assert as_reader(reader, "SELECT user_id, country, customer_segment FROM dbo.users")[0][0] == 1
    assert as_reader(reader, "SELECT COUNT(*) FROM dbo.orders")[0][0] == 1
    assert as_reader(reader, "SELECT COUNT_BIG(*) FROM analytics.fn_qualifying_orders('1900-01-01', '2030-01-01', '2030-01-01')")[0][0] == 1


@pytest.mark.parametrize("sql", [
    "SELECT date_of_birth FROM dbo.users",                       # sensitive column
    "SELECT * FROM dbo.users",                                   # * would include date_of_birth
    "SELECT COUNT(*) FROM dbo.raw_web_events",                   # raw clickstream
    "INSERT dbo.users (user_id, signup_at, country, acquisition_channel, customer_segment, is_marketing_opted_in, created_at) "
    "VALUES (99, SYSUTCDATETIME(), 'US', 'organic', 'standard', 1, SYSUTCDATETIME())",
    "DELETE FROM dbo.orders",
    "UPDATE analytics.customer_features SET lifetime_orders = 0",
    "TRUNCATE TABLE analytics.sessions",
    "EXEC analytics.usp_build_sessions",
    "CREATE TABLE analytics.evil (i INT)",
])
def test_reader_is_denied(reader, sql):
    with pytest.raises(DBAPIError, match="permission|denied"):
        as_reader(reader, sql)
