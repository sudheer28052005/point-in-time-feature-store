"""Runs the SQL data-quality checks in sql/13_validation_queries.sql."""
from __future__ import annotations

import logging
import re

import pandas as pd

from .config import PROJECT_ROOT, Settings
from .database import read_sql_df

log = logging.getLogger(__name__)
VALIDATION_SQL = PROJECT_ROOT / "sql" / "13_validation_queries.sql"


DERIVED_VALIDATION_SQL = PROJECT_ROOT / "sql" / "13b_validation_derived.sql"


def run_data_quality_checks(settings: Settings | None = None, derived: bool = False) -> pd.DataFrame:
    """Return DataFrame(check_name, violations). Every violations value should be 0.

    derived=False: source-table checks (13_validation_queries.sql).
    derived=True : sessions / features / labels / cohorts checks (13b_validation_derived.sql).
    """
    # The file starts with `USE <db>;` for SSMS/sqlcmd. Over ODBC the connection already
    # targets the configured database, and USE would add a result-less batch ahead of the SELECT.
    path = DERIVED_VALIDATION_SQL if derived else VALIDATION_SQL
    sql = re.sub(r"(?im)^USE\s+\w+;\s*$", "", path.read_text(encoding="utf-8"))
    df = read_sql_df(sql, settings=settings)
    for row in df.itertuples():
        (log.error if row.violations else log.debug)("check %s -> %d", row.check_name, row.violations)
    return df


def table_row_counts(settings: Settings | None = None) -> dict[str, int]:
    out = {}
    for table in ("users", "orders", "raw_web_events"):  # fixed whitelist, not user input
        out[table] = int(read_sql_df(f"SELECT COUNT_BIG(*) AS n FROM dbo.{table}", settings=settings).n[0])
    return out


def assert_all_pass(settings: Settings | None = None) -> pd.DataFrame:
    df = run_data_quality_checks(settings)
    failed = df[df.violations != 0]
    if not failed.empty:
        raise AssertionError("Data-quality checks failed:\n" + failed.to_string(index=False))
    return df
