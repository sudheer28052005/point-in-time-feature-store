"""SQLAlchemy 2.x engine factory for SQL Server via pyodbc (ODBC Driver 18).

Security notes
* Credentials come from environment variables only.
* Encryption is ON by default. TrustServerCertificate is a separate, explicit
  setting: a fresh local instance has a self-signed certificate that the client
  cannot validate, so local development typically needs SQL_SERVER_TRUST_CERTIFICATE=yes.
  Do not use that against a remote server.
* All queries take bound parameters; no SQL is built by string concatenation.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Mapping

import pandas as pd
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import URL, Engine
from sqlalchemy.exc import SQLAlchemyError

from .config import Settings, get_settings

log = logging.getLogger(__name__)


class DatabaseError(RuntimeError):
    """Raised with an actionable message when the database cannot be used."""


def _odbc_value(value: str) -> str:
    """Brace-quote values that contain ODBC connection-string metacharacters."""
    if any(ch in value for ch in ";{}="):
        return "{" + value.replace("}", "}}") + "}"
    return value


def build_odbc_connection_string(s: Settings) -> str:
    server = f"{s.host},{s.port}" if s.port else s.host  # named instances: no port
    parts = [
        f"DRIVER={{{s.driver}}}",
        f"SERVER={_odbc_value(server)}",
        f"DATABASE={_odbc_value(s.database)}",
        f"Encrypt={s.encrypt}",
        f"TrustServerCertificate={s.trust_certificate}",
        f"Connection Timeout={s.login_timeout}",
        "APP=PointInTimeFeatureStore",
    ]
    if s.uses_windows_auth:
        parts.append("Trusted_Connection=yes")
    else:
        parts += [f"UID={_odbc_value(s.username)}", f"PWD={_odbc_value(s.password)}"]
    return ";".join(parts)


@lru_cache(maxsize=4)
def get_engine(settings: Settings | None = None) -> Engine:
    """Pooled engine. fast_executemany makes executemany() a true bulk send."""
    s = settings or get_settings()
    url = URL.create("mssql+pyodbc", query={"odbc_connect": build_odbc_connection_string(s)})
    engine = create_engine(
        url,
        pool_size=s.pool_size,
        max_overflow=2,
        pool_pre_ping=True,
        pool_recycle=1800,
        fast_executemany=True,
    )
    if s.query_timeout:
        @event.listens_for(engine, "connect")
        def _set_query_timeout(dbapi_conn, _record):  # pragma: no cover - trivial
            dbapi_conn.timeout = s.query_timeout
    return engine


def check_connection(settings: Settings | None = None) -> str:
    """Return the server version string, or raise DatabaseError with a hint."""
    try:
        with get_engine(settings).connect() as conn:
            return conn.execute(text("SELECT @@VERSION")).scalar_one().splitlines()[0]
    except SQLAlchemyError as exc:
        raise DatabaseError(
            "Cannot connect to SQL Server. Check SQL_SERVER_HOST/PORT/DRIVER, that the "
            "database exists (run scripts/run_sql.ps1), and for a local self-signed "
            "certificate set SQL_SERVER_TRUST_CERTIFICATE=yes. Underlying error: "
            f"{str(exc.orig) if getattr(exc, 'orig', None) else exc}"
        ) from exc


def db_available(settings: Settings | None = None) -> bool:
    try:
        check_connection(settings)
        return True
    except DatabaseError:
        return False


def read_sql_df(
    sql: str,
    params: Mapping[str, Any] | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Run a parameterised SELECT (use :name placeholders) and return a DataFrame."""
    with get_engine(settings).connect() as conn:
        return pd.read_sql_query(text(sql), conn, params=dict(params or {}))


def execute(sql: str, params: Mapping[str, Any] | None = None, settings: Settings | None = None) -> int:
    with get_engine(settings).begin() as conn:
        return conn.execute(text(sql), dict(params or {})).rowcount
